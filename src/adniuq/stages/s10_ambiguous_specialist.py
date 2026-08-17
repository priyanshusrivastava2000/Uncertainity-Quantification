from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from ..features import MODEL_META, STAGE1_COMBINED_META, covered, modality_of
from ..io import write_csv, write_excel
from ..modeling import (
    compute_metrics,
    confidence_cells,
    fit_clg,
    plot_matrix,
    predict_proba,
    preprocess,
    select_lam,
)
from ..paths import (
    PIPELINE_ALL_RESULTS_CSV,
    STAGE2_ALL_CSV,
    ensure_output_dirs,
    require,
    specialist_dir,
    specialist_file,
)
from ..selection import write_feature_selection
from .s09_cross_validation import plot_distribution

META = MODEL_META | STAGE1_COMBINED_META
GRID = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0]

COHORTS = {
    "test-routed": {
        "tag": "test_routed",
        "label": "stage-1 routed, test side of the canonical split",
        "test_only": True,
    },
    "all-routed": {
        "tag": "all_routed",
        "label": "stage-1 routed, both sides of the canonical split",
        "test_only": False,
    },
}

REPORTED = [
    "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity",
    "specificity", "f1_converter", "confidently_wrong", "n_nonzero", "lam",
]


def load_cohort(cohort: str, min_coverage: float):
    require(STAGE2_ALL_CSV)
    df = pd.read_csv(STAGE2_ALL_CSV, low_memory=False)

    routed = (df["routed_stage1"] == 1).to_numpy()
    if COHORTS[cohort]["test_only"]:
        routed &= (df["split"] == "test").to_numpy()
    selected = df[routed].reset_index(drop=True)

    feats, coverage, dropped = covered(
        selected, [c for c in selected.columns if c not in META], min_coverage
    )
    X = selected[feats].to_numpy(dtype=float)
    y = np.where(selected["y"].to_numpy() == 1, 1.0, -1.0)
    return selected, X, y, feats, coverage, dropped


def fit_predict(X, y, train, test, folds, seed):
    lam, _ = select_lam(X[train], y[train], GRID, k=folds, seed=seed, verbose=False)
    X_train, X_test = preprocess(X[train], X[test])
    clf = fit_clg(X_train, y[train], lam)
    return clf, predict_proba(clf, X_test), lam


def run_holdout(X, y, gold, args) -> list:
    rows = []
    print(f"\n[holdout]  {args.repeats} x stratified "
          f"{100 * (1 - args.test_size):.0f}/{100 * args.test_size:.0f} within the cohort")
    splitter = StratifiedShuffleSplit(
        n_splits=args.repeats, test_size=args.test_size, random_state=args.seed
    )
    for i, (train, test) in enumerate(splitter.split(X, gold), start=1):
        clf, proba, lam = fit_predict(X, y, train, test, args.inner_folds, args.seed + i)
        metrics = compute_metrics(proba, gold[test], args.hi)
        rows.append({
            "scheme": "holdout",
            "split": i,
            "train_n": len(train),
            "test_n": len(test),
            "lam": lam,
            "n_nonzero": int(clf.n_nonzero),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        })
        print(f"    split {i:>2}/{args.repeats}   train {len(train)} / test {len(test)}   "
              f"lam={lam:<5g} nz={clf.n_nonzero:<4} AUC={metrics['roc_auc']:.4f}   "
              f"acc={metrics['accuracy']:.4f}")
    return rows


def run_kfold(X, y, gold, args):
    rows = []
    oof = np.full((args.kfold_repeats, len(gold)), np.nan)
    lams = []
    print(f"\n[kfold]  {args.kfold_repeats} x stratified {args.folds}-fold, every patient "
          f"predicted while held out")

    for r in range(args.kfold_repeats):
        splitter = StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed + r
        )
        repeat_lams = []
        for train, test in splitter.split(X, gold):
            clf, proba, lam = fit_predict(X, y, train, test, args.inner_folds, args.seed + r)
            oof[r, test] = proba
            repeat_lams.append(lam)

        metrics = compute_metrics(oof[r], gold, args.hi)
        lams += repeat_lams
        rows.append({
            "scheme": "kfold",
            "split": r + 1,
            "train_n": int(len(gold) * (args.folds - 1) / args.folds),
            "test_n": len(gold),
            "lam": float(np.median(repeat_lams)),
            "n_nonzero": np.nan,
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        })
        print(f"    repeat {r + 1}/{args.kfold_repeats}   lam(median)="
              f"{np.median(repeat_lams):<5g} AUC={metrics['roc_auc']:.4f}   "
              f"acc={metrics['accuracy']:.4f}   (all {len(gold)} patients out-of-fold)")

    return rows, oof, lams


def rescue_table(pred, gold, stage1_correct) -> pd.DataFrame:
    correct = pred == gold
    rows = []
    for label, mask in (
        ("stage 1 was WRONG", stage1_correct == 0),
        ("stage 1 was RIGHT", stage1_correct == 1),
        ("all routed", np.ones(len(gold), dtype=bool)),
    ):
        rows.append({
            "group": label,
            "n": int(mask.sum()),
            "specialist_right": int(correct[mask].sum()),
            "specialist_wrong": int((~correct[mask]).sum()),
            "specialist_accuracy": round(float(correct[mask].mean()), 4),
            "converter_rate": round(float(gold[mask].mean()), 3),
        })
    return pd.DataFrame(rows)


def band_movement(bands: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for band in ("high", "low"):
        cell = bands[bands["specialist_band"] == band]
        if not len(cell):
            rows.append({"specialist_band": band, "n": 0, "pct_of_cohort": 0.0})
            continue
        rows.append({
            "specialist_band": band,
            "n": len(cell),
            "pct_of_cohort": round(100.0 * len(cell) / len(bands), 1),
            "specialist_correct": int(cell["specialist_correct"].sum()),
            "specialist_accuracy": round(float(cell["specialist_correct"].mean()), 4),
            "stage1_would_be_correct": int(cell["stage1_correct"].sum()),
            "stage1_accuracy": round(float(cell["stage1_correct"].mean()), 4),
            "converters": int((cell["y"] == 1).sum()),
            "converter_rate": round(float((cell["y"] == 1).mean()), 3),
            "mean_confidence": round(float(cell["specialist_confidence"].mean()), 4),
        })
    return pd.DataFrame(rows)


def summarise(splits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme in splits["scheme"].unique():
        block = splits[splits["scheme"] == scheme]
        row = {
            "scheme": scheme,
            "n_splits": len(block),
            "train_n": int(block["train_n"].iloc[0]),
            "test_n": int(block["test_n"].iloc[0]),
        }
        for metric in REPORTED:
            values = block[metric].to_numpy(dtype=float)
            if np.isnan(values).all():
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_sd"] = np.nan
                continue
            row[f"{metric}_mean"] = round(float(np.nanmean(values)), 4)
            row[f"{metric}_sd"] = round(float(np.nanstd(values, ddof=1)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s10_ambiguous_specialist",
        description="Train a tri-modal specialist on the patients MMSE+MRI could not call.",
    )
    parser.add_argument("--cohort", default="test-routed", choices=list(COHORTS),
                        help="which routed patients to train on")
    parser.add_argument("--repeats", type=int, default=20,
                        help="resampled holdout splits within the cohort")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--folds", type=int, default=5, help="outer folds for the k-fold run")
    parser.add_argument("--kfold-repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5,
                        help="folds of the lambda search inside every training set")
    parser.add_argument("--hi", type=float, default=0.80)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    tag = COHORTS[args.cohort]["tag"]
    out_dir = specialist_dir(tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort, X, y, feats, coverage, dropped = load_cohort(args.cohort, args.min_coverage)
    gold = (y > 0).astype(int)
    stage1_correct = cohort["stage1_correct"].to_numpy()
    counts = pd.Series([modality_of(c) for c in feats]).value_counts()

    print("=" * 92)
    print("AMBIGUOUS-PATIENT SPECIALIST  -  MMSE + MRI + CSF on the stage-1 routed cohort")
    print("=" * 92)
    print(f"  cohort            : {args.cohort}  ({COHORTS[args.cohort]['label']})")
    print(f"  patients          : {len(cohort)}   "
          + "  ".join(f"{s}={int((cohort['split'] == s).sum())}"
                      for s in ("train", "test")))
    print(f"    stage 1 had right : {int(stage1_correct.sum())}")
    print(f"    stage 1 had wrong : {int((1 - stage1_correct).sum())}")
    print(f"  converters        : {int(gold.sum())} ({100 * gold.mean():.1f}%)   "
          f"majority baseline {max(gold.mean(), 1 - gold.mean()):.4f}")
    print(f"  features          : {len(feats)}  ({counts.get('MMSE', 0)} MMSE + "
          f"{counts.get('MRI', 0)} MRI + {counts.get('CSF', 0)} CSF)")
    if dropped:
        print(f"  dropped for coverage on this subset: {len(dropped)}")

    holdout_rows = run_holdout(X, y, gold, args)
    kfold_rows, oof, kfold_lams = run_kfold(X, y, gold, args)

    splits = pd.DataFrame(holdout_rows + kfold_rows)
    summary = summarise(splits)

    # bands come from a single out-of-fold pass: averaging repeats would shrink every
    # probability toward 0.5 and understate how many patients reach the HIGH band
    band_proba = oof[0]
    mean_proba = oof.mean(axis=0)
    band_pred = (band_proba >= 0.5).astype(int)
    band_conf = np.maximum(band_proba, 1.0 - band_proba)
    high = band_conf >= args.hi

    band_metrics = compute_metrics(band_proba, gold, args.hi)
    mean_metrics = compute_metrics(mean_proba, gold, args.hi)
    stage1_proba = cohort["stage1_p_converter"].to_numpy()
    stage1_metrics = compute_metrics(stage1_proba, gold, args.hi)
    rescue = rescue_table(band_pred, gold, stage1_correct)

    bands = pd.DataFrame({
        "RID": cohort["RID"],
        "split": cohort["split"],
        "conv_label": cohort["conv_label"],
        "y": cohort["y"],
        "stage1_p_converter": np.round(stage1_proba, 6),
        "stage1_confidence": np.round(cohort["stage1_confidence"].to_numpy(), 6),
        "stage1_band": "low",
        "stage1_pred": cohort["stage1_pred"],
        "stage1_correct": stage1_correct,
        "stage1_conf_source": cohort["stage1_conf_source"],
        "specialist_p_converter": np.round(band_proba, 6),
        "specialist_p_mean_oof": np.round(mean_proba, 6),
        "specialist_confidence": np.round(band_conf, 6),
        "specialist_band": np.where(high, "high", "low"),
        "specialist_pred": np.where(band_pred == 1, "converter", "stable"),
        "specialist_correct": (band_pred == gold).astype(int),
        "moved_to_high": high.astype(int),
        "rescued": ((stage1_correct == 0) & (band_pred == gold)).astype(int),
        "broken": ((stage1_correct == 1) & (band_pred != gold)).astype(int),
        "cell": [f"{'correct' if c else 'incorrect'}-{'high' if h else 'low'}"
                 for c, h in zip(band_pred == gold, high)],
    }).sort_values("specialist_p_converter", ascending=False)

    movement = band_movement(bands)
    moved = int(high.sum())

    by_split = pd.DataFrame([{
        "split": split,
        "n": int((cohort["split"] == split).sum()),
        "moved_to_high": int(high[(cohort["split"] == split).to_numpy()].sum()),
        "pct_moved": round(100.0 * float(high[(cohort["split"] == split).to_numpy()].mean()), 1),
    } for split in ("train", "test") if (cohort["split"] == split).any()])

    plot_matrix(confidence_cells(band_proba, gold, args.hi),
                f"Specialist on {len(cohort)} routed patients (out-of-fold)", args.hi,
                len(gold), band_metrics["accuracy"], band_metrics["acc_high_conf"],
                band_metrics["acc_low_conf"],
                specialist_file(tag, "confidence_matrix_specialist.png"),
                auc=band_metrics["roc_auc"])

    labelled = splits.assign(model=splits["scheme"].map(
        {"holdout": f"holdout {100 * (1 - args.test_size):.0f}/"
                    f"{100 * args.test_size:.0f}",
         "kfold": f"{args.folds}-fold out-of-fold"}))
    order = labelled.groupby("model")["roc_auc"].mean().sort_values(
        ascending=False).index.tolist()
    plot_distribution(labelled, order, specialist_file(tag, "specialist_auc_distribution.png"))

    modal_lam = float(pd.Series(kfold_lams).mode().iloc[0])
    X_all, _ = preprocess(X, X)
    final_clf = fit_clg(X_all, y, modal_lam)
    fs_path, by_modality, _, _ = write_feature_selection(
        final_clf, feats, coverage, modal_lam, band_metrics["roc_auc"],
        specialist_file(tag, "specialist_feature_selection.xlsx"),
        f"specialist: MMSE + MRI + CSF on {len(cohort)} routed patients", args.min_coverage,
    )

    config = pd.DataFrame({
        "field": ["cohort", "cohort_definition", "patients", "stage1_right",
                  "stage1_wrong", "converters", "n_features", "holdout_splits", "kfold",
                  "inner_folds", "seed", "band_threshold", "band_probabilities",
                  "final_fit_lambda", "moved_to_high", "caveat_comparability",
                  "caveat_final_fit"],
        "value": [
            args.cohort, COHORTS[args.cohort]["label"], len(cohort),
            int(stage1_correct.sum()), int((1 - stage1_correct).sum()),
            f"{int(gold.sum())} ({100 * gold.mean():.1f}%)", len(feats),
            f"{args.repeats} x stratified {100 * (1 - args.test_size):.0f}/"
            f"{100 * args.test_size:.0f}",
            f"{args.kfold_repeats} x stratified {args.folds}-fold",
            args.inner_folds, args.seed, args.hi,
            "single out-of-fold pass (repeat 1); the repeat-averaged probability is "
            "carried alongside as specialist_p_mean_oof but is not used for banding, "
            "because averaging shrinks confidence toward 0.5",
            modal_lam, f"{moved} of {len(cohort)} ({100.0 * moved / len(cohort):.1f}%)",
            "every patient here was in the LOW band of the stage-1 MMSE+MRI model by "
            "construction. Patients on the test side were the held-out set of the "
            "upstream models, so specialist scores are not comparable to the project's "
            "headline numbers",
            f"the feature-selection workbook refits on all {len(cohort)} patients at "
            f"lambda={modal_lam:g} for inspection only - in-sample, no held-out score",
        ],
    })

    write_csv(splits, specialist_file(tag, "specialist_splits.csv"))
    write_csv(summary, specialist_file(tag, "specialist_summary.csv"))
    write_csv(bands, specialist_file(tag, "specialist_oof_predictions.csv"))
    xlsx = write_excel({
        "patient_bands": bands,
        "band_movement": movement,
        "moved_by_split": by_split,
        "summary": summary,
        "per_split": splits,
        "rescue_analysis": rescue,
        "run_config": config,
    }, specialist_file(tag, "specialist_confidence_bands.xlsx"))

    print("\n" + "=" * 92)
    print(f"RESULTS  -  specialist on the {len(cohort)} ambiguous patients")
    print("=" * 92)
    print(f"  {'protocol':<26}{'splits':>8}{'AUC':>17}{'bal.acc':>17}{'accuracy':>17}")
    for r in summary.itertuples(index=False):
        name = ("holdout 70/30" if r.scheme == "holdout"
                else f"{args.folds}-fold out-of-fold")
        print(f"  {name:<26}{r.n_splits:>8}"
              f"{f'{r.roc_auc_mean:.3f}+-{r.roc_auc_sd:.3f}':>17}"
              f"{f'{r.balanced_accuracy_mean:.3f}+-{r.balanced_accuracy_sd:.3f}':>17}"
              f"{f'{r.accuracy_mean:.3f}+-{r.accuracy_sd:.3f}':>17}")

    print(f"\n  out-of-fold over all {len(cohort)} patients (single pass, used for bands):")
    print(f"    AUC {band_metrics['roc_auc']:.4f}   acc {band_metrics['accuracy']:.4f}   "
          f"bal.acc {band_metrics['balanced_accuracy']:.4f}")
    print(f"    repeat-averaged probabilities: AUC {mean_metrics['roc_auc']:.4f}   "
          f"acc {mean_metrics['accuracy']:.4f}")
    print(f"    stage 1 (MMSE+MRI) on the same patients: "
          f"AUC {stage1_metrics['roc_auc']:.4f}   acc {stage1_metrics['accuracy']:.4f}")

    print(f"\n  BAND MOVEMENT  -  all {len(cohort)} were LOW confidence under MMSE+MRI:")
    print(f"    moved to HIGH (>= {args.hi}) under the specialist: {moved} "
          f"({100.0 * moved / len(cohort):.1f}%)")
    print(f"    still LOW: {len(cohort) - moved}")
    print(movement.to_string(index=False))
    if len(by_split) > 1:
        print("\n  movement by side of the canonical split:")
        print(by_split.to_string(index=False))

    print("\n  rescue analysis (out-of-fold specialist vs what stage 1 had):")
    print(rescue.to_string(index=False))

    if COHORTS[args.cohort]["test_only"] and PIPELINE_ALL_RESULTS_CSV.exists():
        stage2 = pd.read_csv(PIPELINE_ALL_RESULTS_CSV)
        stage2 = stage2[stage2["stage"] == "stage2"]
        print("\n  for reference - stage 08 scored these same patients as a TEST set "
              "(trained on the other 543):")
        for r in stage2.itertuples(index=False):
            print(f"    {r.model:<40}AUC {r.roc_auc:.4f}   acc {r.accuracy:.4f}")

    print(f"\n  feature selection at lambda={modal_lam:g} (in-sample, descriptive):")
    print(by_modality.to_string(index=False))

    print(f"\n  written to {out_dir}:")
    for name in ("specialist_splits.csv", "specialist_summary.csv",
                 "specialist_oof_predictions.csv", xlsx.name, fs_path.name,
                 "confidence_matrix_specialist.png", "specialist_auc_distribution.png"):
        print(f"    {name}")


if __name__ == "__main__":
    main()
