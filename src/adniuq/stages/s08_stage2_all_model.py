from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..experiments import collect_comparison
from ..features import MODEL_META, STAGE1_COMBINED_META, covered, modality_of
from ..io import upsert_row
from ..modeling import (
    confidence_cells,
    evaluate,
    fit_clg,
    plot_matrix,
    predict_proba,
    preprocess,
    report,
    result_row,
    select_lam,
)
from ..paths import (
    COMBINED_ALL_RESULTS_CSV,
    COMBINED_RESULTS_CSV,
    INDIVIDUAL_RESULTS_CSV,
    PIPELINE_ALL_DIR,
    PIPELINE_ALL_RESULTS_CSV,
    STAGE2_ALL_CSV,
    STAGE2_ALL_FEATURE_SELECTION_XLSX,
    STAGE2_ALL_PIPELINE_MATRIX_PNG,
    ensure_output_dirs,
    require,
)
from ..selection import write_feature_selection

META = MODEL_META | STAGE1_COMBINED_META
GRID = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0]
COMPARISON_COLS = (
    "model", "roc_auc", "auc_ci_lo", "auc_ci_hi", "pr_auc", "balanced_accuracy",
    "sensitivity", "specificity", "accuracy",
)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s08_stage2_all_model",
        description="Fit stage 2 (+CSF) and score the MMSE+MRI -> +CSF cascade.",
    )
    parser.add_argument("--train-on", default="all", choices=["all", "routed"])
    parser.add_argument("--hi", type=float, default=0.80)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    require(STAGE2_ALL_CSV)
    df = pd.read_csv(STAGE2_ALL_CSV, low_memory=False)

    y = np.where(df["y"].to_numpy() == 1, 1.0, -1.0)
    gold_all = (y > 0).astype(int)
    rid = df["RID"].to_numpy()
    is_train = (df["split"] == "train").to_numpy()
    is_test = (df["split"] == "test").to_numpy()
    routed = (df["routed_stage1"] == 1).to_numpy()

    feats, coverage, _ = covered(
        df, [c for c in df.columns if c not in META], args.min_coverage
    )
    counts = pd.Series([modality_of(c) for c in feats]).value_counts()

    test_routed = np.flatnonzero(is_test & routed)
    test_kept = np.flatnonzero(is_test & ~routed)
    test_all = np.flatnonzero(is_test)
    stage1_correct = df["stage1_correct"].to_numpy()

    print("=" * 96)
    print("STAGE 2  -  MMSE + MRI + CSF on the patients MMSE+MRI could not call")
    print("=" * 96)
    print(f"  features                : {len(feats)}  ({counts.get('MMSE', 0)} MMSE + "
          f"{counts.get('MRI', 0)} MRI + {counts.get('CSF', 0)} CSF)")
    print(f"  test: kept by stage 1   : {len(test_kept)}   "
          f"(stage-1 accuracy {stage1_correct[test_kept].mean():.4f})")
    print(f"  test: routed to stage 2 : {len(test_routed)}   "
          f"({int(stage1_correct[test_routed].sum())} stage 1 had right, "
          f"{int((1 - stage1_correct[test_routed]).sum())} it had wrong)   "
          f"converter rate {gold_all[test_routed].mean():.3f}")

    X = df[feats].to_numpy(dtype=float)
    rows = []
    primary = None

    populations = [args.train_on] + [p for p in ("all", "routed") if p != args.train_on]
    for population in populations:
        train_idx = (np.flatnonzero(is_train) if population == "all"
                     else np.flatnonzero(is_train & routed))
        print(f"\n{'-' * 96}\n[train-on = {population}]  {len(train_idx)} training patients, "
              f"converter rate {gold_all[train_idx].mean():.3f}")

        lam, curve = select_lam(X[train_idx], y[train_idx], GRID, k=args.folds,
                                seed=args.cv_seed)
        X_train, X_test = preprocess(X[train_idx], X[test_routed])
        clf = fit_clg(X_train, y[train_idx], lam)
        proba = predict_proba(clf, X_test)
        gold = gold_all[test_routed]

        metrics, ci = evaluate(proba, gold, args.hi, args.boot, args.boot_seed)

        if population == args.train_on:
            report(clf, proba, gold, rid[test_routed], feats, metrics,
                   "Stage 2: MMSE+MRI+CSF on routed patients", "stage2_all", args.hi,
                   PIPELINE_ALL_DIR, lam, curve, ci=ci)
            primary = (clf, proba, lam, metrics)

            pred2 = (proba >= 0.5).astype(int)
            was_wrong = stage1_correct[test_routed] == 0
            print(f"\n  rescue analysis on the {len(test_routed)} routed:")
            print(f"    stage 1 had WRONG ({int(was_wrong.sum())}): stage 2 now right "
                  f"on {int((pred2[was_wrong] == gold[was_wrong]).sum())}")
            print(f"    stage 1 had RIGHT ({int((~was_wrong).sum())}): stage 2 still "
                  f"right on {int((pred2[~was_wrong] == gold[~was_wrong]).sum())}")
        else:
            print(f"  AUC {metrics['roc_auc']:.4f}  acc {metrics['accuracy']:.4f}  "
                  f"bal.acc {metrics['balanced_accuracy']:.4f}  nz {clf.n_nonzero}")

        rows.append({
            "model": f"Stage2 MMSE+MRI+CSF (train={population})",
            "stage": "stage2",
            "train_pop": population,
            "train_n": len(train_idx),
            "n_features": len(feats),
            "n_nonzero": int(clf.n_nonzero),
            "lam": lam,
            **result_row(metrics, ci),
        })

    clf, proba, lam, metrics = primary

    fs_path, by_modality, _, _ = write_feature_selection(
        clf, feats, coverage, lam, metrics["roc_auc"], STAGE2_ALL_FEATURE_SELECTION_XLSX,
        "stage 2: MMSE + MRI + CSF on routed patients", args.min_coverage,
    )
    print(f"\n  feature selection: {int(clf.n_nonzero)} of {len(feats)} kept")
    print(by_modality.to_string(index=False))
    print(f"  -> {fs_path.name}")

    position = {row: i for i, row in enumerate(test_all)}
    cascade_proba = np.full(len(test_all), np.nan)
    for i, row in enumerate(test_routed):
        cascade_proba[position[row]] = proba[i]
    for row in test_kept:
        cascade_proba[position[row]] = df.at[row, "stage1_p_converter"]
    gold_test = gold_all[test_all]

    cascade_metrics, cascade_ci = evaluate(
        cascade_proba, gold_test, args.hi, args.boot, args.boot_seed
    )
    plot_matrix(confidence_cells(cascade_proba, gold_test, args.hi),
                "Cascade: MMSE+MRI -> (+CSF if unsure)", args.hi, len(gold_test),
                cascade_metrics["accuracy"], cascade_metrics["acc_high_conf"],
                cascade_metrics["acc_low_conf"], STAGE2_ALL_PIPELINE_MATRIX_PNG,
                auc=cascade_metrics["roc_auc"])

    lp_avoided = 100.0 * len(test_kept) / len(test_all)
    rows.append({
        "model": "CASCADE end-to-end",
        "stage": "pipeline",
        "train_pop": args.train_on,
        "train_n": np.nan,
        "n_features": len(feats),
        "n_nonzero": int(clf.n_nonzero),
        "lam": lam,
        **result_row(cascade_metrics, cascade_ci),
        "lp_avoided_pct": round(lp_avoided, 1),
    })
    for row in rows:
        upsert_row(row, PIPELINE_ALL_RESULTS_CSV)

    comparison = [
        {k: r[k] for k in COMPARISON_COLS}
        for r in collect_comparison([
            INDIVIDUAL_RESULTS_CSV, COMBINED_RESULTS_CSV, COMBINED_ALL_RESULTS_CSV,
        ])
    ]
    comparison += [{k: r[k] for k in COMPARISON_COLS} for r in rows]

    print("\n" + "=" * 100)
    print(f"COMPARISON  ({len(test_kept)} answered by MMSE+MRI, {len(test_routed)} by "
          f"MMSE+MRI+CSF; all other models see all {len(gold_test)})")
    print("=" * 100)
    print(f"  {'model':<36}{'AUC':>8}{'95% CI':>17}{'PR-AUC':>9}{'bal.acc':>9}"
          f"{'sens':>7}{'spec':>7}{'acc':>8}")
    for r in sorted(comparison, key=lambda d: -d["roc_auc"]):
        print(f"  {r['model']:<36}{r['roc_auc']:>8.4f}"
              f"  [{r['auc_ci_lo']:.3f},{r['auc_ci_hi']:.3f}]{r['pr_auc']:>9.4f}"
              f"{r['balanced_accuracy']:>9.4f}{r['sensitivity']:>7.3f}"
              f"{r['specificity']:>7.3f}{r['accuracy']:>8.4f}")

    print(f"\n  lumbar punctures avoided: {len(test_kept)}/{len(test_all)} "
          f"({lp_avoided:.1f}%)")
    print(f"\n  written:\n    confidence_matrix_stage2_all.png"
          f"\n    {STAGE2_ALL_PIPELINE_MATRIX_PNG.name}\n    stage2_all_results.xlsx"
          f"\n    {fs_path.name}\n    {PIPELINE_ALL_RESULTS_CSV.name}")


if __name__ == "__main__":
    main()
