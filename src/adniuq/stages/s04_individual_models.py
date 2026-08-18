from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..features import all_features, covered, modality_of
from ..io import upsert_row
from ..modeling import (
    evaluate,
    fit_clg,
    predict_proba,
    preprocess,
    report,
    result_row,
    select_lam,
)
from ..paths import (
    INDIVIDUAL_DIR,
    INDIVIDUAL_RESULTS_CSV,
    conversion_csv,
    ensure_output_dirs,
    require,
)

NARROW_GRID = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
WIDE_GRID = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0]

# A model is one or more modalities. Fusion feature blocks are concatenated here from the
# per-modality conversion tables, which are row-aligned on RID, label and split, so the
# fusion models are fitted on exactly the same patients as the single-modality ones.
MODELS = {
    "mmse": {"display": "MMSE", "modalities": ("mmse",), "grid": NARROW_GRID},
    "mri": {"display": "MRI", "modalities": ("mri",), "grid": WIDE_GRID},
    "csf": {"display": "CSF", "modalities": ("csf",), "grid": NARROW_GRID},
    "mmse_mri": {"display": "MMSE+MRI", "modalities": ("mmse", "mri"),
                 "grid": WIDE_GRID},
    "mmse_mri_csf": {"display": "MMSE+MRI+CSF", "modalities": ("mmse", "mri", "csf"),
                     "grid": WIDE_GRID},
}


def assemble(key: str, min_coverage: float):
    """Feature block for one model, concatenated across its modalities.

    The per-modality conversion tables share a row order - same RIDs, same labels, same
    canonical split - so the blocks stack horizontally without a join, and every model
    is scored on identical held-out patients.
    """
    modalities = MODELS[key]["modalities"]
    require(*[conversion_csv(m) for m in modalities])

    frames = {m: pd.read_csv(conversion_csv(m), low_memory=False) for m in modalities}
    base = frames[modalities[0]]
    for m in modalities[1:]:
        assert frames[m]["RID"].tolist() == base["RID"].tolist(), (
            f"conversion tables are not aligned on RID: {modalities[0]} vs {m}"
        )
        assert frames[m]["y"].tolist() == base["y"].tolist(), (
            f"labels disagree between conversion tables: {modalities[0]} vs {m}"
        )

    feats, blocks, coverages, n_columns, dropped = [], [], [], 0, []
    for m in modalities:
        df = frames[m]
        columns = all_features(m, df)
        kept, coverage, missing = covered(df, columns, min_coverage)
        overlap = set(kept) & set(feats)
        assert not overlap, f"feature name collision across modalities: {overlap}"
        feats += kept
        blocks.append(df[kept].to_numpy(dtype=float))
        coverages.append(coverage[kept])
        n_columns += len(columns)
        dropped += missing

    return base, np.hstack(blocks), feats, pd.concat(coverages), n_columns, dropped


def run(key: str, args) -> dict:
    spec = MODELS[key]
    display = spec["display"]
    df, X, feats, coverage, n_columns, dropped = assemble(key, args.min_coverage)

    y = np.where(df["y"].to_numpy() == 1, 1.0, -1.0)
    rid = df["RID"].to_numpy()
    train = np.flatnonzero((df["split"] == "train").to_numpy())
    test = np.flatnonzero((df["split"] == "test").to_numpy())
    gold = (y[test] > 0).astype(int)
    majority = max(gold.mean(), 1 - gold.mean())

    counts = pd.Series([modality_of(c) for c in feats]).value_counts()
    breakdown = " + ".join(
        f"{counts.get(m, 0)} {m}" for m in ("MMSE", "MRI", "CSF") if counts.get(m, 0)
    )

    print("\n" + "#" * 88)
    print(f"# {display}  ->  MCI-to-AD conversion")
    print(f"# {len(df)} patients   {len(train)} train / {len(test)} test   "
          f"converters {int((y > 0).sum())} ({100 * (y > 0).mean():.1f}%)   "
          f"majority {majority:.4f}")
    print(f"# features: {len(feats)} of {n_columns} columns "
          f"(coverage >= {args.min_coverage:.0%})   [{breakdown}]")
    if dropped:
        print("#   dropped for coverage: "
              + ", ".join(f"{c} ({p}%)" for c, p in dropped))
    print("#" * 88)

    lam, curve = select_lam(X[train], y[train], spec["grid"], k=args.folds,
                            seed=args.cv_seed)

    X_train, X_test = preprocess(X[train], X[test])
    clf = fit_clg(X_train, y[train], lam)
    proba = predict_proba(clf, X_test)

    metrics, ci = evaluate(proba, gold, args.hi, args.boot, args.boot_seed)
    report(clf, proba, gold, rid[test], feats, metrics,
           f"{display} -> MCI-to-AD conversion", key, args.hi,
           INDIVIDUAL_DIR, lam, curve, ci=ci)

    row = {
        "model": display,
        "modality": display,
        "n_columns": n_columns,
        "n_features": len(feats),
        "n_nonzero": int(clf.n_nonzero),
        "lam": lam,
        "cohort_n": len(df),
        "train_n": len(train),
        "test_n": len(test),
        **result_row(metrics, ci),
    }
    upsert_row(row, INDIVIDUAL_RESULTS_CSV)
    return row


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s04_individual_models",
        description="Fit one CLG-Lasso conversion classifier per modality and per "
                    "fusion of modalities, all on the same canonical split.",
    )
    parser.add_argument("--model", "--modality", dest="model", default="all",
                        choices=[*MODELS, "all", "single", "fusion"],
                        help="'single' is the three modalities alone, 'fusion' the two "
                             "combined models, 'all' is every model (default)")
    parser.add_argument("--hi", type=float, default=0.80)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    groups = {
        "all": list(MODELS),
        "single": ["mmse", "mri", "csf"],
        "fusion": ["mmse_mri", "mmse_mri_csf"],
    }
    keys = groups.get(args.model, [args.model])
    rows = [run(key, args) for key in keys]

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    print("\n" + "=" * 104)
    print(f"SUMMARY  -  {rows[0]['test_n']} held-out patients, identical across all models")
    print(f"majority baseline {rows[0]['majority_baseline']:.4f}   "
          f"PR-AUC baseline {rows[0]['pr_auc_baseline']:.4f}")
    print("=" * 104)
    print(f"  {'model':<16}{'feat':>6}{'nz':>5}{'lam':>7}{'AUC':>8}{'95% CI':>17}"
          f"{'PR-AUC':>9}{'bal.acc':>9}{'sens':>7}{'spec':>7}{'acc':>8}")
    for r in results.itertuples(index=False):
        print(f"  {r.model:<16}{r.n_features:>6}{r.n_nonzero:>5}{r.lam:>7g}"
              f"{r.roc_auc:>8.4f}  [{r.auc_ci_lo:.3f},{r.auc_ci_hi:.3f}]"
              f"{r.pr_auc:>9.4f}{r.balanced_accuracy:>9.4f}"
              f"{r.sensitivity:>7.3f}{r.specificity:>7.3f}{r.accuracy:>8.4f}")
    print(f"\n  results table -> {INDIVIDUAL_RESULTS_CSV.name}")


if __name__ == "__main__":
    main()
