from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..features import DISPLAY, all_features, covered
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

GRIDS = {
    "mmse": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "csf": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "mri": [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0],
}


def run(modality: str, args) -> dict:
    path = conversion_csv(modality)
    require(path)
    df = pd.read_csv(path, low_memory=False)

    y = np.where(df["y"].to_numpy() == 1, 1.0, -1.0)
    rid = df["RID"].to_numpy()
    train = np.flatnonzero((df["split"] == "train").to_numpy())
    test = np.flatnonzero((df["split"] == "test").to_numpy())
    gold = (y[test] > 0).astype(int)
    majority = max(gold.mean(), 1 - gold.mean())

    columns = all_features(modality, df)
    feats, _, dropped = covered(df, columns, args.min_coverage)

    print("\n" + "#" * 88)
    print(f"# {DISPLAY[modality]}  ->  MCI-to-AD conversion")
    print(f"# {len(df)} patients   {len(train)} train / {len(test)} test   "
          f"converters {int((y > 0).sum())} ({100 * (y > 0).mean():.1f}%)   "
          f"majority {majority:.4f}")
    print(f"# features: {len(feats)} of {len(columns)} columns "
          f"(coverage >= {args.min_coverage:.0%})")
    if dropped:
        print("#   dropped for coverage: "
              + ", ".join(f"{c} ({p}%)" for c, p in dropped))
    print("#" * 88)

    X = df[feats].to_numpy(dtype=float)
    lam, curve = select_lam(X[train], y[train], GRIDS[modality], k=args.folds,
                            seed=args.cv_seed)

    X_train, X_test = preprocess(X[train], X[test])
    clf = fit_clg(X_train, y[train], lam)
    proba = predict_proba(clf, X_test)

    metrics, ci = evaluate(proba, gold, args.hi, args.boot, args.boot_seed)
    report(clf, proba, gold, rid[test], feats, metrics,
           f"{DISPLAY[modality]} -> MCI-to-AD conversion", modality, args.hi,
           INDIVIDUAL_DIR, lam, curve, ci=ci)

    row = {
        "model": DISPLAY[modality],
        "modality": DISPLAY[modality],
        "n_columns": len(columns),
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
        description="Fit one CLG-Lasso conversion classifier per modality.",
    )
    parser.add_argument("--modality", default="all", choices=["mmse", "mri", "csf", "all"])
    parser.add_argument("--hi", type=float, default=0.80)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    modalities = ["mmse", "mri", "csf"] if args.modality == "all" else [args.modality]
    rows = [run(modality, args) for modality in modalities]

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    print("\n" + "=" * 96)
    print(f"SUMMARY  -  {rows[0]['test_n']} held-out patients, identical across all models")
    print(f"majority baseline {rows[0]['majority_baseline']:.4f}   "
          f"PR-AUC baseline {rows[0]['pr_auc_baseline']:.4f}")
    print("=" * 96)
    print(f"  {'model':<8}{'feat':>6}{'nz':>5}{'lam':>7}{'AUC':>8}{'95% CI':>17}"
          f"{'PR-AUC':>9}{'bal.acc':>9}{'sens':>7}{'spec':>7}{'acc':>8}")
    for r in results.itertuples(index=False):
        print(f"  {r.model:<8}{r.n_features:>6}{r.n_nonzero:>5}{r.lam:>7g}"
              f"{r.roc_auc:>8.4f}  [{r.auc_ci_lo:.3f},{r.auc_ci_hi:.3f}]"
              f"{r.pr_auc:>9.4f}{r.balanced_accuracy:>9.4f}"
              f"{r.sensitivity:>7.3f}{r.specificity:>7.3f}{r.accuracy:>8.4f}")
    print(f"\n  results table -> {INDIVIDUAL_RESULTS_CSV.name}")


if __name__ == "__main__":
    main()
