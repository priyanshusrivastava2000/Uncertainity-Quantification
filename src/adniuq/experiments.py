from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import MODEL_META, covered, modality_of
from .io import upsert_row
from .modeling import (
    evaluate,
    fit_clg,
    predict_proba,
    preprocess,
    report,
    result_row,
    select_lam,
)
from .selection import write_feature_selection

LAMBDA_GRID = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0]

COMPARISON_COLS = (
    "model", "n_features", "n_nonzero", "roc_auc", "auc_ci_lo", "auc_ci_hi", "pr_auc",
    "balanced_accuracy", "sensitivity", "specificity", "accuracy",
)


def print_comparison(rows: list, title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    print(f"  {'model':<26}{'feat':>6}{'nz':>5}{'AUC':>8}{'95% CI':>17}{'PR-AUC':>9}"
          f"{'bal.acc':>9}{'sens':>7}{'spec':>7}{'acc':>8}")
    for r in sorted(rows, key=lambda d: -d["roc_auc"]):
        print(f"  {r['model']:<26}{r['n_features']:>6}{r['n_nonzero']:>5}"
              f"{r['roc_auc']:>8.4f}  [{r['auc_ci_lo']:.3f},{r['auc_ci_hi']:.3f}]"
              f"{r['pr_auc']:>9.4f}{r['balanced_accuracy']:>9.4f}"
              f"{r['sensitivity']:>7.3f}{r['specificity']:>7.3f}{r['accuracy']:>8.4f}")


def collect_comparison(paths) -> list:
    rows = []
    for path in paths:
        if Path(path).exists():
            for r in pd.read_csv(path).itertuples(index=False):
                rows.append({k: getattr(r, k) for k in COMPARISON_COLS})
    return rows


def run_fusion_model(
    source_csv: Path,
    out_dir: Path,
    task_label: str,
    display_name: str,
    model_name: str,
    modality_label: str,
    feature_selection_path: Path,
    results_csv: Path,
    comparison_csvs,
    args,
) -> dict:
    df = pd.read_csv(source_csv, low_memory=False)

    y = np.where(df["y"].to_numpy() == 1, 1.0, -1.0)
    rid = df["RID"].to_numpy()
    train = np.flatnonzero((df["split"] == "train").to_numpy())
    test = np.flatnonzero((df["split"] == "test").to_numpy())
    gold = (y[test] > 0).astype(int)
    majority = max(gold.mean(), 1 - gold.mean())

    feats, coverage, _ = covered(
        df, [c for c in df.columns if c not in MODEL_META], args.min_coverage
    )
    counts = pd.Series([modality_of(c) for c in feats]).value_counts()

    print("#" * 92)
    print(f"# {display_name}  ->  MCI-to-AD conversion")
    print(f"# {len(df)} patients   {len(train)} train / {len(test)} test   "
          f"converters {int((y > 0).sum())} ({100 * (y > 0).mean():.1f}%)   "
          f"majority {majority:.4f}")
    print(f"# features: {len(feats)}  ("
          + " + ".join(f"{counts.get(m, 0)} {m}" for m in ("MMSE", "MRI", "CSF")
                       if counts.get(m, 0))
          + ")")
    print("#" * 92)

    X = df[feats].to_numpy(dtype=float)
    lam, curve = select_lam(X[train], y[train], LAMBDA_GRID, k=args.folds, seed=args.cv_seed)

    X_train, X_test = preprocess(X[train], X[test])
    clf = fit_clg(X_train, y[train], lam)
    proba = predict_proba(clf, X_test)

    metrics, ci = evaluate(proba, gold, args.hi, args.boot, args.boot_seed)
    report(clf, proba, gold, rid[test], feats, metrics, display_name, task_label,
           args.hi, out_dir, lam, curve, ci=ci)

    fs_path, by_modality, _, _ = write_feature_selection(
        clf, feats, coverage, lam, metrics["roc_auc"], feature_selection_path,
        model_name, args.min_coverage,
    )
    print(f"\n  feature selection: {int(clf.n_nonzero)} of {len(feats)} kept")
    print(by_modality.to_string(index=False))
    print(f"  -> {fs_path.name}")

    row = {
        "model": model_name,
        "modality": modality_label,
        "n_features": len(feats),
        "n_nonzero": int(clf.n_nonzero),
        "lam": lam,
        "cohort_n": len(df),
        "train_n": len(train),
        "test_n": len(test),
        **result_row(metrics, ci),
    }
    upsert_row(row, results_csv)

    comparison = collect_comparison(comparison_csvs)
    comparison.append({k: row[k] for k in COMPARISON_COLS})
    print_comparison(
        comparison,
        f"COMPARISON  -  identical {len(test)} held-out patients, majority "
        f"{majority:.4f}, PR-AUC baseline {metrics['pr_auc_baseline']:.4f}",
    )
    return row
