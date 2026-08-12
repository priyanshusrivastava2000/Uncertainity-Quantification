from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..features import MODEL_META, STAGE1_MMSE_META, covered
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
    INDIVIDUAL_RESULTS_CSV,
    PIPELINE_DIR,
    PIPELINE_RESULTS_CSV,
    STAGE2_FEATURE_SELECTION_XLSX,
    STAGE2_MMSE_MRI_CSV,
    STAGE2_PIPELINE_MATRIX_PNG,
    ensure_output_dirs,
    require,
)
from ..selection import write_feature_selection

META = MODEL_META | STAGE1_MMSE_META
GRID = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0]


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s05_stage2_model",
        description="Fit stage 2 (MMSE+MRI) and score the MMSE -> MMSE+MRI cascade.",
    )
    parser.add_argument("--train-on", default="routed", choices=["routed", "all"],
                        help="stage-2 training population (default: routed)")
    parser.add_argument("--hi", type=float, default=0.80)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    require(STAGE2_MMSE_MRI_CSV, INDIVIDUAL_RESULTS_CSV)
    df = pd.read_csv(STAGE2_MMSE_MRI_CSV, low_memory=False)

    y = np.where(df["y"].to_numpy() == 1, 1.0, -1.0)
    gold_all = (y > 0).astype(int)
    rid = df["RID"].to_numpy()
    is_train = (df["split"] == "train").to_numpy()
    is_test = (df["split"] == "test").to_numpy()
    routed = (df["routed_stage1"] == 1).to_numpy()

    feats, coverage, _ = covered(
        df, [c for c in df.columns if c not in META], args.min_coverage
    )

    test_routed = np.flatnonzero(is_test & routed)
    test_kept = np.flatnonzero(is_test & ~routed)
    test_all = np.flatnonzero(is_test)

    print("=" * 92)
    print("STAGE 2  -  MMSE + MRI on the patients MMSE could not call")
    print("=" * 92)
    print(f"  features                : {len(feats)}")
    print(f"  test: routed to stage 2 : {len(test_routed)}   "
          f"(converter rate {gold_all[test_routed].mean():.3f})")
    print(f"  test: kept by stage 1   : {len(test_kept)}   "
          f"(stage-1 accuracy {df.loc[test_kept, 'mmse_correct'].mean():.4f})")

    X = df[feats].to_numpy(dtype=float)
    rows = []
    primary = None

    populations = ["routed", "all"] if args.train_on == "routed" else ["all", "routed"]
    for population in populations:
        train_idx = (np.flatnonzero(is_train & routed) if population == "routed"
                     else np.flatnonzero(is_train))
        print(f"\n{'-' * 92}\n[train-on = {population}]  {len(train_idx)} training patients, "
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
                   "Stage 2: MMSE+MRI on routed patients", "stage2", args.hi,
                   PIPELINE_DIR, lam, curve, ci=ci)
            primary = (clf, proba, lam, metrics)
        else:
            print(f"  AUC {metrics['roc_auc']:.4f}  acc {metrics['accuracy']:.4f}  "
                  f"bal.acc {metrics['balanced_accuracy']:.4f}  nz {clf.n_nonzero}")

        rows.append({
            "model": f"Stage2 MMSE+MRI (train={population})",
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
        clf, feats, coverage, lam, metrics["roc_auc"], STAGE2_FEATURE_SELECTION_XLSX,
        "stage 2: MMSE + MRI", args.min_coverage,
    )
    print(f"\n  feature selection: {int(clf.n_nonzero)} of {len(feats)} kept")
    print(by_modality.to_string(index=False))
    print(f"  -> {fs_path.name}")

    position = {row: i for i, row in enumerate(test_all)}
    cascade_proba = np.full(len(test_all), np.nan)
    for i, row in enumerate(test_routed):
        cascade_proba[position[row]] = proba[i]
    for row in test_kept:
        cascade_proba[position[row]] = df.at[row, "mmse_p_converter"]
    gold_test = gold_all[test_all]

    cascade_metrics, cascade_ci = evaluate(
        cascade_proba, gold_test, args.hi, args.boot, args.boot_seed
    )
    plot_matrix(confidence_cells(cascade_proba, gold_test, args.hi),
                "Cascade: MMSE -> (MMSE+MRI if unsure)", args.hi, len(gold_test),
                cascade_metrics["accuracy"], cascade_metrics["acc_high_conf"],
                cascade_metrics["acc_low_conf"], STAGE2_PIPELINE_MATRIX_PNG,
                auc=cascade_metrics["roc_auc"])

    scans_avoided = 100.0 * len(test_kept) / len(test_all)
    rows.append({
        "model": "CASCADE end-to-end",
        "stage": "pipeline",
        "train_pop": args.train_on,
        "train_n": np.nan,
        "n_features": len(feats),
        "n_nonzero": int(clf.n_nonzero),
        "lam": lam,
        **result_row(cascade_metrics, cascade_ci),
        "scans_avoided_pct": round(scans_avoided, 1),
    })

    for row in rows:
        upsert_row(row, PIPELINE_RESULTS_CSV)

    individual = pd.read_csv(INDIVIDUAL_RESULTS_CSV)
    print("\n" + "=" * 92)
    print(f"CASCADE END-TO-END  (all {len(gold_test)} test patients: {len(test_kept)} "
          f"answered by MMSE alone, {len(test_routed)} by MMSE+MRI)")
    print("=" * 92)
    print(f"  {'model':<34}{'AUC':>8}{'95% CI':>17}{'acc':>9}{'bal.acc':>9}"
          f"{'sens':>7}{'spec':>7}{'scans':>8}")
    for r in individual.itertuples(index=False):
        print(f"  {r.model + ' (alone)':<34}{r.roc_auc:>8.4f}"
              f"  [{r.auc_ci_lo:.3f},{r.auc_ci_hi:.3f}]{r.accuracy:>9.4f}"
              f"{r.balanced_accuracy:>9.4f}{r.sensitivity:>7.3f}{r.specificity:>7.3f}"
              f"{'100%' if r.modality == 'MRI' else '-':>8}")
    for r in rows:
        scans = (f"{round(100 - r.get('scans_avoided_pct', 0), 0)}%"
                 if r["stage"] == "pipeline" else "-")
        print(f"  {r['model']:<34}{r['roc_auc']:>8.4f}"
              f"  [{r['auc_ci_lo']:.3f},{r['auc_ci_hi']:.3f}]{r['accuracy']:>9.4f}"
              f"{r['balanced_accuracy']:>9.4f}{r['sensitivity']:>7.3f}"
              f"{r['specificity']:>7.3f}{scans:>8}")

    print(f"\n  scans avoided by the cascade: {len(test_kept)}/{len(test_all)} "
          f"({scans_avoided:.1f}%) - those patients never needed an MRI")
    print(f"\n  written:\n    confidence_matrix_stage2.png"
          f"\n    {STAGE2_PIPELINE_MATRIX_PNG.name}\n    stage2_results.xlsx"
          f"\n    {fs_path.name}\n    {PIPELINE_RESULTS_CSV.name}")


if __name__ == "__main__":
    main()
