from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..features import LABEL_COLS, all_features, covered
from ..io import write_csv, write_excel
from ..modeling import fit_clg, predict_proba, preprocess
from ..paths import (
    INDIVIDUAL_RESULTS_CSV,
    STAGE2_BASELINE_XLSX,
    STAGE2_MMSE_MRI_CSV,
    conversion_csv,
    ensure_output_dirs,
    require,
)


def stage1_probabilities(X, y, train, test, lam, folds, seed):
    proba = np.full(len(y), np.nan)
    source = np.array(["fitted"] * len(y), dtype=object)

    gold_train = (y[train] > 0).astype(int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fit_idx, hold_idx in splitter.split(X[train], gold_train):
        X_fit, X_hold = preprocess(X[train][fit_idx], X[train][hold_idx])
        clf = fit_clg(X_fit, y[train][fit_idx], lam)
        proba[train[hold_idx]] = predict_proba(clf, X_hold)
    source[train] = "out_of_fold"

    X_train, X_test = preprocess(X[train], X[test])
    proba[test] = predict_proba(fit_clg(X_train, y[train], lam), X_test)
    return proba, source


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s05_stage2_dataset",
        description="Build the MMSE -> MMSE+MRI cascade dataset with stage-1 routing.",
    )
    parser.add_argument("--hi-stage1", type=float, default=0.70,
                        help="confidence at or above which stage 1 keeps the patient")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    require(conversion_csv("mmse"), conversion_csv("mri"), INDIVIDUAL_RESULTS_CSV)

    print("=" * 90)
    print("STAGE-2 BASELINE DATASET  -  MMSE + MRI, with stage-1 routing")
    print("=" * 90)

    mmse = pd.read_csv(conversion_csv("mmse"), low_memory=False)
    mri = pd.read_csv(conversion_csv("mri"), low_memory=False)
    assert mmse["RID"].tolist() == mri["RID"].tolist(), \
        "mmse and mri conversion files are not aligned"
    print(f"  cohort: {len(mmse)} patients (MMSE and MRI files aligned)")

    results = pd.read_csv(INDIVIDUAL_RESULTS_CSV).set_index("modality")
    lam1 = float(results.at["MMSE", "lam"])

    y = np.where(mmse["y"].to_numpy() == 1, 1.0, -1.0)
    gold = (y > 0).astype(int)
    train = np.flatnonzero((mmse["split"] == "train").to_numpy())
    test = np.flatnonzero((mmse["split"] == "test").to_numpy())

    mmse_feats, _, _ = covered(mmse, all_features("mmse", mmse), args.min_coverage)
    X_mmse = mmse[mmse_feats].to_numpy(dtype=float)

    print(f"\n  stage 1 (MMSE, {len(mmse_feats)} features, lam={lam1:g})")
    p1, source = stage1_probabilities(X_mmse, y, train, test, lam1, args.folds, args.seed)
    confidence = np.maximum(p1, 1.0 - p1)
    pred1 = (p1 >= 0.5).astype(int)
    routed = (confidence < args.hi_stage1).astype(int)

    for tag, idx in (("train", train), ("test", test)):
        kept = idx[routed[idx] == 0]
        sent = idx[routed[idx] == 1]
        accuracy = (pred1[kept] == gold[kept]).mean() if len(kept) else np.nan
        print(f"    {tag}: kept {len(kept)} ({100 * len(kept) / len(idx):.1f}%) at accuracy "
              f"{accuracy:.4f}   routed {len(sent)}  (converter rate "
              f"{gold[sent].mean():.3f} vs {gold[idx].mean():.3f} overall)")

    curve_rows = []
    for threshold in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        for tag, idx in (("train", train), ("test", test)):
            keep = confidence[idx] >= threshold
            curve_rows.append({
                "threshold": threshold,
                "split": tag,
                "kept_n": int(keep.sum()),
                "kept_pct": round(100 * keep.mean(), 1),
                "acc_kept": (round(float((pred1[idx][keep] == gold[idx][keep]).mean()), 4)
                             if keep.any() else np.nan),
                "routed_n": int((~keep).sum()),
                "scans_avoided_pct": round(100 * keep.mean(), 1),
                "conv_rate_routed": (round(float(gold[idx][~keep].mean()), 3)
                                     if (~keep).any() else np.nan),
            })
    curve = pd.DataFrame(curve_rows)

    mri_feats, _, _ = covered(mri, all_features("mri", mri), args.min_coverage)
    overlap = set(mmse_feats) & set(mri_feats)
    assert not overlap, f"feature name collision between modalities: {overlap}"
    print(f"\n  stage 2 features: {len(mmse_feats)} MMSE + {len(mri_feats)} MRI = "
          f"{len(mmse_feats) + len(mri_feats)}")

    stage1 = pd.DataFrame({
        "mmse_p_converter": np.round(p1, 6),
        "mmse_confidence": np.round(confidence, 6),
        "mmse_pred": np.where(pred1 == 1, "converter", "stable"),
        "mmse_correct": (pred1 == gold).astype(int),
        "mmse_conf_source": source,
        "routed_stage1": routed,
    }, index=mmse.index)
    out = pd.concat(
        [mmse[["RID"] + LABEL_COLS], stage1, mmse[mmse_feats], mri[mri_feats]], axis=1
    )

    assert out["RID"].is_unique
    csv_path = write_csv(out, STAGE2_MMSE_MRI_CSV)

    feats = mmse_feats + mri_feats
    routing = pd.DataFrame([{
        "split": tag,
        "n": len(idx),
        "kept_by_stage1": int((routed[idx] == 0).sum()),
        "routed_to_stage2": int((routed[idx] == 1).sum()),
        "pct_routed": round(100 * routed[idx].mean(), 1),
        "stage1_acc_on_kept": (
            round(float((pred1[idx[routed[idx] == 0]] == gold[idx[routed[idx] == 0]]).mean()), 4)
            if (routed[idx] == 0).any() else np.nan
        ),
        "converter_rate_all": round(float(gold[idx].mean()), 3),
        "converter_rate_routed": round(float(gold[idx[routed[idx] == 1]].mean()), 3),
        "confidence_source": "out-of-fold" if tag == "train" else "fitted model",
    } for tag, idx in (("train", train), ("test", test))])

    coverage = pd.DataFrame([{
        "feature": column,
        "modality": "MMSE" if column in mmse_feats else "MRI",
        "non_missing": int(out[column].notna().sum()),
        "coverage_pct": round(100 * out[column].notna().mean(), 1),
    } for column in feats]).sort_values("coverage_pct")

    config = pd.DataFrame({
        "field": ["task", "cohort", "stage1", "stage1_lambda", "stage1_threshold",
                  "train_routing", "test_routing", "n_features", "mmse_features",
                  "mri_features", "split_source", "note"],
        "value": [
            "MCI -> AD conversion (no horizon)", "tri-modal, MCI at t=0",
            "MMSE-only CLG-Lasso", lam1, args.hi_stage1,
            f"{args.folds}-fold out-of-fold predictions", "fitted stage-1 model",
            len(feats), len(mmse_feats), len(mri_feats),
            "inherited from the MMSE conversion dataset",
            "MMSE max confidence is 0.738, so a 0.80 gate would route every patient and "
            "stage 1 would do nothing; 0.70 is the lowest threshold that still retains a "
            "meaningful fraction.",
        ],
    })

    xlsx = write_excel({
        "stage2_data": out,
        "routing_summary": routing,
        "threshold_curve": curve,
        "coverage": coverage,
        "run_config": config,
    }, STAGE2_BASELINE_XLSX)

    print("\n" + "=" * 90)
    print(f"DATASET: {len(out)} patients x {len(feats)} features ({out.shape[1]} columns)")
    print("=" * 90)
    print(routing.to_string(index=False))
    print("\nthreshold trade curve:")
    print(curve[curve.split == "train"].to_string(index=False))
    print(f"\nwritten:\n  {csv_path.name}\n  {xlsx.name}")


if __name__ == "__main__":
    main()
