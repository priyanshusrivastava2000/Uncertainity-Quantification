from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..features import LABEL_COLS, MODEL_META, covered, modality_of
from ..io import write_csv, write_excel
from ..modeling import fit_clg, predict_proba, preprocess
from ..paths import (
    COMBINED_ALL_CSV,
    COMBINED_RESULTS_CSV,
    STAGE2_ALL_BASELINE_XLSX,
    STAGE2_ALL_CSV,
    ensure_output_dirs,
    require,
)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s08_stage2_all_dataset",
        description="Build the MMSE+MRI -> MMSE+MRI+CSF cascade dataset with routing.",
    )
    parser.add_argument("--hi-stage1", type=float, default=0.80)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    require(COMBINED_ALL_CSV, COMBINED_RESULTS_CSV)

    print("=" * 92)
    print("STAGE-2 DATASET  -  MMSE+MRI (stage 1)  ->  MMSE+MRI+CSF (stage 2)")
    print("=" * 92)

    df = pd.read_csv(COMBINED_ALL_CSV, low_memory=False)
    lam1 = float(pd.read_csv(COMBINED_RESULTS_CSV).iloc[0]["lam"])

    feats_all, _, _ = covered(
        df, [c for c in df.columns if c not in MODEL_META], args.min_coverage
    )
    feats_stage1 = [c for c in feats_all if not c.startswith("ELE_")]
    csf_feats = [c for c in feats_all if c.startswith("ELE_")]

    print(f"  cohort              : {len(df)} patients")
    print(f"  stage-1 features    : {len(feats_stage1)}  (MMSE + MRI)   lam={lam1:g}")
    print(f"  stage-2 features    : {len(feats_all)}  (+{len(csf_feats)} CSF)")

    y = np.where(df["y"].to_numpy() == 1, 1.0, -1.0)
    gold = (y > 0).astype(int)
    train = np.flatnonzero((df["split"] == "train").to_numpy())
    test = np.flatnonzero((df["split"] == "test").to_numpy())
    X1 = df[feats_stage1].to_numpy(dtype=float)

    p1 = np.full(len(y), np.nan)
    source = np.array(["fitted"] * len(y), dtype=object)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fit_idx, hold_idx in splitter.split(X1[train], gold[train]):
        X_fit, X_hold = preprocess(X1[train][fit_idx], X1[train][hold_idx])
        p1[train[hold_idx]] = predict_proba(fit_clg(X_fit, y[train][fit_idx], lam1), X_hold)
    source[train] = "out_of_fold"
    X_train, X_test = preprocess(X1[train], X1[test])
    p1[test] = predict_proba(fit_clg(X_train, y[train], lam1), X_test)

    confidence = np.maximum(p1, 1.0 - p1)
    pred1 = (p1 >= 0.5).astype(int)
    routed = (confidence < args.hi_stage1).astype(int)

    print(f"\n  stage-1 routing at confidence >= {args.hi_stage1}:")
    for tag, idx in (("train", train), ("test", test)):
        kept, sent = idx[routed[idx] == 0], idx[routed[idx] == 1]
        accuracy = (pred1[kept] == gold[kept]).mean() if len(kept) else np.nan
        correct_sent = int((pred1[sent] == gold[sent]).sum())
        print(f"    {tag}: kept {len(kept)} ({100 * len(kept) / len(idx):.1f}%) at accuracy "
              f"{accuracy:.4f}   routed {len(sent)} ({correct_sent} would have been right, "
              f"{len(sent) - correct_sent} wrong)   converter rate {gold[sent].mean():.3f}")

    curve_rows = []
    for threshold in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
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
                "lp_avoided_pct": round(100 * keep.mean(), 1),
                "conv_rate_routed": (round(float(gold[idx][~keep].mean()), 3)
                                     if (~keep).any() else np.nan),
            })
    curve = pd.DataFrame(curve_rows)

    stage1 = pd.DataFrame({
        "stage1_p_converter": np.round(p1, 6),
        "stage1_confidence": np.round(confidence, 6),
        "stage1_pred": np.where(pred1 == 1, "converter", "stable"),
        "stage1_correct": (pred1 == gold).astype(int),
        "stage1_conf_source": source,
        "routed_stage1": routed,
    }, index=df.index)
    out = pd.concat([df[["RID"] + LABEL_COLS], stage1, df[feats_all]], axis=1)
    assert out["RID"].is_unique

    csv_path = write_csv(out, STAGE2_ALL_CSV)

    routing = pd.DataFrame([{
        "split": tag,
        "n": len(idx),
        "kept_by_stage1": int((routed[idx] == 0).sum()),
        "routed_to_stage2": int((routed[idx] == 1).sum()),
        "pct_routed": round(100 * routed[idx].mean(), 1),
        "stage1_acc_on_kept": round(float(
            (pred1[idx[routed[idx] == 0]] == gold[idx[routed[idx] == 0]]).mean()), 4),
        "routed_correct_by_stage1": int(
            (pred1[idx[routed[idx] == 1]] == gold[idx[routed[idx] == 1]]).sum()),
        "routed_wrong_by_stage1": int(
            (pred1[idx[routed[idx] == 1]] != gold[idx[routed[idx] == 1]]).sum()),
        "converter_rate_all": round(float(gold[idx].mean()), 3),
        "converter_rate_routed": round(float(gold[idx[routed[idx] == 1]].mean()), 3),
        "confidence_source": "out-of-fold" if tag == "train" else "fitted model",
    } for tag, idx in (("train", train), ("test", test))])

    coverage = pd.DataFrame([{
        "feature": column,
        "modality": modality_of(column),
        "stage": 1 if column in feats_stage1 else 2,
        "non_missing": int(out[column].notna().sum()),
        "coverage_pct": round(100 * out[column].notna().mean(), 1),
    } for column in feats_all]).sort_values("coverage_pct")

    config = pd.DataFrame({
        "field": ["cascade", "stage1", "stage1_features", "stage1_lambda",
                  "stage1_threshold", "stage2", "stage2_features", "train_routing",
                  "test_routing", "split_source", "rationale"],
        "value": [
            "MMSE+MRI -> MMSE+MRI+CSF", "combined MMSE + MRI CLG-Lasso",
            len(feats_stage1), lam1, args.hi_stage1,
            "combined MMSE + MRI + CSF CLG-Lasso", len(feats_all),
            f"{args.folds}-fold out-of-fold predictions", "fitted stage-1 model",
            "inherited from the conversion datasets",
            "MMSE and MRI are routinely collected; CSF needs a lumbar puncture. The "
            "cascade asks how many patients can be called without one.",
        ],
    })

    xlsx = write_excel({
        "stage2_all_data": out,
        "routing_summary": routing,
        "threshold_curve": curve,
        "coverage": coverage,
        "run_config": config,
    }, STAGE2_ALL_BASELINE_XLSX)

    print("\n" + "=" * 92)
    print(f"DATASET: {len(out)} patients x {len(feats_all)} features "
          f"({out.shape[1]} columns)")
    print("=" * 92)
    print(routing.to_string(index=False))
    print("\nthreshold trade curve (test):")
    print(curve[curve.split == "test"].to_string(index=False))
    print(f"\nwritten:\n  {csv_path.name}\n  {xlsx.name}")


if __name__ == "__main__":
    main()
