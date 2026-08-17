from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..features import MODEL_META, covered, modality_of
from ..io import write_excel
from ..modeling import compute_metrics, out_of_fold_predictions
from ..paths import (
    COMBINED_ALL_BANDS_XLSX,
    COMBINED_ALL_CSV,
    COMBINED_ALL_RESULTS_CSV,
    COMBINED_BANDS_XLSX,
    COMBINED_CSV,
    COMBINED_RESULTS_CSV,
    ensure_output_dirs,
    require,
)

MODELS = {
    "mmse_mri": {
        "display": "MMSE + MRI",
        "source": COMBINED_CSV,
        "results": COMBINED_RESULTS_CSV,
        "output": COMBINED_BANDS_XLSX,
    },
    "mmse_mri_csf": {
        "display": "MMSE + MRI + CSF",
        "source": COMBINED_ALL_CSV,
        "results": COMBINED_ALL_RESULTS_CSV,
        "output": COMBINED_ALL_BANDS_XLSX,
    },
}


def band_summary(bands: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("train", "test", "all"):
        block = bands if split == "all" else bands[bands["split"] == split]
        for band in ("high", "low", "both"):
            cell = block if band == "both" else block[block["band"] == band]
            if not len(cell):
                continue
            rows.append({
                "split": split,
                "band": band,
                "n": len(cell),
                "pct_of_split": round(100.0 * len(cell) / len(block), 1),
                "correct": int(cell["correct"].sum()),
                "incorrect": int((1 - cell["correct"]).sum()),
                "accuracy": round(float(cell["correct"].mean()), 4),
                "converters": int((cell["y"] == 1).sum()),
                "converter_rate": round(float((cell["y"] == 1).mean()), 3),
                "mean_confidence": round(float(cell["confidence"].mean()), 4),
            })
    return pd.DataFrame(rows)


def confidence_matrix(bands: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("train", "test", "all"):
        block = bands if split == "all" else bands[bands["split"] == split]
        high = block["band"] == "high"
        correct = block["correct"] == 1
        rows.append({
            "split": split,
            "n": len(block),
            "correct_high": int((correct & high).sum()),
            "correct_low": int((correct & ~high).sum()),
            "incorrect_high": int((~correct & high).sum()),
            "incorrect_low": int((~correct & ~high).sum()),
            "confidently_wrong_pct": round(100.0 * float((~correct & high).mean()), 1),
        })
    return pd.DataFrame(rows)


def run(key: str, args) -> pd.DataFrame:
    spec = MODELS[key]
    require(spec["source"], spec["results"])

    df = pd.read_csv(spec["source"], low_memory=False)
    lam = float(pd.read_csv(spec["results"]).iloc[0]["lam"])

    feats, _, _ = covered(
        df, [c for c in df.columns if c not in MODEL_META], args.min_coverage
    )
    counts = pd.Series([modality_of(c) for c in feats]).value_counts()
    X = df[feats].to_numpy(dtype=float)
    y = np.where(df["y"].to_numpy() == 1, 1.0, -1.0)
    gold = (y > 0).astype(int)
    train = np.flatnonzero((df["split"] == "train").to_numpy())
    test = np.flatnonzero((df["split"] == "test").to_numpy())

    print(f"\n{'=' * 92}")
    print(f"CONFIDENCE BANDS  -  {spec['display']}")
    print("=" * 92)
    print(f"  patients   : {len(df)}  ({len(train)} train / {len(test)} test)")
    print(f"  features   : {len(feats)}  ("
          + " + ".join(f"{counts.get(m, 0)} {m}" for m in ("MMSE", "MRI", "CSF")
                       if counts.get(m, 0)) + f")   lam={lam:g}")
    print(f"  band split : confidence >= {args.hi} is HIGH")

    fitted = out_of_fold_predictions(X, y, train, test, lam, args.folds, args.seed)
    proba, source = fitted["proba"], fitted["source"]
    confidence = np.maximum(proba, 1.0 - proba)
    pred = (proba >= 0.5).astype(int)
    high = confidence >= args.hi

    bands = pd.DataFrame({
        "RID": df["RID"],
        "split": df["split"],
        "conv_label": df["conv_label"],
        "y": df["y"],
        "p_converter": np.round(proba, 6),
        "confidence": np.round(confidence, 6),
        "band": np.where(high, "high", "low"),
        "predicted": np.where(pred == 1, "converter", "stable"),
        "correct": (pred == gold).astype(int),
        "cell": [f"{'correct' if c else 'incorrect'}-{'high' if h else 'low'}"
                 for c, h in zip(pred == gold, high)],
        "confidence_source": source,
        "time_to_conv_months": df["time_to_conv_months"],
        "followup_months": df["followup_months"],
    })

    summary = band_summary(bands)
    matrix = confidence_matrix(bands)

    test_metrics = compute_metrics(proba[test], gold[test], args.hi)
    config = pd.DataFrame({
        "field": ["model", "patients", "train_n", "test_n", "n_features", "lambda",
                  "band_threshold", "train_probabilities", "test_probabilities",
                  "test_auc", "test_accuracy", "note"],
        "value": [
            spec["display"], len(df), len(train), len(test), len(feats), lam, args.hi,
            f"{args.folds}-fold out-of-fold (seed {args.seed})",
            "model fitted on all training rows",
            round(test_metrics["roc_auc"], 4), round(test_metrics["accuracy"], 4),
            "train rows carry out-of-fold probabilities because a model's confidence "
            "about a patient it was fitted on is optimistic; test rows carry the fitted "
            "model's probability, as in deployment. Compare bands within a split, not "
            "across splits.",
        ],
    })

    path = write_excel({
        "patient_bands": bands,
        "band_summary": summary,
        "confidence_matrix": matrix,
        "run_config": config,
    }, spec["output"])

    print("\n  band composition:")
    print(summary.to_string(index=False))
    print("\n  confidence x correctness:")
    print(matrix.to_string(index=False))
    print(f"\n  -> {path}")
    return summary


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s11_confidence_bands",
        description="Export every patient's confidence band for the two fusion models.",
    )
    parser.add_argument("--model", default="both", choices=list(MODELS) + ["both"])
    parser.add_argument("--hi", type=float, default=0.80,
                        help="confidence at or above which a patient is in the HIGH band")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    keys = list(MODELS) if args.model == "both" else [args.model]
    for key in keys:
        run(key, args)


if __name__ == "__main__":
    main()
