from __future__ import annotations

import argparse
import sys

import pandas as pd

from ..features import DX_COLS, ICV_COL, LABEL_COLS
from ..io import write_csv, write_excel
from ..paths import (
    CONVERSION_LABELS_CSV,
    SPLIT_SOURCE_CSV,
    conversion_csv,
    conversion_xlsx,
    ensure_output_dirs,
    require,
    trimodal_csv,
)


def mri_spec(df: pd.DataFrame) -> tuple:
    provenance = [
        "mri_date", "mri_visit", "mri_gap_months", "mri_is_bl_coded", "mri_n_records",
        "FIELD_STRENGTH", "IMAGEUID", "OVERALLQC", "FSVER",
    ] + DX_COLS
    features = [c for c in df.columns if c.startswith("ST") and c not in provenance]

    def by_suffix(*suffixes):
        return [c for c in features if c != ICV_COL and c[-2:].upper() in suffixes]

    sets = {
        "raw": features,
        "volume": by_suffix("CV", "SV"),
        "thickness": by_suffix("TA", "TS"),
        "area": by_suffix("SA"),
    }
    return provenance, features, sets


def csf_spec(df: pd.DataFrame) -> tuple:
    provenance = [
        "csf_date", "csf_visit", "csf_gap_months", "csf_is_bl_coded", "csf_n_records",
        "BATCH", "ELE_censored_ab42", "ELE_censored_tau", "ELE_censored_ptau",
        "ELE_censored_any",
    ] + DX_COLS
    features = [c for c in df.columns if c.startswith("ELE_") and c not in provenance]
    core = ["ELE_ABETA42", "ELE_TAU", "ELE_PTAU"]
    sets = {"core": core, "atn": core + ["ELE_PTAU_ABETA42", "ELE_TAU_ABETA42"]}
    return provenance, features, sets


SPEC = {"mri": mri_spec, "csf": csf_spec}


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s03_modality_conversion",
        description="Assemble the MRI or CSF conversion dataset on the canonical split.",
    )
    parser.add_argument("--modality", required=True, choices=sorted(SPEC))
    parser.add_argument("--max-gap-months", type=float, default=None)
    args = parser.parse_args(argv)
    modality = args.modality

    ensure_output_dirs()
    source = trimodal_csv(modality)
    require(source, CONVERSION_LABELS_CSV, SPLIT_SOURCE_CSV)

    print("=" * 84)
    print(f"{modality.upper()} BASELINE DATASET  ->  MCI-to-AD CONVERSION (no horizon)")
    print("=" * 84)

    df = pd.read_csv(source, low_memory=False)
    labels = pd.read_csv(CONVERSION_LABELS_CSV)
    reference = pd.read_csv(SPLIT_SOURCE_CSV, usecols=["RID", "split", "conv_label"])
    print(f"  tri-modal {modality} table       : {len(df)}")

    mci = df[df["DX_baseline"] == "MCI"].copy()
    print(f"  MCI at t=0                : {len(mci)}")

    missing = set(mci["RID"]) - set(labels["RID"])
    if missing:
        sys.exit(f"[error] {len(missing)} MCI patients have no label row.")

    df = mci.merge(labels.drop(columns=["baseline_date", "n_dx_visits"]), on="RID", how="left")

    unknown = set(df["RID"]) - set(reference["RID"])
    if unknown:
        sys.exit(f"[error] {len(unknown)} patients are absent from the canonical split in "
                 f"{SPLIT_SOURCE_CSV.name}. The cohorts have diverged.")
    df = df.merge(reference[["RID", "split"]], on="RID", how="left")
    print(f"  labelled                  : {len(df)}   "
          f"(converter={int(df.y.sum())}, stable={int((df.y == 0).sum())}, "
          f"rate={100.0 * df.y.mean():.1f}%)")

    check = df.merge(reference[["RID", "conv_label"]], on="RID", suffixes=("", "_ref"))
    assert (check["conv_label"] == check["conv_label_ref"]).all(), \
        "label disagrees with the MMSE dataset for the same RID"
    print(f"  split read from            : {SPLIT_SOURCE_CSV.name}  "
          f"(train {int((df.split == 'train').sum())}, "
          f"test {int((df.split == 'test').sum())})")

    provenance, features, sets = SPEC[modality](df)
    provenance = [c for c in provenance if c in df.columns]

    if args.max_gap_months is not None:
        before = len(df)
        df = df[df[f"{modality}_gap_months"].abs() <= args.max_gap_months].reset_index(drop=True)
        print(f"  gap filter |gap| <= {args.max_gap_months}  : {before} -> {len(df)}"
              f"   [WARNING: cohort now differs from the other modalities]")

    df = df.sort_values("RID").reset_index(drop=True)
    columns = ["RID"] + provenance + LABEL_COLS + features
    extra = [c for c in df.columns if c not in columns]
    if extra:
        sys.exit(f"[error] unplaced columns: {extra}")
    df = df[columns]

    assert df["RID"].is_unique, "duplicate RIDs"
    assert df["conv_label"].notna().all(), "unlabelled rows survived the merge"
    assert (df["DX_baseline"] == "MCI").all(), "non-MCI row in the cohort"

    csv_path = write_csv(df, conversion_csv(modality))

    composition = (
        df.groupby(["conv_label", "split"]).size().unstack(fill_value=0)
        .assign(total=lambda d: d.sum(axis=1))
    )
    composition.loc["TOTAL"] = composition.sum()

    membership = {c: [k for k, v in sets.items() if c in v] for c in features}
    coverage = pd.DataFrame([{
        "feature": column,
        "sets": ", ".join(membership[column]) or "(none)",
        "non_missing": int(df[column].notna().sum()),
        "coverage_pct": round(100.0 * df[column].notna().mean(), 1),
    } for column in features]).sort_values("coverage_pct")

    set_info = pd.DataFrame([{
        "feature_set": name,
        "n_features": len(columns_in_set),
        "min_coverage_pct": round(100.0 * df[columns_in_set].notna().mean().min(), 1),
        "features": (", ".join(columns_in_set) if len(columns_in_set) <= 8
                     else f"{len(columns_in_set)} columns"),
    } for name, columns_in_set in sets.items()])

    summary = pd.DataFrame([{"metric": k, "value": v} for k, v in [
        ("patients", len(df)),
        ("converters", int(df.y.sum())),
        ("stable", int((df.y == 0).sum())),
        ("converter_rate", round(float(df.y.mean()), 4)),
        ("majority_baseline", round(float(max(df.y.mean(), 1 - df.y.mean())), 4)),
        ("train_n", int((df.split == "train").sum())),
        ("test_n", int((df.split == "test").sum())),
        ("test_converters", int(df.loc[df.split == "test", "y"].sum())),
        ("n_features", len(features)),
        ("median_gap_months", round(float(df[f"{modality}_gap_months"].median()), 2)),
        ("records_gt_12mo_after_t0", int((df[f"{modality}_gap_months"] > 12).sum())),
    ]])

    config = pd.DataFrame({
        "field": ["task", "horizon", "eligibility", "cohort", "split_source", "n_features",
                  "feature_sets", "gap_filter", "leakage_note"],
        "value": [
            "MCI -> AD conversion", "none - ever converted vs never",
            "DX_baseline == MCI at t=0",
            "tri-modal (MMSE + MRI + CSF present at t=0)",
            f"read by RID from {SPLIT_SOURCE_CSV.name}", len(features),
            "; ".join(f"{k} ({len(v)})" for k, v in sets.items()),
            "none" if args.max_gap_months is None
            else f"|gap| <= {args.max_gap_months} months",
            f"features are the last {len(features)} columns only; identity, provenance "
            f"and label blocks must never enter X",
        ],
    })

    xlsx = write_excel({
        f"{modality}_conversion": df,
        "label_summary": summary,
        "split_composition": composition.reset_index(),
        "feature_sets": set_info,
        "coverage": coverage,
        "run_config": config,
    }, conversion_xlsx(modality))

    print("\n" + "=" * 84)
    print(f"DATASET: {len(df)} patients x {len(features)} features "
          f"({df.shape[1]} columns total)")
    print("=" * 84)
    print(summary.to_string(index=False))
    print("\nsplit composition:")
    print(composition.to_string())
    print("\nfeature sets:")
    print(set_info.to_string(index=False))

    low = coverage[coverage.coverage_pct < 100.0]
    print(f"\ncoverage: {len(coverage) - len(low)} of {len(coverage)} features at 100%")
    if len(low):
        print(low.head(10).to_string(index=False))

    if modality == "mri" and "FIELD_STRENGTH" in df.columns:
        crosstab = pd.crosstab(df["FIELD_STRENGTH"], df["conv_label"])
        crosstab["conv_rate"] = (crosstab.get("converter", 0) / crosstab.sum(axis=1)).round(3)
        print("\nCONFOUND WATCH - conversion rate by field strength:")
        print(crosstab.to_string())

    print(f"\ncolumn blocks:  identity 1 | provenance {len(provenance)} | "
          f"label {len(LABEL_COLS)} | features {len(features)}  <- X starts at column "
          f"{df.columns.get_loc(features[0]) + 1}")
    print(f"\nwritten:\n  {csv_path.name}\n  {xlsx.name}")


if __name__ == "__main__":
    main()
