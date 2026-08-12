from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from ..features import DOMAINS, ITEMS, LABEL_COLS, MMSE_FEATURES
from ..io import write_csv, write_excel
from ..paths import (
    CONVERSION_LABELS_CSV,
    conversion_csv,
    conversion_xlsx,
    ensure_output_dirs,
    require,
    trimodal_csv,
)

FEATURE_SETS = {"total": ["MMSCORE"], "domains": list(DOMAINS), "items": ITEMS}
IDENTITY = ["RID"]
PROVENANCE = [
    "mmse_date", "mmse_visit", "mmse_gap_months", "mmse_is_bl_coded", "mmse_n_records",
    "mmse_n_bl_rows", "mmse_bl_span_days", "DX_at_test", "DX_at_test_date",
    "DX_at_test_source", "DX_baseline",
]


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s03_mmse_conversion",
        description="Assemble the MMSE conversion dataset and fix the canonical split.",
    )
    parser.add_argument("--max-gap-months", type=float, default=None,
                        help="drop patients whose MMSE sits further than this from t=0")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    source = trimodal_csv("mmse")
    require(source, CONVERSION_LABELS_CSV)

    print("=" * 84)
    print("MMSE BASELINE DATASET  ->  MCI-to-AD CONVERSION (no horizon)")
    print("=" * 84)

    df = pd.read_csv(source, low_memory=False)
    labels = pd.read_csv(CONVERSION_LABELS_CSV)
    print(f"  tri-modal MMSE table      : {len(df)}")

    mci = df[df["DX_baseline"] == "MCI"].copy()
    print(f"  MCI at t=0                : {len(mci)}")

    missing = set(mci["RID"]) - set(labels["RID"])
    if missing:
        sys.exit(f"[error] {len(missing)} MCI patients have no label row - the baseline "
                 f"build and the label build disagree on t=0.")

    df = mci.merge(labels.drop(columns=["baseline_date", "n_dx_visits"]), on="RID", how="left")
    print(f"  labelled                  : {len(df)}   "
          f"(converter={int(df.y.sum())}, stable={int((df.y == 0).sum())}, "
          f"rate={100.0 * df.y.mean():.1f}%)")

    if args.max_gap_months is not None:
        before = len(df)
        df = df[df["mmse_gap_months"].abs() <= args.max_gap_months].reset_index(drop=True)
        print(f"  gap filter |gap| <= {args.max_gap_months}  : {before} -> {len(df)}")

    _, test_idx = train_test_split(
        df.index.to_numpy(), test_size=args.test_size,
        random_state=args.seed, stratify=df["conv_label"],
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    df = df.sort_values("RID").reset_index(drop=True)

    columns = (
        IDENTITY + [c for c in PROVENANCE if c in df.columns] + LABEL_COLS + MMSE_FEATURES
    )
    extra = [c for c in df.columns if c not in columns]
    if extra:
        sys.exit(f"[error] unplaced columns: {extra}")
    df = df[columns]

    assert df["RID"].is_unique, "duplicate RIDs"
    assert df["conv_label"].notna().all(), "unlabelled rows survived the merge"
    assert (df["DX_baseline"] == "MCI").all(), "non-MCI row in the cohort"

    csv_path = write_csv(df, conversion_csv("mmse"))

    composition = (
        df.groupby(["conv_label", "split"]).size().unstack(fill_value=0)
        .assign(total=lambda d: d.sum(axis=1))
    )
    composition.loc["TOTAL"] = composition.sum()

    coverage = pd.DataFrame([{
        "feature": column,
        "set": ("total" if column == "MMSCORE" else "domain" if column in DOMAINS else "item"),
        "non_missing": int(df[column].notna().sum()),
        "coverage_pct": round(100.0 * df[column].notna().mean(), 1),
    } for column in MMSE_FEATURES]).sort_values(["set", "coverage_pct"])

    summary = pd.DataFrame([{"metric": k, "value": v} for k, v in [
        ("patients", len(df)),
        ("converters", int(df.y.sum())),
        ("stable", int((df.y == 0).sum())),
        ("converter_rate", round(float(df.y.mean()), 4)),
        ("majority_baseline", round(float(max(df.y.mean(), 1 - df.y.mean())), 4)),
        ("train_n", int((df.split == "train").sum())),
        ("test_n", int((df.split == "test").sum())),
        ("test_converters", int(df.loc[df.split == "test", "y"].sum())),
        ("median_time_to_conv_months",
         round(float(df.loc[df.y == 1, "time_to_conv_months"].median()), 1)),
        ("median_followup_stable_months",
         round(float(df.loc[df.y == 0, "followup_months"].median()), 1)),
        ("stable_followed_lt_36mo", int(((df.y == 0) & (df.followup_months < 36)).sum())),
        ("reverters", int(df.reverted.sum())),
    ]])

    config = pd.DataFrame({
        "field": ["task", "horizon", "eligibility", "cohort", "label_source", "n_features",
                  "feature_sets", "split", "seed", "gap_filter", "leakage_note"],
        "value": [
            "MCI -> AD conversion", "none - ever converted vs never",
            "DX_baseline == MCI at t=0",
            "tri-modal (MMSE + MRI + CSF present at t=0)",
            CONVERSION_LABELS_CSV.name, len(MMSE_FEATURES),
            "; ".join(f"{k} ({len(v)})" for k, v in FEATURE_SETS.items()),
            f"{100 * (1 - args.test_size):.0f}/{100 * args.test_size:.0f} stratified on "
            f"conv_label", args.seed,
            "none" if args.max_gap_months is None
            else f"|gap| <= {args.max_gap_months} months",
            f"features are the last {len(MMSE_FEATURES)} columns only; identity, "
            f"provenance and label blocks must never enter X",
        ],
    })

    xlsx = write_excel({
        "mmse_conversion": df,
        "label_summary": summary,
        "split_composition": composition.reset_index(),
        "coverage": coverage,
        "run_config": config,
    }, conversion_xlsx("mmse"))

    print("\n" + "=" * 84)
    print(f"DATASET: {len(df)} patients x {len(MMSE_FEATURES)} features "
          f"({df.shape[1]} columns total)")
    print("=" * 84)
    print(summary.to_string(index=False))
    print("\nsplit composition:")
    print(composition.to_string())

    print("\nfeature coverage:")
    partial = coverage[coverage.coverage_pct < 100.0]
    print(f"  {len(coverage) - len(partial)} of {len(coverage)} features at 100%")
    if len(partial):
        print("  below 100%:")
        print(partial.to_string(index=False))

    print("\ncolumn blocks:")
    print(f"  identity   {len(IDENTITY):>3}  {IDENTITY}")
    print(f"  provenance {len([c for c in PROVENANCE if c in df.columns]):>3}")
    print(f"  label      {len(LABEL_COLS):>3}  {LABEL_COLS}")
    print(f"  features   {len(MMSE_FEATURES):>3}  <- X starts at column "
          f"{df.columns.get_loc('MMSCORE') + 1}")

    print(f"\nwritten:\n  {csv_path.name}\n  {xlsx.name}")


if __name__ == "__main__":
    main()
