from __future__ import annotations

import argparse
import sys

import pandas as pd

from ..io import write_csv, write_excel
from ..paths import (
    TRIMODAL_RIDS_CSV,
    TRIMODAL_XLSX,
    baseline_csv,
    ensure_output_dirs,
    require,
    trimodal_csv,
)

MODALITIES = ("mmse", "mri", "csf")

PRESENCE = {
    "mmse": lambda d: d["MMSCORE"].notna(),
    "mri": lambda d: d[[c for c in d.columns if c.startswith("ST")]].notna().any(axis=1),
    "csf": lambda d: d["ELE_ABETA42"].notna(),
}
PRESENCE_DESC = {
    "mmse": "MMSCORE present",
    "mri": "at least one ST* region present",
    "csf": "ELE_ABETA42 present (Elecsys core analyte)",
}


def meta_cols(modality: str) -> set:
    return {
        "RID", f"{modality}_date", f"{modality}_visit", f"{modality}_gap_months",
        f"{modality}_is_bl_coded", f"{modality}_n_records", "mmse_n_bl_rows",
        "mmse_bl_span_days", "FIELD_STRENGTH", "IMAGEUID", "OVERALLQC", "FSVER",
        "BATCH", "DX_at_test", "DX_at_test_date", "DX_at_test_source", "DX_baseline",
    }


def load(modality: str) -> pd.DataFrame:
    path = baseline_csv(modality)
    df = pd.read_csv(path, low_memory=False)
    if df["RID"].duplicated().any():
        sys.exit(f"[error] {path.name} has duplicate RIDs - it must be one row per patient.")
    return df


def main(argv: list | None = None) -> None:
    argparse.ArgumentParser(
        prog="s02_trimodal_baseline",
        description="Restrict the baseline tables to patients with MMSE + MRI + CSF.",
    ).parse_args(argv)

    ensure_output_dirs()
    require(*[baseline_csv(m) for m in MODALITIES])

    print("=" * 88)
    print("TRI-MODAL RESTRICTION  -  patients with MMSE + MRI + CSF at t=0")
    print("=" * 88)

    tables, row_ids, value_ids = {}, {}, {}
    for modality in MODALITIES:
        df = load(modality)
        tables[modality] = df
        row_ids[modality] = set(df["RID"])
        value_ids[modality] = set(df.loc[PRESENCE[modality](df), "RID"])
        print(f"  {modality:<5}{len(df):>6} rows   {len(value_ids[modality]):>6} with data   "
              f"({PRESENCE_DESC[modality]})")

    by_row = row_ids["mmse"] & row_ids["mri"] & row_ids["csf"]
    cohort = sorted(value_ids["mmse"] & value_ids["mri"] & value_ids["csf"])
    print(f"\n  intersection by row      : {len(by_row)}")
    print(f"  intersection by value    : {len(cohort)}"
          f"   <- this is the cohort ({len(by_row) - len(cohort)} row-present but "
          f"value-empty)")

    print("\n  patients lost, by the modality they lack:")
    for modality in MODALITIES:
        others = [m for m in MODALITIES if m != modality]
        have_others = value_ids[others[0]] & value_ids[others[1]]
        print(f"    has the other two but not {modality:<5}: "
              f"{len(have_others - value_ids[modality]):>5}")

    out = {}
    for modality, df in tables.items():
        subset = df[df["RID"].isin(cohort)].sort_values("RID").reset_index(drop=True)
        assert len(subset) == len(cohort), \
            f"{modality}: expected {len(cohort)} rows, got {len(subset)}"
        out[modality] = subset
        write_csv(subset, trimodal_csv(modality))

    reference = out["mmse"]["RID"].tolist()
    for modality in ("mri", "csf"):
        assert out[modality]["RID"].tolist() == reference, \
            f"{modality} RID order does not match mmse"
    print("\n  alignment check: all three files have identical RID order  [ok]")

    write_csv(pd.DataFrame({"RID": cohort}), TRIMODAL_RIDS_CSV)

    membership = pd.DataFrame([{
        "modality": modality,
        "presence_rule": PRESENCE_DESC[modality],
        "patients_in_baseline": len(tables[modality]),
        "with_data": len(value_ids[modality]),
        "in_trimodal_cohort": len(cohort),
        "pct_of_modality_retained": round(100.0 * len(cohort) / len(value_ids[modality]), 1),
    } for modality in MODALITIES])

    coverage_rows = []
    for modality, df in out.items():
        for column in [c for c in df.columns if c not in meta_cols(modality)]:
            non_missing = int(df[column].notna().sum())
            coverage_rows.append({
                "modality": modality,
                "feature": column,
                "non_missing": non_missing,
                "n_patients": len(df),
                "coverage_pct": round(100.0 * non_missing / len(df), 1),
            })
    coverage = pd.DataFrame(coverage_rows)

    timing = pd.DataFrame([{
        "modality": modality,
        "median_gap_months": round(float(df[f"{modality}_gap_months"].median()), 2),
        "min_gap": round(float(df[f"{modality}_gap_months"].min()), 1),
        "max_gap": round(float(df[f"{modality}_gap_months"].max()), 1),
        "fallback_used": int((df[f"{modality}_is_bl_coded"] == 0).sum()),
        "gap_gt_12mo_after": int((df[f"{modality}_gap_months"] > 12).sum()),
        "gap_gt_36mo_after": int((df[f"{modality}_gap_months"] > 36).sum()),
    } for modality, df in out.items()])

    sheets = {f"{modality}_trimodal": df for modality, df in out.items()}
    sheets.update({"membership": membership, "timing": timing, "coverage": coverage})
    xlsx = write_excel(sheets, TRIMODAL_XLSX)

    print("\n" + "=" * 88)
    print(f"TRI-MODAL COHORT: {len(cohort)} patients")
    print("=" * 88)
    print(membership.to_string(index=False))

    print("\nfeature counts (rows = patients, columns = features):")
    for modality, df in out.items():
        features = [c for c in df.columns if c not in meta_cols(modality)]
        print(f"  {modality:<6}{len(df):>6} patients x {len(features):>4} features")

    print("\ntiming of the retained records (months from t=0):")
    print(timing.to_string(index=False))

    print("\nDX_at_test composition (diagnosis at the visit the measurement was taken):")
    dx_at_test = pd.DataFrame([{
        "modality": modality,
        **{k: int((df["DX_at_test"] == k).sum()) for k in ("CN", "MCI", "AD")},
        "unresolved": int(df["DX_at_test"].isna().sum()),
        "differs_from_DX_baseline": int((
            df["DX_at_test"].notna()
            & df["DX_baseline"].notna()
            & (df["DX_at_test"] != df["DX_baseline"])
        ).sum()),
    } for modality, df in out.items()])
    print(dx_at_test.to_string(index=False))

    moved = out["csf"][out["csf"]["DX_at_test"] != out["csf"]["DX_baseline"]]
    if len(moved):
        print(f"\n  CSF records whose DX moved between t=0 and the draw ({len(moved)}):")
        print(moved[["RID", "csf_gap_months", "DX_baseline", "DX_at_test"]]
              .sort_values("csf_gap_months").to_string(index=False))

    low = coverage[coverage["coverage_pct"] < 60].sort_values("coverage_pct")
    print(f"\nfeatures below 60% coverage on this cohort: {len(low)}")
    if len(low):
        print(low.head(12).to_string(index=False))

    print(f"\nwritten to {TRIMODAL_XLSX.parent}:")
    for modality in out:
        print(f"  {trimodal_csv(modality).name}")
    print(f"  {TRIMODAL_RIDS_CSV.name}\n  {xlsx.name}")


if __name__ == "__main__":
    main()
