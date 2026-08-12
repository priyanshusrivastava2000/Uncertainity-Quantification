from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import DOMAINS, ICV_RAW_COL, ITEM_TO_DOMAIN, ITEMS, MRI_SUFFIX
from ..io import markdown_table, write_csv, write_excel, write_text
from ..paths import (
    AUDIT_REPORT_MD,
    BASELINE_XLSX,
    COGNITIVE_DIR,
    CSF_DIR,
    DXSUM_PATTERN,
    ELECSYS_PATTERN,
    MMSE_PATTERN,
    MRI_DIR,
    RAW_DATA_DIR,
    UCSFFSX7_PATTERN,
    baseline_csv,
    ensure_output_dirs,
    find_one,
)

BASELINE_CODES = {"sc", "scmri", "bl", "init"}
BAD_QC = {"Fail", "Hippocampus Only"}
DX_MAP = {1: "CN", 2: "MCI", 3: "AD"}
MONTHS = 365.25 / 12.0
NOT_A_REGION = {"STATUS"}


def read_ids(path: Path, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["RID"] = pd.to_numeric(df["RID"], errors="coerce")
    df = df.dropna(subset=["RID"]).copy()
    df["RID"] = df["RID"].astype(int)
    df["_date"] = pd.to_datetime(df.get(date_col), errors="coerce")
    visit = (
        df["VISCODE2"].astype(str).str.strip().str.lower()
        if "VISCODE2" in df.columns
        else pd.Series("", index=df.index)
    )
    df["_visit"] = visit
    df["_is_bl"] = visit.isin(BASELINE_CODES)
    return df


def dx_history() -> pd.DataFrame:
    dx = read_ids(find_one(DXSUM_PATTERN, RAW_DATA_DIR), "EXAMDATE")
    dx["DIAGNOSIS"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce")
    dx = dx.dropna(subset=["DIAGNOSIS", "_date"])
    dx["DX"] = dx["DIAGNOSIS"].round().map(DX_MAP)
    return (
        dx[["RID", "_visit", "_date", "_is_bl", "DX"]]
        .sort_values(["RID", "_date"])
        .reset_index(drop=True)
    )


def reference_baseline(history: pd.DataFrame) -> pd.DataFrame:
    dx = history.sort_values(["_is_bl", "_date"], ascending=[False, True]).drop_duplicates("RID")
    out = dx[["RID", "_date", "DX", "_is_bl"]].rename(columns={
        "_date": "baseline_date",
        "DX": "DX_baseline",
        "_is_bl": "dx_is_bl_coded",
    })
    out["dx_is_bl_coded"] = out["dx_is_bl_coded"].astype(int)
    return out.sort_values("RID").reset_index(drop=True)


def attach_dx_at_test(selected: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    n = len(selected)
    dx_value = pd.Series(pd.NA, index=selected.index, dtype="object")
    dx_date = pd.Series(pd.NaT, index=selected.index, dtype="datetime64[ns]")
    source = pd.Series("none", index=selected.index, dtype="object")

    key = history.dropna(subset=["_visit"]).drop_duplicates(["RID", "_visit"])
    merged = selected[["RID", "_visit"]].merge(
        key[["RID", "_visit", "DX", "_date"]], on=["RID", "_visit"], how="left"
    )
    hit = merged["DX"].notna().to_numpy()
    dx_value[hit] = merged.loc[hit, "DX"].to_numpy()
    dx_date[hit] = merged.loc[hit, "_date"].to_numpy()
    source[hit] = "visit_match"

    by_rid = {
        rid: (group["_date"].to_numpy(), group["DX"].to_numpy())
        for rid, group in history.groupby("RID")
    }
    for i in np.flatnonzero(~hit):
        rid, when = selected["RID"].iloc[i], selected["_date"].iloc[i]
        group = by_rid.get(rid)
        if group is None or pd.isna(when):
            continue
        dates, values = group
        j = int(np.searchsorted(dates, np.datetime64(when), side="left"))
        if j < len(dates):
            dx_value.iloc[i], dx_date.iloc[i], source.iloc[i] = values[j], dates[j], "next_visit"
        elif len(dates):
            dx_value.iloc[i] = values[-1]
            dx_date.iloc[i] = dates[-1]
            source.iloc[i] = "prior_visit"

    selected = selected.copy()
    selected["DX_at_test"] = dx_value.to_numpy()
    selected["DX_at_test_date"] = dx_date.to_numpy()
    selected["DX_at_test_source"] = source.to_numpy()
    resolved = int(pd.Series(dx_value).notna().sum())
    print(f"  DX_at_test resolved for {resolved}/{n}   "
          f"{pd.Series(source).value_counts().to_dict()}")
    return selected


def select_earliest(
    df: pd.DataFrame, ref: pd.DataFrame, modality: str, keep: list, history: pd.DataFrame
) -> pd.DataFrame:
    n_records = df.groupby("RID").size().rename(f"{modality}_n_records")

    baseline_rows = df[df["_is_bl"]].sort_values("_date").drop_duplicates("RID")
    rest = (
        df[~df["RID"].isin(set(baseline_rows["RID"]))]
        .sort_values("_date")
        .drop_duplicates("RID")
    )
    selected = pd.concat([baseline_rows, rest], ignore_index=True)

    selected = selected.merge(
        ref[["RID", "baseline_date", "DX_baseline"]], on="RID", how="left"
    )
    selected[f"{modality}_gap_months"] = (
        selected["_date"] - selected["baseline_date"]
    ).dt.days / MONTHS
    selected = attach_dx_at_test(selected, history)

    out = selected[["RID"] + keep].copy()
    out[f"{modality}_date"] = selected["_date"]
    out[f"{modality}_visit"] = selected["_visit"]
    out[f"{modality}_gap_months"] = selected[f"{modality}_gap_months"].round(2)
    out[f"{modality}_is_bl_coded"] = selected["_is_bl"].astype(int)
    out["DX_at_test"] = selected["DX_at_test"]
    out["DX_at_test_date"] = selected["DX_at_test_date"]
    out["DX_at_test_source"] = selected["DX_at_test_source"]
    out["DX_baseline"] = selected["DX_baseline"]
    out = out.merge(n_records, on="RID", how="left")
    return out.sort_values("RID").reset_index(drop=True)


def front_load(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    provenance = [
        f"{modality}_date", f"{modality}_visit", f"{modality}_gap_months",
        f"{modality}_is_bl_coded", f"{modality}_n_records",
        "DX_at_test", "DX_at_test_date", "DX_at_test_source", "DX_baseline",
    ]
    provenance = [c for c in provenance if c in df.columns]
    rest = [c for c in df.columns if c not in provenance + ["RID"]]
    return df[["RID"] + provenance + rest]


def build_mmse(ref: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    path = find_one(MMSE_PATTERN, COGNITIVE_DIR)
    df = read_ids(path, "VISDATE")
    print(f"  source: {path.name}  ({len(df)} visit rows, {df['RID'].nunique()} patients)")

    columns = ITEMS + ["MMSCORE"]
    for column in columns:
        values = pd.to_numeric(df.get(column), errors="coerce")
        df[column] = values.mask(values < 0)

    base = select_earliest(df, ref, "mmse", columns, history)

    baseline_rows = df[df["_is_bl"]].sort_values("_date")
    coalesced = baseline_rows.groupby("RID")[columns].first()
    span = (
        baseline_rows.groupby("RID")["_date"].max()
        - baseline_rows.groupby("RID")["_date"].min()
    ).dt.days
    n_baseline = baseline_rows.groupby("RID").size()

    base = base.set_index("RID")
    base.loc[coalesced.index, columns] = coalesced
    base["mmse_n_bl_rows"] = n_baseline.reindex(base.index).fillna(0).astype(int)
    base["mmse_bl_span_days"] = span.reindex(base.index)
    base = base.reset_index()

    for name, items in DOMAINS.items():
        complete = base[items].notna().all(axis=1)
        base[name] = base[items].sum(axis=1).where(complete)

    return front_load(base, "mmse")


def build_mri(ref: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    path = find_one(UCSFFSX7_PATTERN, MRI_DIR)
    df = read_ids(path, "EXAMDATE")
    print(f"  source: {path.name}  ({len(df)} scan rows, {df['RID'].nunique()} patients)")

    if "OVERALLQC" in df.columns:
        before = len(df)
        df = df[~df["OVERALLQC"].isin(BAD_QC)]
        print(f"  dropped {before - len(df)} rows failing QC {sorted(BAD_QC)}")

    regions = [c for c in df.columns if c.startswith("ST") and c not in NOT_A_REGION]
    for column in regions:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    meta = [c for c in ["FIELD_STRENGTH", "IMAGEUID", "OVERALLQC", "FSVER"] if c in df.columns]
    base = select_earliest(df, ref, "mri", meta + regions, history)
    if ICV_RAW_COL in base:
        base = base.rename(columns={ICV_RAW_COL: f"{ICV_RAW_COL}_ICV"})
    print(f"  {len(regions)} ST* region features kept ({ICV_RAW_COL} = IntraCranialVol)")
    return front_load(base, "mri")


def build_csf(ref: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    path = find_one(ELECSYS_PATTERN, CSF_DIR)
    df = read_ids(path, "EXAMDATE")
    print(f"  source: {path.name}  ({len(df)} draw rows, {df['RID'].nunique()} patients)")

    for column in ("ABETA42", "ABETA40", "TAU", "PTAU"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    comment = df["COMMENT"].astype(str).fillna("")
    df["censored_ab42"] = comment.str.contains(r"Abeta42\s*[<>]", case=False, regex=True).astype(int)
    df["censored_tau"] = comment.str.contains(
        r"(?:^|[^A-Za-z])Tau\s*[<>]", case=False, regex=True
    ).astype(int)
    df["censored_ptau"] = comment.str.contains(r"PTau\s*[<>]", case=False, regex=True).astype(int)
    df["censored_any"] = df[["censored_ab42", "censored_tau", "censored_ptau"]].max(axis=1)

    def ratio(numerator, denominator):
        den = pd.to_numeric(denominator, errors="coerce")
        num = pd.to_numeric(numerator, errors="coerce")
        return np.where(den > 0, num / den, np.nan)

    df["PTAU_ABETA42"] = ratio(df["PTAU"], df["ABETA42"])
    df["TAU_ABETA42"] = ratio(df["TAU"], df["ABETA42"])
    df["ABETA42_40"] = ratio(df["ABETA42"], df["ABETA40"])

    keep = [
        "ABETA42", "ABETA40", "TAU", "PTAU",
        "PTAU_ABETA42", "TAU_ABETA42", "ABETA42_40",
        "censored_ab42", "censored_tau", "censored_ptau", "censored_any", "BATCH",
    ]
    base = select_earliest(df, ref, "csf", keep, history)
    base = base.rename(columns={c: f"ELE_{c}" for c in keep if c != "BATCH"})
    return front_load(base, "csf")


def feature_cols(df: pd.DataFrame, modality: str) -> list:
    provenance = {
        f"{modality}_date", f"{modality}_visit", f"{modality}_gap_months",
        f"{modality}_is_bl_coded", f"{modality}_n_records", "mmse_n_bl_rows",
        "mmse_bl_span_days", "RID", "FIELD_STRENGTH", "IMAGEUID", "OVERALLQC",
        "FSVER", "BATCH", "DX_at_test", "DX_at_test_date", "DX_at_test_source",
        "DX_baseline",
    }
    return [c for c in df.columns if c not in provenance]


def gap_summary(tables: dict) -> pd.DataFrame:
    rows = []
    for modality, df in tables.items():
        gap = df[f"{modality}_gap_months"].dropna()
        rows.append({
            "modality": modality,
            "patients": len(df),
            "baseline_coded": int(df[f"{modality}_is_bl_coded"].sum()),
            "fallback_used": int((df[f"{modality}_is_bl_coded"] == 0).sum()),
            "median_gap_months": round(float(gap.median()), 2),
            "min_gap": round(float(gap.min()), 1),
            "max_gap": round(float(gap.max()), 1),
            "within_6mo": int((gap.abs() <= 6).sum()),
            "within_12mo": int((gap.abs() <= 12).sum()),
            "after_baseline_gt_6mo": int((gap > 6).sum()),
            "after_baseline_gt_12mo": int((gap > 12).sum()),
            "after_baseline_gt_36mo": int((gap > 36).sum()),
            "multi_record_patients": int((df[f"{modality}_n_records"] > 1).sum()),
        })
    return pd.DataFrame(rows)


def coverage_table(tables: dict) -> pd.DataFrame:
    rows = []
    for modality, df in tables.items():
        for column in feature_cols(df, modality):
            non_missing = int(df[column].notna().sum())
            rows.append({
                "modality": modality,
                "feature": column,
                "non_missing": non_missing,
                "n_patients": len(df),
                "coverage_pct": round(100.0 * non_missing / len(df), 1),
            })
    return pd.DataFrame(rows)


def dictionary(mmse: pd.DataFrame, mri: pd.DataFrame, csf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in feature_cols(mmse, "mmse"):
        if column == "MMSCORE":
            group, text = "total", "MMSE total score, 0-30"
        elif column in DOMAINS:
            group, text = "domain", f"sum of {len(DOMAINS[column])} items, non-null only"
        else:
            group, text = "item", f"scored item, domain = {ITEM_TO_DOMAIN.get(column, '?')}"
        rows.append({"modality": "mmse", "feature": column, "group": group,
                     "description": text})

    for column in feature_cols(mri, "mri"):
        group = MRI_SUFFIX.get(column[-2:].upper(), "other")
        text = (
            "IntraCranialVol (ICV)" if column.startswith(ICV_RAW_COL)
            else f"Desikan-Killiany region, {group}"
        )
        rows.append({"modality": "mri", "feature": column, "group": group,
                     "description": text})

    descriptions = {
        "ELE_ABETA42": "amyloid-beta 1-42, pg/mL",
        "ELE_ABETA40": "amyloid-beta 1-40, pg/mL (sparse on Elecsys)",
        "ELE_TAU": "total tau, pg/mL",
        "ELE_PTAU": "phospho-tau 181, pg/mL",
        "ELE_PTAU_ABETA42": "p-tau / Abeta42 ratio",
        "ELE_TAU_ABETA42": "t-tau / Abeta42 ratio",
        "ELE_ABETA42_40": "Abeta42 / Abeta40 ratio",
    }
    for column in feature_cols(csf, "csf"):
        group = (
            "censoring flag" if "censored" in column
            else ("ratio" if "_" in column[4:] else "analyte")
        )
        rows.append({"modality": "csf", "feature": column, "group": group,
                     "description": descriptions.get(column, "assay out-of-range flag")})
    return pd.DataFrame(rows)


def audit_report(ref, tables, gaps, coverage, max_gap) -> str:
    lines = [
        "# Baseline feature build - audit report", "",
        f"Reference t=0: DXSUM baseline visit, {len(ref)} patients "
        f"({int(ref['dx_is_bl_coded'].sum())} baseline-coded).", "",
        f"Gap filter applied: "
        f"{'none' if max_gap is None else f'|gap| <= {max_gap} months'}",
        "", "## Record selection and timing", "", markdown_table(gaps), "",
        "## Notes", "",
    ]
    for modality, df in tables.items():
        fallback = int((df[f"{modality}_is_bl_coded"] == 0).sum())
        late = int((df[f"{modality}_gap_months"] > 12).sum())
        lines.append(
            f"- **{modality}**: {len(df)} patients, "
            f"{len(feature_cols(df, modality))} features. {fallback} used the "
            f"earliest-record fallback; {late} sit more than 12 months after t=0 and "
            f"are candidates for exclusion."
        )
    if "mmse" in tables:
        span = tables["mmse"]["mmse_bl_span_days"].dropna()
        lines.append(
            f"- **mmse coalesce window**: median {span.median():.0f} days, "
            f"max {span.max():.0f} days across the baseline-coded rows being merged."
        )
    if "mri" in tables and "FIELD_STRENGTH" in tables["mri"].columns:
        counts = tables["mri"]["FIELD_STRENGTH"].value_counts(dropna=False)
        lines.append(
            f"- **mri field strength**: {counts.to_dict()}. This tracks ADNI study era "
            f"(1.5T = ADNI-1, 3T = ADNI-GO/2/3) and is a known confound for any outcome "
            f"that differs by era - carried here so it can be controlled downstream."
        )
    lines += [
        "", "## Lowest-coverage features (bottom 15)", "",
        markdown_table(coverage.sort_values("coverage_pct").head(15)), "",
    ]
    return "\n".join(lines)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s01_baseline_data",
        description="Build the t=0 MMSE / MRI / CSF baseline feature tables.",
    )
    parser.add_argument(
        "--max-gap-months", type=float, default=None,
        help="drop records whose |gap| from t=0 exceeds this (default: keep everything)",
    )
    args = parser.parse_args(argv)

    ensure_output_dirs()

    print("=" * 88)
    print("BASELINE (t=0) FEATURE BUILD  -  MMSE / MRI / CSF")
    print("=" * 88)

    history = dx_history()
    ref = reference_baseline(history)
    print(f"\nreference t=0 from DXSUM: {len(ref)} patients, "
          f"{len(history)} diagnosis visits total\n")

    print("[1/3] MMSE")
    mmse = build_mmse(ref, history)
    print("\n[2/3] MRI")
    mri = build_mri(ref, history)
    print("\n[3/3] CSF")
    csf = build_csf(ref, history)

    tables = {"mmse": mmse, "mri": mri, "csf": csf}

    if args.max_gap_months is not None:
        print(f"\napplying gap filter |gap| <= {args.max_gap_months} months:")
        for modality in tables:
            df = tables[modality]
            before = len(df)
            tables[modality] = df[
                df[f"{modality}_gap_months"].abs() <= args.max_gap_months
            ].reset_index(drop=True)
            print(f"  {modality}: {before} -> {len(tables[modality])}  "
                  f"(dropped {before - len(tables[modality])})")
        mmse, mri, csf = tables["mmse"], tables["mri"], tables["csf"]

    gaps = gap_summary(tables)
    coverage = coverage_table(tables)
    dictionary_table = dictionary(mmse, mri, csf)

    dx_at_test = pd.DataFrame([{
        "modality": modality,
        "resolved": int(df["DX_at_test"].notna().sum()),
        "unresolved": int(df["DX_at_test"].isna().sum()),
        **{f"n_{k}": int((df["DX_at_test"] == k).sum()) for k in ("CN", "MCI", "AD")},
        **{f"src_{k}": int((df["DX_at_test_source"] == k).sum())
           for k in ("visit_match", "next_visit", "prior_visit", "none")},
        "differs_from_DX_baseline": int((
            df["DX_at_test"].notna()
            & df["DX_baseline"].notna()
            & (df["DX_at_test"] != df["DX_baseline"])
        ).sum()),
    } for modality, df in tables.items()])

    for modality, df in tables.items():
        write_csv(df, baseline_csv(modality))

    run_config = pd.DataFrame({
        "field": ["reference", "selection_rule", "mmse_note", "csf_platform", "mri_note",
                  "gap_filter", "n_dxsum_patients"],
        "value": [
            "DXSUM baseline visit = t=0",
            "earliest baseline-coded record; else earliest of any code",
            "baseline-window rows coalesced field-wise",
            "Roche Elecsys only",
            f"{len([c for c in mri.columns if c.startswith('ST')])} ST* regions, "
            f"QC failures dropped, no ICV normalisation applied",
            "none" if args.max_gap_months is None
            else f"|gap| <= {args.max_gap_months} months",
            len(ref),
        ],
    })

    sheets = {f"{modality}_baseline": df for modality, df in tables.items()}
    sheets.update({
        "coverage": coverage,
        "dictionary": dictionary_table,
        "record_selection": gaps,
        "dx_at_test": dx_at_test,
        "run_config": run_config,
    })
    xlsx = write_excel(sheets, BASELINE_XLSX)
    write_text(audit_report(ref, tables, gaps, coverage, args.max_gap_months), AUDIT_REPORT_MD)

    print("\n" + "=" * 88)
    print("RECORD SELECTION AND TIMING")
    print("=" * 88)
    print(gaps.to_string(index=False))

    print("\nDX_AT_TEST  (diagnosis carried at the visit the measurement was taken)")
    print(dx_at_test.to_string(index=False))

    print("\nfeature counts (rows = patients, columns = features):")
    for modality, df in tables.items():
        print(f"  {modality:<6}{len(df):>6} patients x "
              f"{len(feature_cols(df, modality)):>4} features")

    trimodal = set(mmse.RID) & set(mri.RID) & set(csf.RID)
    print(f"\npatients with all three modalities: {len(trimodal)}")

    print(f"\nwritten to {baseline_csv('mmse').parent}:")
    for modality in tables:
        print(f"  {baseline_csv(modality).name}")
    print(f"  {xlsx.name}")
    print(f"  {AUDIT_REPORT_MD.name}")


if __name__ == "__main__":
    main()
