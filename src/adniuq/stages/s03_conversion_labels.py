from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..io import write_csv
from ..paths import (
    CONVERSION_LABELS_CSV,
    DXSUM_PATTERN,
    RAW_DATA_DIR,
    ensure_output_dirs,
    find_one,
)

BASELINE_CODES = {"sc", "scmri", "bl", "init"}
DX_MAP = {1: "CN", 2: "MCI", 3: "AD"}
MONTHS = 365.25 / 12.0


def dx_history() -> pd.DataFrame:
    path = find_one(DXSUM_PATTERN, RAW_DATA_DIR)
    dx = pd.read_csv(path, low_memory=False)
    print(f"DXSUM: {path.name}  ({len(dx)} rows)")

    dx["RID"] = pd.to_numeric(dx["RID"], errors="coerce")
    dx["DIAGNOSIS"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce")
    dx["_date"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx = dx.dropna(subset=["RID", "DIAGNOSIS", "_date"])
    dx["RID"] = dx["RID"].astype(int)
    dx["DX"] = dx["DIAGNOSIS"].round().map(DX_MAP)
    dx["_is_bl"] = dx["VISCODE2"].astype(str).str.strip().str.lower().isin(BASELINE_CODES)
    return dx.sort_values(["RID", "_date"]).reset_index(drop=True)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s03_conversion_labels",
        description="Derive the MCI-to-AD conversion label from the DXSUM trajectory.",
    )
    parser.add_argument(
        "--min-followup-months", type=float, default=0.0,
        help="drop STABLE patients followed for less than this (default 0: keep everyone)",
    )
    parser.add_argument(
        "--exclude-reverters", action="store_true",
        help="drop patients whose AD diagnosis was later reversed",
    )
    args = parser.parse_args(argv)

    ensure_output_dirs()
    dx = dx_history()

    baseline = (
        dx.sort_values(["_is_bl", "_date"], ascending=[False, True])
        .drop_duplicates("RID")
        .set_index("RID")[["_date", "DX"]]
        .rename(columns={"_date": "baseline_date", "DX": "DX_baseline"})
    )

    mci = baseline.index[baseline["DX_baseline"] == "MCI"]
    print(f"\npatients in DXSUM         : {baseline.shape[0]}")
    print(f"MCI at t=0 (eligible)     : {len(mci)}")

    rows = []
    for rid, group in dx[dx["RID"].isin(set(mci))].groupby("RID", sort=True):
        t0 = baseline.at[rid, "baseline_date"]
        later = group[group["_date"] > t0]
        ad = later[later["DX"] == "AD"]

        if len(ad):
            first_ad = ad["_date"].min()
            time_to_conversion = (first_ad - t0).days / MONTHS
            reverted = bool((later[later["_date"] > first_ad]["DX"] != "AD").any())
        else:
            time_to_conversion, reverted = np.nan, False

        rows.append({
            "RID": rid,
            "baseline_date": t0.date(),
            "conv_label": "converter" if len(ad) else "stable",
            "y": 1 if len(ad) else 0,
            "time_to_conv_months": (
                round(time_to_conversion, 1) if not np.isnan(time_to_conversion) else np.nan
            ),
            "followup_months": round((group["_date"].max() - t0).days / MONTHS, 1),
            "n_followup_visits": int(len(later)),
            "n_dx_visits": int(len(group)),
            "reverted": int(reverted),
        })

    labels = pd.DataFrame(rows)

    if args.exclude_reverters:
        before = len(labels)
        labels = labels[labels["reverted"] == 0].reset_index(drop=True)
        print(f"\nexcluded {before - len(labels)} reverters")

    if args.min_followup_months > 0:
        before = len(labels)
        drop = (labels["conv_label"] == "stable") & (
            labels["followup_months"] < args.min_followup_months
        )
        labels = labels[~drop].reset_index(drop=True)
        print(f"\ndropped {before - len(labels)} stable patients followed < "
              f"{args.min_followup_months} months")

    path = write_csv(labels, CONVERSION_LABELS_CSV)

    counts = labels["conv_label"].value_counts()
    print("\n" + "=" * 78)
    print("MCI -> AD CONVERSION LABEL  (no horizon: ever converted vs never)")
    print("=" * 78)
    for key in ("converter", "stable"):
        print(f"  {key:<12}{int(counts.get(key, 0)):>6}")
    print(f"  {'total':<12}{len(labels):>6}")
    print(f"\n  converter rate    : {100.0 * labels['y'].mean():.1f}%")
    print(f"  majority baseline : {max(labels['y'].mean(), 1 - labels['y'].mean()):.4f}")

    conversion_times = labels.loc[labels["y"] == 1, "time_to_conv_months"]
    print(f"\n  time to conversion (months): median={conversion_times.median():.1f}  "
          f"min={conversion_times.min():.1f}  max={conversion_times.max():.1f}")
    for month in (12, 24, 36, 48, 60):
        print(f"    by {month:>3}mo : {int((conversion_times <= month).sum()):>4}  "
              f"({100.0 * (conversion_times <= month).mean():.1f}% of converters)")

    stable_followup = labels.loc[labels["y"] == 0, "followup_months"]
    print("\n  follow-up of the STABLE group - how much each negative is worth:")
    print(f"    median={stable_followup.median():.1f}  min={stable_followup.min():.1f}  "
          f"max={stable_followup.max():.1f}")
    for month in (12, 24, 36, 48):
        print(f"    followed < {month:>2}mo : {int((stable_followup < month).sum()):>4}  "
              f"({100.0 * (stable_followup < month).mean():.1f}% of stable)")
    no_followup = int((stable_followup <= 0).sum())
    if no_followup:
        print(f"    ** {no_followup} stable patients have NO post-baseline visit at all - "
              f"their label rests on no evidence")

    print(f"\n  reverters (AD then a later non-AD dx): {int(labels['reverted'].sum())}")
    print(f"\n  saved -> {path.name}")


if __name__ == "__main__":
    main()
