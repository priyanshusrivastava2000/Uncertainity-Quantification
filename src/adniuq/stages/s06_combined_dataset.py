from __future__ import annotations

import argparse

import pandas as pd

from ..features import LABEL_COLS, all_features, covered
from ..io import write_csv, write_excel
from ..paths import (
    COMBINED_BASELINE_XLSX,
    COMBINED_CSV,
    conversion_csv,
    ensure_output_dirs,
    require,
)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s06_combined_dataset",
        description="Build the plain MMSE + MRI fusion dataset (no routing).",
    )
    parser.add_argument("--min-coverage", type=float, default=0.5)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    require(conversion_csv("mmse"), conversion_csv("mri"))

    print("=" * 88)
    print("COMBINED MMSE + MRI BASELINE DATASET")
    print("=" * 88)

    mmse = pd.read_csv(conversion_csv("mmse"), low_memory=False)
    mri = pd.read_csv(conversion_csv("mri"), low_memory=False)
    assert mmse["RID"].tolist() == mri["RID"].tolist(), "source files are not aligned"
    assert mmse["y"].tolist() == mri["y"].tolist(), "labels disagree between source files"
    print(f"  patients: {len(mmse)}   (MMSE and MRI files aligned on RID and label)")

    mmse_cols = all_features("mmse", mmse)
    mri_cols = all_features("mri", mri)
    mmse_feats, _, mmse_dropped = covered(mmse, mmse_cols, args.min_coverage)
    mri_feats, _, mri_dropped = covered(mri, mri_cols, args.min_coverage)

    assert not (set(mmse_feats) & set(mri_feats)), "feature name collision between modalities"
    feats = mmse_feats + mri_feats
    print(f"  MMSE features : {len(mmse_feats)} of {len(mmse_cols)}"
          + (f"   dropped {mmse_dropped}" if mmse_dropped else ""))
    print(f"  MRI  features : {len(mri_feats)} of {len(mri_cols)}"
          + (f"   dropped {mri_dropped}" if mri_dropped else ""))
    print(f"  combined      : {len(feats)}")

    out = pd.concat([mmse[["RID"] + LABEL_COLS], mmse[mmse_feats], mri[mri_feats]], axis=1)
    assert out["RID"].is_unique
    assert out.shape[1] == 1 + len(LABEL_COLS) + len(feats)

    csv_path = write_csv(out, COMBINED_CSV)

    composition = (
        out.groupby(["conv_label", "split"]).size().unstack(fill_value=0)
        .assign(total=lambda d: d.sum(axis=1))
    )
    composition.loc["TOTAL"] = composition.sum()

    coverage = pd.DataFrame([{
        "feature": column,
        "modality": "MMSE" if column in mmse_feats else "MRI",
        "non_missing": int(out[column].notna().sum()),
        "coverage_pct": round(100 * out[column].notna().mean(), 1),
    } for column in feats]).sort_values("coverage_pct")

    config = pd.DataFrame({
        "field": ["task", "horizon", "eligibility", "cohort", "patients", "n_features",
                  "mmse_features", "mri_features", "coverage_floor", "split",
                  "split_source", "note"],
        "value": [
            "MCI -> AD conversion", "none - ever converted vs never",
            "DX_baseline == MCI at t=0",
            "tri-modal (MMSE + MRI + CSF present at t=0)", len(out), len(feats),
            len(mmse_feats), len(mri_feats), f">= {args.min_coverage:.0%}",
            f"{int((out.split == 'train').sum())} train / "
            f"{int((out.split == 'test').sum())} test",
            "inherited from the MMSE conversion dataset - identical held-out patients "
            "as the single-modality models",
            "MMSE columns are exactly nested (MMSCORE = sum of items, domains = partial "
            "sums); predictions are unaffected but MMSE weights are not effect sizes.",
        ],
    })

    xlsx = write_excel({
        "combined_data": out,
        "split_composition": composition.reset_index(),
        "coverage": coverage,
        "run_config": config,
    }, COMBINED_BASELINE_XLSX)

    print("\n" + "=" * 88)
    print(f"DATASET: {len(out)} patients x {len(feats)} features ({out.shape[1]} columns)")
    print("=" * 88)
    print(composition.to_string())
    print(f"\n  converter rate {out.y.mean():.4f}   "
          f"majority baseline {max(out.y.mean(), 1 - out.y.mean()):.4f}")
    print(f"\nwritten:\n  {csv_path.name}\n  {xlsx.name}")


if __name__ == "__main__":
    main()
