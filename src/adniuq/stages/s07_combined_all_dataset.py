from __future__ import annotations

import argparse

import pandas as pd

from ..features import LABEL_COLS, all_features, covered, modality_of
from ..io import write_csv, write_excel
from ..paths import (
    COMBINED_ALL_BASELINE_XLSX,
    COMBINED_ALL_CSV,
    conversion_csv,
    ensure_output_dirs,
    require,
)

MODALITIES = ("mmse", "mri", "csf")


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s07_combined_all_dataset",
        description="Build the full tri-modal MMSE + MRI + CSF fusion dataset.",
    )
    parser.add_argument("--min-coverage", type=float, default=0.5)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    require(*[conversion_csv(m) for m in MODALITIES])

    print("=" * 88)
    print("TRI-MODAL BASELINE DATASET  -  MMSE + MRI + CSF")
    print("=" * 88)

    sources = {
        m: pd.read_csv(conversion_csv(m), low_memory=False) for m in MODALITIES
    }
    reference = sources["mmse"]
    for modality in ("mri", "csf"):
        assert sources[modality]["RID"].tolist() == reference["RID"].tolist(), \
            f"{modality} is not aligned"
        assert sources[modality]["y"].tolist() == reference["y"].tolist(), \
            f"{modality} labels disagree"
    print(f"  patients: {len(reference)}   (all three files aligned on RID and label)")

    blocks = {}
    for modality in MODALITIES:
        columns = all_features(modality, sources[modality])
        kept, _, dropped = covered(sources[modality], columns, args.min_coverage)
        blocks[modality] = kept
        print(f"  {modality.upper():<5} features: {len(kept)} of {len(columns)}"
              + (f"   dropped {dropped}" if dropped else ""))

    feats = blocks["mmse"] + blocks["mri"] + blocks["csf"]
    assert len(set(feats)) == len(feats), "feature name collision across modalities"
    print(f"  {'TOTAL':<5} features: {len(feats)}")

    out = pd.concat(
        [reference[["RID"] + LABEL_COLS]] + [sources[m][blocks[m]] for m in MODALITIES],
        axis=1,
    )
    assert out["RID"].is_unique
    assert out.shape[1] == 1 + len(LABEL_COLS) + len(feats)

    csv_path = write_csv(out, COMBINED_ALL_CSV)

    composition = (
        out.groupby(["conv_label", "split"]).size().unstack(fill_value=0)
        .assign(total=lambda d: d.sum(axis=1))
    )
    composition.loc["TOTAL"] = composition.sum()

    coverage = pd.DataFrame([{
        "feature": column,
        "modality": modality_of(column),
        "non_missing": int(out[column].notna().sum()),
        "coverage_pct": round(100 * out[column].notna().mean(), 1),
    } for column in feats]).sort_values("coverage_pct")

    config = pd.DataFrame({
        "field": ["task", "horizon", "eligibility", "cohort", "patients", "n_features",
                  "mmse_features", "mri_features", "csf_features", "coverage_floor",
                  "split", "split_source", "caveat_csf", "caveat_mmse"],
        "value": [
            "MCI -> AD conversion", "none - ever converted vs never",
            "DX_baseline == MCI at t=0",
            "tri-modal (MMSE + MRI + CSF present at t=0)", len(out), len(feats),
            len(blocks["mmse"]), len(blocks["mri"]), len(blocks["csf"]),
            f">= {args.min_coverage:.0%}",
            f"{int((out.split == 'train').sum())} train / "
            f"{int((out.split == 'test').sum())} test",
            "inherited from the MMSE conversion dataset - identical held-out patients "
            "as every other model in the project",
            "4 of the 9 CSF columns are assay censoring flags, not patient biology; "
            "kept for consistency with the CSF-only model",
            "MMSE columns are exactly nested; weights are not effect sizes",
        ],
    })

    xlsx = write_excel({
        "combined_all_data": out,
        "split_composition": composition.reset_index(),
        "coverage": coverage,
        "run_config": config,
    }, COMBINED_ALL_BASELINE_XLSX)

    print("\n" + "=" * 88)
    print(f"DATASET: {len(out)} patients x {len(feats)} features ({out.shape[1]} columns)")
    print("=" * 88)
    print(composition.to_string())
    print(f"\n  converter rate {out.y.mean():.4f}   "
          f"majority baseline {max(out.y.mean(), 1 - out.y.mean()):.4f}")
    print(f"\nwritten:\n  {csv_path.name}\n  {xlsx.name}")


if __name__ == "__main__":
    main()
