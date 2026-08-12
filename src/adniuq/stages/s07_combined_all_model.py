from __future__ import annotations

import argparse

from ..experiments import run_fusion_model
from ..paths import (
    COMBINED_ALL_CSV,
    COMBINED_ALL_DIR,
    COMBINED_ALL_FEATURE_SELECTION_XLSX,
    COMBINED_ALL_RESULTS_CSV,
    COMBINED_RESULTS_CSV,
    INDIVIDUAL_RESULTS_CSV,
    ensure_output_dirs,
    require,
)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s07_combined_all_model",
        description="Fit CLG-Lasso on the full tri-modal MMSE + MRI + CSF baseline.",
    )
    parser.add_argument("--hi", type=float, default=0.80)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    require(COMBINED_ALL_CSV)

    run_fusion_model(
        source_csv=COMBINED_ALL_CSV,
        out_dir=COMBINED_ALL_DIR,
        task_label="combined_all",
        display_name="Tri-modal MMSE + MRI + CSF -> conversion",
        model_name="MMSE+MRI+CSF combined",
        modality_label="ALL",
        feature_selection_path=COMBINED_ALL_FEATURE_SELECTION_XLSX,
        results_csv=COMBINED_ALL_RESULTS_CSV,
        comparison_csvs=[INDIVIDUAL_RESULTS_CSV, COMBINED_RESULTS_CSV],
        args=args,
    )

    print(f"\n  written:\n    confidence_matrix_combined_all.png"
          f"\n    combined_all_results.xlsx"
          f"\n    {COMBINED_ALL_FEATURE_SELECTION_XLSX.name}"
          f"\n    {COMBINED_ALL_RESULTS_CSV.name}")


if __name__ == "__main__":
    main()
