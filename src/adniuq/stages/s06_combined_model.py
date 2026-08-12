from __future__ import annotations

import argparse

from ..experiments import run_fusion_model
from ..paths import (
    COMBINED_CSV,
    COMBINED_DIR,
    COMBINED_FEATURE_SELECTION_XLSX,
    COMBINED_RESULTS_CSV,
    INDIVIDUAL_RESULTS_CSV,
    ensure_output_dirs,
    require,
)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s06_combined_model",
        description="Fit CLG-Lasso on the combined MMSE + MRI baseline.",
    )
    parser.add_argument("--hi", type=float, default=0.80)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    require(COMBINED_CSV)

    run_fusion_model(
        source_csv=COMBINED_CSV,
        out_dir=COMBINED_DIR,
        task_label="combined",
        display_name="Combined MMSE + MRI -> conversion",
        model_name="MMSE+MRI combined",
        modality_label="MMSE+MRI",
        feature_selection_path=COMBINED_FEATURE_SELECTION_XLSX,
        results_csv=COMBINED_RESULTS_CSV,
        comparison_csvs=[INDIVIDUAL_RESULTS_CSV],
        args=args,
    )

    print(f"\n  written:\n    confidence_matrix_combined.png\n    combined_results.xlsx"
          f"\n    {COMBINED_FEATURE_SELECTION_XLSX.name}\n    {COMBINED_RESULTS_CSV.name}")


if __name__ == "__main__":
    main()
