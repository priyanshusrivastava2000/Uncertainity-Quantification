from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

RAW_DATA_DIR = Path(
    os.environ.get("ADNIUQ_DATA_DIR", PROJECT_ROOT / "data" / "raw")
).resolve()
COGNITIVE_DIR = RAW_DATA_DIR / "Cognitive"
MRI_DIR = RAW_DATA_DIR / "MRI"
CSF_DIR = RAW_DATA_DIR / "CSF"

OUTPUT_ROOT = Path(
    os.environ.get("ADNIUQ_OUTPUT_DIR", PROJECT_ROOT / "outputs")
).resolve()

BASELINE_DIR = OUTPUT_ROOT / "01_baseline_data"
TRIMODAL_DIR = OUTPUT_ROOT / "02_trimodal_baseline"
CONVERSION_DIR = OUTPUT_ROOT / "03_conversion_dataset"
INDIVIDUAL_DIR = OUTPUT_ROOT / "04_individual_models"
PIPELINE_DIR = OUTPUT_ROOT / "05_pipeline"
COMBINED_DIR = OUTPUT_ROOT / "06_combined_mmse_mri"
COMBINED_ALL_DIR = OUTPUT_ROOT / "07_combined_all"
PIPELINE_ALL_DIR = OUTPUT_ROOT / "08_pipeline_all"
CROSS_VALIDATION_DIR = OUTPUT_ROOT / "09_cross_validation"
SPECIALIST_DIR = OUTPUT_ROOT / "10_ambiguous_specialist"
BOUNDARY_DIR = OUTPUT_ROOT / "12_boundary_distance"
HANDOFF_DIR = OUTPUT_ROOT / "13_modality_handoff"
FUSION_HANDOFF_DIR = OUTPUT_ROOT / "14_fusion_handoff"

STAGE_DIRS = (
    BASELINE_DIR,
    TRIMODAL_DIR,
    CONVERSION_DIR,
    INDIVIDUAL_DIR,
    PIPELINE_DIR,
    COMBINED_DIR,
    COMBINED_ALL_DIR,
    PIPELINE_ALL_DIR,
    CROSS_VALIDATION_DIR,
    SPECIALIST_DIR,
    BOUNDARY_DIR,
    HANDOFF_DIR,
    FUSION_HANDOFF_DIR,
)

DXSUM_PATTERN = "*DXSUM*.csv"
MMSE_PATTERN = "All_Subjects_MMSE_*.csv"
UCSFFSX7_PATTERN = "All_Subjects_UCSFFSX7_*.csv"
ELECSYS_PATTERN = "UPENNBIOMK_ROCHE_ELECSYS_*.csv"

BASELINE_XLSX = BASELINE_DIR / "baseline_features.xlsx"
AUDIT_REPORT_MD = BASELINE_DIR / "audit_report.md"

TRIMODAL_XLSX = TRIMODAL_DIR / "trimodal_baseline.xlsx"
TRIMODAL_RIDS_CSV = TRIMODAL_DIR / "trimodal_rids.csv"

CONVERSION_LABELS_CSV = CONVERSION_DIR / "conversion_labels.csv"
SPLIT_SOURCE_CSV = CONVERSION_DIR / "mmse_conversion.csv"

INDIVIDUAL_RESULTS_CSV = INDIVIDUAL_DIR / "individual_results.csv"

STAGE2_MMSE_MRI_CSV = PIPELINE_DIR / "stage2_mmse_mri.csv"
STAGE2_BASELINE_XLSX = PIPELINE_DIR / "stage2_baseline.xlsx"
STAGE2_FEATURE_SELECTION_XLSX = PIPELINE_DIR / "stage2_feature_selection.xlsx"
STAGE2_PIPELINE_MATRIX_PNG = PIPELINE_DIR / "confidence_matrix_pipeline.png"
PIPELINE_RESULTS_CSV = PIPELINE_DIR / "pipeline_results.csv"

COMBINED_CSV = COMBINED_DIR / "combined_mmse_mri.csv"
COMBINED_BASELINE_XLSX = COMBINED_DIR / "combined_baseline.xlsx"
COMBINED_FEATURE_SELECTION_XLSX = COMBINED_DIR / "combined_feature_selection.xlsx"
COMBINED_RESULTS_CSV = COMBINED_DIR / "combined_results.csv"
COMBINED_BANDS_XLSX = COMBINED_DIR / "combined_confidence_bands.xlsx"

COMBINED_ALL_CSV = COMBINED_ALL_DIR / "combined_all_mmse_mri_csf.csv"
COMBINED_ALL_BASELINE_XLSX = COMBINED_ALL_DIR / "combined_all_baseline.xlsx"
COMBINED_ALL_FEATURE_SELECTION_XLSX = (
    COMBINED_ALL_DIR / "combined_all_feature_selection.xlsx"
)
COMBINED_ALL_RESULTS_CSV = COMBINED_ALL_DIR / "combined_all_results.csv"
COMBINED_ALL_BANDS_XLSX = COMBINED_ALL_DIR / "combined_all_confidence_bands.xlsx"

STAGE2_ALL_CSV = PIPELINE_ALL_DIR / "stage2_all.csv"
STAGE2_ALL_BASELINE_XLSX = PIPELINE_ALL_DIR / "stage2_all_baseline.xlsx"
STAGE2_ALL_FEATURE_SELECTION_XLSX = (
    PIPELINE_ALL_DIR / "stage2_all_feature_selection.xlsx"
)
STAGE2_ALL_PIPELINE_MATRIX_PNG = PIPELINE_ALL_DIR / "confidence_matrix_pipeline_all.png"
PIPELINE_ALL_RESULTS_CSV = PIPELINE_ALL_DIR / "pipeline_all_results.csv"

BOUNDARY_DISTANCES_CSV = BOUNDARY_DIR / "boundary_distances.csv"
BOUNDARY_DISTANCES_XLSX = BOUNDARY_DIR / "boundary_distances.xlsx"
BOUNDARY_COMPARISON_PNG = BOUNDARY_DIR / "boundary_distance_comparison.png"

HANDOFF_SPLIT_CSV = HANDOFF_DIR / "handoff_split.csv"
HANDOFF_ALL_PATIENTS_CSV = HANDOFF_DIR / "handoff_all_patients.csv"
HANDOFF_PREDICTIONS_CSV = HANDOFF_DIR / "handoff_test_predictions.csv"
HANDOFF_XLSX = HANDOFF_DIR / "modality_handoff.xlsx"
HANDOFF_RESULTS_CSV = HANDOFF_DIR / "handoff_model_results.csv"

FUSION_SPLIT_CSV = FUSION_HANDOFF_DIR / "fusion_split.csv"
FUSION_ALL_PATIENTS_CSV = FUSION_HANDOFF_DIR / "fusion_all_patients.csv"
FUSION_PREDICTIONS_CSV = FUSION_HANDOFF_DIR / "fusion_test_predictions.csv"
FUSION_XLSX = FUSION_HANDOFF_DIR / "fusion_handoff.xlsx"
FUSION_RESULTS_CSV = FUSION_HANDOFF_DIR / "fusion_model_results.csv"

CV_REPEATS_CSV = CROSS_VALIDATION_DIR / "cv_repeats.csv"
CV_SUMMARY_CSV = CROSS_VALIDATION_DIR / "cv_summary.csv"
CV_XLSX = CROSS_VALIDATION_DIR / "cross_validation.xlsx"
CV_AUC_PNG = CROSS_VALIDATION_DIR / "cv_auc_distribution.png"


def baseline_csv(modality: str) -> Path:
    return BASELINE_DIR / f"{modality}_baseline.csv"


def trimodal_csv(modality: str) -> Path:
    return TRIMODAL_DIR / f"{modality}_trimodal.csv"


def conversion_csv(modality: str) -> Path:
    return CONVERSION_DIR / f"{modality}_conversion.csv"


def conversion_xlsx(modality: str) -> Path:
    return CONVERSION_DIR / f"{modality}_conversion.xlsx"


def feature_selection_xlsx(modality: str) -> Path:
    return INDIVIDUAL_DIR / f"{modality}_feature_selection.xlsx"


def specialist_dir(cohort_tag: str) -> Path:
    return SPECIALIST_DIR / cohort_tag


def specialist_file(cohort_tag: str, name: str) -> Path:
    return specialist_dir(cohort_tag) / name


def ensure_output_dirs() -> None:
    for directory in STAGE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def find_one(pattern: str, directory: Path) -> Path:
    hits = sorted(glob.glob(str(directory / pattern)))
    if not hits:
        sys.exit(
            f"[error] no file matching {pattern} in {directory}\n"
            f"        set ADNIUQ_DATA_DIR to the folder holding the ADNI tables"
        )
    return Path(hits[0])


def require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        sys.exit(
            "[error] required input(s) not found - run the upstream stage first:\n  "
            + "\n  ".join(missing)
        )
