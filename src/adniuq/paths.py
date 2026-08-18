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
CROSS_VALIDATION_DIR = OUTPUT_ROOT / "09_cross_validation"
BOUNDARY_DIR = OUTPUT_ROOT / "12_boundary_distance"
HANDOFF_DIR = OUTPUT_ROOT / "13_modality_handoff"
# the mirror of the handoff: what the target model does to the patients the source
# model was confident about and therefore kept
CONFIDENT_AUDIT_DIR = HANDOFF_DIR / "confident_audit"
FUSION_HANDOFF_DIR = OUTPUT_ROOT / "14_fusion_handoff"

STAGE_DIRS = (
    BASELINE_DIR,
    TRIMODAL_DIR,
    CONVERSION_DIR,
    INDIVIDUAL_DIR,
    CROSS_VALIDATION_DIR,
    BOUNDARY_DIR,
    HANDOFF_DIR,
    CONFIDENT_AUDIT_DIR,
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


def feature_selection_xlsx(model: str) -> Path:
    return INDIVIDUAL_DIR / f"{model}_feature_selection.xlsx"


def confidence_bands_xlsx(model: str) -> Path:
    return INDIVIDUAL_DIR / f"{model}_confidence_bands.xlsx"


# every artefact is named for the pair it audits, so auditing a second pair adds to
# the directory instead of overwriting the first
def confident_audit_path(tag: str, suffix: str) -> Path:
    return CONFIDENT_AUDIT_DIR / f"confident_{tag}_{suffix}"


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
