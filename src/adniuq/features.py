from __future__ import annotations

import pandas as pd

DOMAINS = {
    "orientation_time": ["MMDATE", "MMYEAR", "MMMONTH", "MMDAY", "MMSEASON"],
    "orientation_place": ["MMHOSPIT", "MMFLOOR", "MMCITY", "MMAREA", "MMSTATE"],
    "registration": ["WORD1", "WORD2", "WORD3"],
    "attention": ["MMD", "MML", "MMR", "MMO", "MMW"],
    "recall": ["WORD1DL", "WORD2DL", "WORD3DL"],
    "language": [
        "MMWATCH", "MMPENCIL", "MMREPEAT", "MMHAND", "MMFOLD",
        "MMONFLR", "MMREAD", "MMWRITE", "MMDRAW",
    ],
}
ITEMS = [item for items in DOMAINS.values() for item in items]
ITEM_TO_DOMAIN = {item: domain for domain, items in DOMAINS.items() for item in items}
MMSE_FEATURES = ["MMSCORE"] + list(DOMAINS) + ITEMS

ICV_RAW_COL = "ST10CV"
ICV_COL = "ST10CV_ICV"

MRI_SUFFIX = {
    "CV": "cortical volume",
    "SV": "subcortical volume",
    "SA": "surface area",
    "TA": "thickness average",
    "TS": "thickness std dev",
}

LABEL_COLS = [
    "conv_label", "y", "time_to_conv_months", "followup_months",
    "n_followup_visits", "reverted", "split",
]
DX_COLS = ["DX_at_test", "DX_at_test_date", "DX_at_test_source", "DX_baseline"]

METADATA_PREFIX = (
    "RID", "DX_", "conv_label", "y", "time_to_conv", "followup_months",
    "n_followup_visits", "reverted", "split",
)
METADATA_EXACT = {
    "FIELD_STRENGTH", "IMAGEUID", "OVERALLQC", "FSVER", "BATCH",
    "mmse_n_bl_rows", "mmse_bl_span_days",
}

MODEL_META = {"RID", *LABEL_COLS}
STAGE1_MMSE_META = {
    "mmse_p_converter", "mmse_confidence", "mmse_pred", "mmse_correct",
    "mmse_conf_source", "routed_stage1",
}
STAGE1_COMBINED_META = {
    "stage1_p_converter", "stage1_confidence", "stage1_pred", "stage1_correct",
    "stage1_conf_source", "routed_stage1",
}

DISPLAY = {"mmse": "MMSE", "mri": "MRI", "csf": "CSF"}


def all_features(modality: str, df: pd.DataFrame) -> list:
    return [
        c for c in df.columns
        if c not in METADATA_EXACT
        and not c.startswith(METADATA_PREFIX)
        and not c.startswith(f"{modality}_")
    ]


def covered(df: pd.DataFrame, columns: list, min_coverage: float) -> tuple:
    coverage = df[columns].notna().mean()
    kept = [c for c in columns if coverage[c] >= min_coverage]
    dropped = [(c, round(100 * coverage[c], 1)) for c in columns if c not in kept]
    return kept, coverage, dropped


def modality_of(column: str) -> str:
    if column.startswith("ST"):
        return "MRI"
    return "CSF" if column.startswith("ELE_") else "MMSE"


def describe(modality: str, column: str) -> tuple:
    modality = modality.lower()
    if modality == "mmse":
        if column == "MMSCORE":
            return "total", "MMSE total, 0-30 (= sum of the 30 items)"
        if column in DOMAINS:
            return "domain", f"sum of {len(DOMAINS[column])} items"
        return "item", f"scored item, domain = {ITEM_TO_DOMAIN.get(column, '?')}"
    if modality == "mri":
        if column.endswith("_ICV"):
            return "ICV", "IntraCranialVol (head size)"
        suffix = column[-2:].upper()
        group = suffix if suffix in MRI_SUFFIX else "other"
        return group, f"Desikan-Killiany region, {MRI_SUFFIX.get(suffix, 'other')}"
    if "censored" in column:
        return "assay flag", "out-of-range marker - assay behaviour, not patient biology"
    if column.count("_") > 1:
        return "ratio", "derived analyte ratio"
    return "analyte", "Elecsys concentration, pg/mL"
