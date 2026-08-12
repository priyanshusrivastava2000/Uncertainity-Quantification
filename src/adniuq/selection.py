from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import describe, modality_of
from .io import write_excel

NOTE = (
    "zero weight = eliminated by the lasso; dropping the column changes no prediction. "
    "MMSE columns are exactly nested, so an MMSE zero often means redundant rather "
    "than uninformative."
)


def _grouped(tbl: pd.DataFrame, keys: list) -> pd.DataFrame:
    return (
        tbl.groupby(keys)
        .agg(n_features=("feature", "size"), n_selected=("selected", "sum"))
        .assign(
            n_zero=lambda d: d.n_features - d.n_selected,
            pct_selected=lambda d: (100 * d.n_selected / d.n_features).round(1),
        )
        .reset_index()
    )


def selection_table(clf, feats: list, coverage: pd.Series) -> pd.DataFrame:
    beta = clf.beta_
    modalities = [modality_of(c) for c in feats]
    groups = [describe(m, c) for c, m in zip(feats, modalities)]
    return pd.DataFrame({
        "feature": feats,
        "modality": modalities,
        "group": [g for g, _ in groups],
        "description": [d for _, d in groups],
        "weight": np.round(beta, 6),
        "abs_weight": np.round(np.abs(beta), 6),
        "direction": [
            "-> converter" if b > 0 else ("-> stable" if b < 0 else "(zero)") for b in beta
        ],
        "selected": (beta != 0).astype(int),
        "coverage_pct": [round(100 * coverage[c], 1) for c in feats],
    })


def write_feature_selection(
    clf,
    feats: list,
    coverage: pd.Series,
    lam: float,
    auc: float,
    path: Path,
    model_name: str,
    min_coverage: float = 0.5,
):
    tbl = selection_table(clf, feats, coverage)

    non_zero = (
        tbl[tbl.selected == 1].sort_values("abs_weight", ascending=False).reset_index(drop=True)
    )
    non_zero.insert(0, "rank", np.arange(1, len(non_zero) + 1))
    zero = (
        tbl[tbl.selected == 0]
        .drop(columns=["weight", "abs_weight", "direction", "selected"])
        .sort_values(["modality", "group", "feature"])
        .reset_index(drop=True)
    )
    by_modality = _grouped(tbl, ["modality"])
    by_group = _grouped(tbl, ["modality", "group"])

    n_nonzero = int(clf.n_nonzero)
    summary = pd.DataFrame({
        "field": [
            "model", "lambda", "features", "non_zero", "zero", "sparsity_pct",
            "coverage_floor", "mmse_selected", "mri_selected", "csf_selected",
            "test_auc", "note",
        ],
        "value": [
            model_name, lam, len(feats), n_nonzero, len(feats) - n_nonzero,
            round(100 * (1 - n_nonzero / len(feats)), 1), f">= {min_coverage:.0%}",
            int((non_zero.modality == "MMSE").sum()),
            int((non_zero.modality == "MRI").sum()),
            int((non_zero.modality == "CSF").sum()),
            round(float(auc), 4), NOTE,
        ],
    })

    written = write_excel({
        "summary": summary,
        "non_zero": non_zero,
        "zero": zero,
        "by_modality": by_modality,
        "by_group": by_group,
        "all_features": tbl.sort_values("abs_weight", ascending=False),
    }, path)
    return written, by_modality, by_group, non_zero
