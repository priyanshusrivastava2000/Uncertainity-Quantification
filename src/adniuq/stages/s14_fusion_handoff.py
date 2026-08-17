from __future__ import annotations

from ..paths import (
    FUSION_ALL_PATIENTS_CSV,
    FUSION_HANDOFF_DIR,
    FUSION_PREDICTIONS_CSV,
    FUSION_RESULTS_CSV,
    FUSION_SPLIT_CSV,
    FUSION_XLSX,
)
from .s13_modality_handoff import build_parser, resolve_gates, run_handoff

# same machinery as stage 13, same 50-50 split, but the chain grows a modality at each
# step instead of swapping one for another: MMSE -> MMSE+MRI -> MMSE+MRI+CSF
CHAIN = ["mmse", "mmse_mri", "mmse_mri_csf"]

FUSION_PATHS = {
    "dir": FUSION_HANDOFF_DIR,
    "split_csv": FUSION_SPLIT_CSV,
    "all_patients_csv": FUSION_ALL_PATIENTS_CSV,
    "predictions_csv": FUSION_PREDICTIONS_CSV,
    "results_csv": FUSION_RESULTS_CSV,
    "xlsx": FUSION_XLSX,
}


def main(argv: list | None = None) -> None:
    parser = build_parser(
        "s14_fusion_handoff",
        "Train MMSE, MMSE+MRI and MMSE+MRI+CSF on the same 50-50 split and hand the "
        "patients each model is unsure about to the next, wider one.",
        CHAIN,
    )
    args = parser.parse_args(argv)
    source_gates = resolve_gates(parser, args)
    run_handoff(args, source_gates, FUSION_PATHS,
                "FUSION HANDOFF  -  MMSE -> MMSE+MRI -> MMSE+MRI+CSF on a 50-50 split")


if __name__ == "__main__":
    main()
