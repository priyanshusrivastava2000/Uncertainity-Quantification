from __future__ import annotations

import importlib
import sys

from .stages import STAGES

PIPELINE = (
    ("s01_baseline_data", []),
    ("s02_trimodal_baseline", []),
    ("s03_conversion_labels", []),
    ("s03_mmse_conversion", []),
    ("s03_modality_conversion", ["--modality", "mri"]),
    ("s03_modality_conversion", ["--modality", "csf"]),
    ("s04_individual_models", []),
    ("s04_feature_selection", []),
    ("s09_cross_validation", []),
    ("s09_readme_table", []),
    ("s11_confidence_bands", []),
    ("s12_boundary_distance", []),
    ("s13_modality_handoff", []),
    ("s14_fusion_handoff", []),
    ("s15_confident_audit", []),
)

USAGE = (
    "usage: adniuq <stage> [stage options]\n"
    "       adniuq all            run every stage in order\n"
    "       adniuq list           list the available stages\n\n"
    "stages:\n  " + "\n  ".join(STAGES)
)


def run_stage(name: str, argv: list) -> None:
    module = importlib.import_module(f".stages.{name}", package="adniuq")
    module.main(argv)


def main(argv: list | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    command, rest = argv[0], argv[1:]

    if command == "list":
        print("\n".join(STAGES))
        return 0

    if command == "all":
        if rest:
            print(f"[error] 'all' takes no options: {rest}", file=sys.stderr)
            return 2
        for name, stage_args in PIPELINE:
            print(f"\n{'#' * 92}\n### {name} {' '.join(stage_args)}\n{'#' * 92}")
            run_stage(name, stage_args)
        return 0

    if command not in STAGES:
        print(f"[error] unknown stage: {command}\n\n{USAGE}", file=sys.stderr)
        return 2

    run_stage(command, rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
