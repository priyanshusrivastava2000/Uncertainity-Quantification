from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..paths import CV_SUMMARY_CSV, PROJECT_ROOT, require

START = "<!-- cv-summary:start -->"
END = "<!-- cv-summary:end -->"

COLUMNS = [
    ("ROC-AUC", "roc_auc"),
    ("PR-AUC", "pr_auc"),
    ("Balanced acc", "balanced_accuracy"),
    ("Accuracy", "accuracy"),
    ("Sensitivity", "sensitivity"),
    ("Specificity", "specificity"),
]


def render(summary: pd.DataFrame) -> str:
    header = "| Model | Splits | " + " | ".join(name for name, _ in COLUMNS) + " |"
    rule = "|---|---|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [header, rule]
    for row in summary.itertuples(index=False):
        cells = [
            f"{getattr(row, f'{key}_mean'):.3f} ± {getattr(row, f'{key}_sd'):.3f}"
            for _, key in COLUMNS
        ]
        lines.append(f"| {row.model} | {row.n_splits} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s09_readme_table",
        description="Write the stage-09 mean +- sd table into the README.",
    )
    parser.add_argument("--readme", default=str(PROJECT_ROOT / "README.md"))
    args = parser.parse_args(argv)

    require(CV_SUMMARY_CSV)
    table = render(pd.read_csv(CV_SUMMARY_CSV))

    path = Path(args.readme)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        sys.exit(f"[error] markers {START} / {END} not found in {path}")

    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    path.write_text(f"{head}{START}\n{table}\n{END}{tail}", encoding="utf-8")

    print(table)
    print(f"\n  README table updated -> {path}")


if __name__ == "__main__":
    main()
