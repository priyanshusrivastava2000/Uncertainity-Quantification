from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def _fallback(path: Path) -> Path:
    return path.with_name(f"{path.stem}_new{path.suffix}")


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=index)
    except PermissionError:
        path = _fallback(path)
        df.to_csv(path, index=index)
    return path


def write_excel(sheets: Mapping[str, pd.DataFrame], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def dump(target: Path) -> None:
        with pd.ExcelWriter(target, engine="xlsxwriter") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name, index=False)

    try:
        dump(path)
    except PermissionError:
        path = _fallback(path)
        dump(path)
    return path


def write_text(text: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def upsert_row(row: dict, path: Path, keys=("model",)) -> Path:
    path = Path(path)
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        mask = np.ones(len(old), dtype=bool)
        for key in keys:
            mask &= (old[key] == row[key]).to_numpy()
        df = pd.concat([old[~mask], df], ignore_index=True)
    return write_csv(df, path)


def markdown_table(df: pd.DataFrame) -> str:
    columns = [str(c) for c in df.columns]
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([head, rule] + body)
