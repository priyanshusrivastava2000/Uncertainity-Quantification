from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ..features import all_features, covered
from ..io import upsert_row, write_csv, write_excel
from ..modeling import (
    evaluate,
    out_of_fold_predictions,
    result_row,
    select_lam,
)
from ..paths import (
    HANDOFF_ALL_PATIENTS_CSV,
    HANDOFF_DIR,
    HANDOFF_PREDICTIONS_CSV,
    HANDOFF_RESULTS_CSV,
    HANDOFF_SPLIT_CSV,
    HANDOFF_XLSX,
    conversion_csv,
    ensure_output_dirs,
    require,
)

NARROW_GRID = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
WIDE_GRID = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0]

# a chain member is any of these: one modality, or a fusion of several. Feature blocks
# are concatenated from the per-modality conversion tables, which are row-aligned.
MODELS = {
    "mmse": {"display": "MMSE", "modalities": ("mmse",), "grid": NARROW_GRID},
    "mri": {"display": "MRI", "modalities": ("mri",), "grid": WIDE_GRID},
    "csf": {"display": "CSF", "modalities": ("csf",), "grid": NARROW_GRID},
    "mmse_mri": {"display": "MMSE+MRI", "modalities": ("mmse", "mri"),
                 "grid": WIDE_GRID},
    "mmse_mri_csf": {"display": "MMSE+MRI+CSF", "modalities": ("mmse", "mri", "csf"),
                     "grid": WIDE_GRID},
}
MODALITIES = ("mmse", "mri", "csf")

SWEEP = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)

# the gate a source model must clear to keep a patient defaults to the band threshold;
# MMSE is the exception because its confidence tops out around 0.74, so at 0.80 it would
# hand off every patient and the hop would stop being a selection
SOURCE_GATE = {"mmse": 0.70}

INK, MUTED, SURFACE = "#1a1a1a", "#6b6b6b", "#ffffff"
CELL_COLOURS = {
    "high-correct": "#2c6fbb",
    "high-incorrect": "#c0392b",
    "low-correct": "#a9c8e8",
    "low-incorrect": "#e8b0a8",
}


def name(key: str) -> str:
    return MODELS[key]["display"]


def make_split(y: np.ndarray, test_size: float, seed: int):
    gold = (y > 0).astype(int)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(np.zeros((len(gold), 1)), gold))
    return np.sort(train), np.sort(test)


def assemble_features(key: str, frames: dict, min_coverage: float):
    """Feature block for one chain member, concatenated across its modalities."""
    feats, blocks, n_columns, dropped = [], [], 0, []
    for modality in MODELS[key]["modalities"]:
        df = frames[modality]
        columns = all_features(modality, df)
        kept, _, missing = covered(df, columns, min_coverage)
        overlap = set(kept) & set(feats)
        assert not overlap, f"feature name collision across modalities: {overlap}"
        feats += kept
        blocks.append(df[kept].to_numpy(dtype=float))
        n_columns += len(columns)
        dropped += missing
    return np.hstack(blocks), feats, n_columns, dropped


def fit_model(key: str, frames: dict, y, train, test, args) -> dict:
    """One CLG-Lasso per chain member: lambda from the train half, scored on the test half.

    Every one of the 776 patients also gets a probability: test rows from the model
    fitted on the train half, train rows out-of-fold, so no patient is ever scored by
    a model that saw them.
    """
    display = MODELS[key]["display"]
    X, feats, n_columns, dropped = assemble_features(key, frames, args.min_coverage)
    gold_all = (y > 0).astype(int)

    lam, curve = select_lam(X[train], y[train], MODELS[key]["grid"], k=args.folds,
                            seed=args.cv_seed, verbose=False)
    fitted = out_of_fold_predictions(X, y, train, test, lam, args.folds, args.cv_seed)
    clf = fitted["model"]

    proba_all = fitted["proba"]
    confidence_all = np.maximum(proba_all, 1.0 - proba_all)
    pred_all = (proba_all >= 0.5).astype(int)

    proba = proba_all[test]
    metrics, ci = evaluate(proba, gold_all[test], args.hi, args.boot, args.boot_seed)

    print(f"    {display:<13}{len(feats):>4} features -> {clf.n_nonzero:>3} "
          f"non-zero   lam={lam:<6g} AUC={metrics['roc_auc']:.4f}   "
          f"acc={metrics['accuracy']:.4f}   "
          f"HIGH conf on {100 * metrics['frac_high_conf']:.1f}% of test "
          f"(max confidence {confidence_all[test].max():.3f})"
          + (f"   [{len(dropped)} columns below coverage]" if dropped else ""))

    return {
        "key": key,
        "display": display,
        "lam": lam,
        "n_features": len(feats),
        "n_columns": n_columns,
        "n_nonzero": int(clf.n_nonzero),
        "proba": proba,
        "confidence": confidence_all[test],
        "pred": pred_all[test],
        "correct": (pred_all[test] == gold_all[test]).astype(int),
        "proba_all": proba_all,
        "confidence_all": confidence_all,
        "pred_all": pred_all,
        "correct_all": (pred_all == gold_all).astype(int),
        "source_all": fitted["source"],
        "metrics": metrics,
        "ci": ci,
        "curve": curve,
    }


def transition_label(source_correct: np.ndarray, target_correct: np.ndarray):
    return np.array([
        f"{'right' if s else 'wrong'} -> {'right' if t else 'wrong'}"
        for s, t in zip(source_correct, target_correct)
    ], dtype=object)


def status_of(confidence: np.ndarray, gate: float) -> np.ndarray:
    return np.where(confidence >= gate, "confident", "underconfident")


def all_patients_sheet(fits, base, rid, y, keys, train, source_gates, args):
    """Every patient in the cohort under every chain model, before any handoff."""
    gold = (y > 0).astype(int)
    is_train = np.isin(np.arange(len(rid)), train)
    extra_gates = sorted(set(source_gates) - {args.hi})

    sheet = pd.DataFrame({
        "RID": rid,
        "split": np.where(is_train, "train", "test"),
        "gold": np.where(gold == 1, "converter", "stable"),
        "y": gold,
        "time_to_conv_months": base["time_to_conv_months"].to_numpy(),
        "followup_months": base["followup_months"].to_numpy(),
    })
    for m in keys:
        f = fits[m]
        sheet[f"{m}_p_converter"] = np.round(f["proba_all"], 6)
        sheet[f"{m}_confidence"] = np.round(f["confidence_all"], 6)
        sheet[f"{m}_status_hi{args.hi:.2f}"] = status_of(f["confidence_all"], args.hi)
        for gate in extra_gates:
            sheet[f"{m}_status_gate{gate:.2f}"] = status_of(f["confidence_all"], gate)
        sheet[f"{m}_pred"] = np.where(f["pred_all"] == 1, "converter", "stable")
        sheet[f"{m}_correct"] = f["correct_all"]
        sheet[f"{m}_conf_source"] = f["source_all"]

    confident = np.vstack([fits[m]["confidence_all"] >= args.hi for m in keys])
    correct = np.vstack([fits[m]["correct_all"] for m in keys])
    preds = np.vstack([fits[m]["pred_all"] for m in keys])
    sheet["n_models_confident"] = confident.sum(axis=0)
    sheet["n_models_correct"] = correct.sum(axis=0)
    sheet["models_confident"] = [
        "/".join(name(m) for m, c in zip(keys, column) if c) or "none"
        for column in confident.T
    ]
    sheet["all_models_agree"] = (preds.min(axis=0) == preds.max(axis=0)).astype(int)
    return sheet


def confidence_counts(fits, keys, train_mask, source_gates, args) -> pd.DataFrame:
    """How many patients each classifier is confident about, and how right it is."""
    gates = sorted({args.hi} | set(source_gates) | set(SWEEP))
    rows = []
    for m in keys:
        f = fits[m]
        for split, mask in (("train", train_mask), ("test", ~train_mask),
                            ("all", np.ones(len(train_mask), dtype=bool))):
            confidence = f["confidence_all"][mask]
            correct = f["correct_all"][mask] == 1
            for gate in gates:
                high = confidence >= gate
                rows.append({
                    "model": name(m),
                    "split": split,
                    "n": int(mask.sum()),
                    "gate": gate,
                    "confident_n": int(high.sum()),
                    "confident_pct": round(100.0 * float(high.mean()), 1),
                    "underconfident_n": int((~high).sum()),
                    "underconfident_pct": round(100.0 * float((~high).mean()), 1),
                    "acc_confident": (round(float(correct[high].mean()), 4)
                                      if high.any() else np.nan),
                    "acc_underconfident": (round(float(correct[~high].mean()), 4)
                                           if (~high).any() else np.nan),
                    "confidently_wrong": int((high & ~correct).sum()),
                    "max_confidence": round(float(confidence.max()), 4),
                })
    return pd.DataFrame(rows)


def handed_mask(fits, incoming, source, args) -> np.ndarray:
    """The incoming patients the source model left below its gate."""
    return incoming & (fits[source]["confidence"] < args.hi_source)


def hop_patients(fits, rid, gold, handed, source, target, args) -> pd.DataFrame:
    """One row per patient the source model could not call confidently."""
    s, t = fits[source], fits[target]
    idx = np.flatnonzero(handed)

    became = t["confidence"][idx] >= args.hi
    s_correct = s["correct"][idx].astype(bool)
    t_correct = t["correct"][idx].astype(bool)

    return pd.DataFrame({
        "RID": rid[idx],
        "gold": np.where(gold[idx] == 1, "converter", "stable"),
        "y": gold[idx],
        f"{source}_p_converter": np.round(s["proba"][idx], 6),
        f"{source}_confidence": np.round(s["confidence"][idx], 6),
        f"{source}_pred": np.where(s["pred"][idx] == 1, "converter", "stable"),
        f"{source}_correct": s_correct.astype(int),
        f"{target}_p_converter": np.round(t["proba"][idx], 6),
        f"{target}_confidence": np.round(t["confidence"][idx], 6),
        f"{target}_band": np.where(became, "high", "low"),
        f"{target}_pred": np.where(t["pred"][idx] == 1, "converter", "stable"),
        f"{target}_correct": t_correct.astype(int),
        "confidence_gain": np.round(t["confidence"][idx] - s["confidence"][idx], 6),
        "became_confident": became.astype(int),
        "correctness_move": transition_label(s_correct, t_correct),
        "fixed": (~s_correct & t_correct).astype(int),
        "broken": (s_correct & ~t_correct).astype(int),
        "target_cell": [
            f"{'high' if b else 'low'}-{'correct' if c else 'incorrect'}"
            for b, c in zip(became, t_correct)
        ],
        "confidently_wrong_now": (became & ~t_correct).astype(int),
    }).sort_values("confidence_gain", ascending=False)


def confidence_transition(fits, incoming, source, target, args) -> pd.DataFrame:
    """Source band x target band, over every patient the source model saw."""
    s, t = fits[source], fits[target]
    idx = np.flatnonzero(incoming)
    s_high = s["confidence"][idx] >= args.hi_source
    t_high = t["confidence"][idx] >= args.hi
    rows = []
    for s_band, s_mask in (("high", s_high), ("low", ~s_high)):
        for t_band, t_mask in (("high", t_high), ("low", ~t_high)):
            cell = s_mask & t_mask
            rows.append({
                f"{source}_band": s_band,
                f"{target}_band": t_band,
                "n": int(cell.sum()),
                "pct_of_incoming": round(100.0 * float(cell.mean()), 1),
                f"{target}_correct": int(t["correct"][idx][cell].sum()),
                f"{target}_accuracy": (round(float(t["correct"][idx][cell].mean()), 4)
                                       if cell.any() else np.nan),
                "mean_confidence_gain": (
                    round(float((t["confidence"][idx][cell]
                                 - s["confidence"][idx][cell]).mean()), 4)
                    if cell.any() else np.nan),
            })
    return pd.DataFrame(rows)


def correctness_transition(patients: pd.DataFrame, source, target) -> pd.DataFrame:
    """Source right/wrong x target right/wrong, on the handed-off patients only."""
    s_correct = patients[f"{source}_correct"].to_numpy() == 1
    t_correct = patients[f"{target}_correct"].to_numpy() == 1
    became = patients["became_confident"].to_numpy() == 1
    rows = []
    for s_label, s_mask in (("right", s_correct), ("wrong", ~s_correct)):
        for t_label, t_mask in (("right", t_correct), ("wrong", ~t_correct)):
            cell = s_mask & t_mask
            rows.append({
                f"{source}_was": s_label,
                f"{target}_is": t_label,
                "n": int(cell.sum()),
                "pct_of_handed_off": (round(100.0 * float(cell.mean()), 1)
                                      if len(cell) else np.nan),
                "of_those_now_confident": int((cell & became).sum()),
                "of_those_still_unsure": int((cell & ~became).sum()),
                "converters": int(((patients["y"].to_numpy() == 1) & cell).sum()),
            })
    return pd.DataFrame(rows)


def threshold_sweep(fits, incoming, source, target, args) -> pd.DataFrame:
    """What the hop would look like at other gates - the 0.80 gate is not privileged."""
    s, t = fits[source], fits[target]
    idx = np.flatnonzero(incoming)
    rows = []
    for gate in SWEEP:
        handed = s["confidence"][idx] < gate
        if not handed.any():
            rows.append({"source_gate": gate, "handed_off_n": 0})
            continue
        s_correct = s["correct"][idx][handed] == 1
        t_correct = t["correct"][idx][handed] == 1
        became = t["confidence"][idx][handed] >= args.hi
        rows.append({
            "source_gate": gate,
            "handed_off_n": int(handed.sum()),
            "handed_off_pct": round(100.0 * float(handed.mean()), 1),
            "kept_by_source_n": int((~handed).sum()),
            "kept_by_source_acc": (round(float((s["correct"][idx][~handed] == 1).mean()), 4)
                                   if (~handed).any() else np.nan),
            f"{source}_acc_on_handed": round(float(s_correct.mean()), 4),
            f"{target}_acc_on_handed": round(float(t_correct.mean()), 4),
            "became_confident": int(became.sum()),
            "became_confident_pct": round(100.0 * float(became.mean()), 1),
            "confidently_right": int((became & t_correct).sum()),
            "confidently_wrong": int((became & ~t_correct).sum()),
            "fixed": int((~s_correct & t_correct).sum()),
            "broken": int((s_correct & ~t_correct).sum()),
            "net_change": int((~s_correct & t_correct).sum()) - int((s_correct & ~t_correct).sum()),
        })
    return pd.DataFrame(rows)


def hop_trajectory(hop_frames, args) -> pd.DataFrame:
    """What becomes of each hop-1 outcome group once the next model sees it.

    A patient only reaches the next hop if the current target left them under the band,
    so e.g. the 'right -> wrong' (broken) group carries forward only its unsure members.
    """
    order = ["right -> right", "right -> wrong", "wrong -> right", "wrong -> wrong"]
    rows = []
    for (i, _s1, t1, first), (j, _s2, t2, second) in zip(hop_frames, hop_frames[1:]):
        carried = second.set_index("RID")
        for group in order:
            block = first[first["correctness_move"] == group]
            landed = carried.reindex(block["RID"]).dropna(subset=[f"{t2}_correct"])
            if not len(block):
                continue
            confident = landed[f"{t2}_band"] == "high"
            correct = landed[f"{t2}_correct"] == 1
            was_correct = landed[f"{t1}_correct"] == 1
            rows.append({
                "from_hop": i,
                "to_hop": j,
                "group_after_hop": f"{name(t1)} {group}",
                "n_in_hop": len(block),
                "carried_on": len(landed),
                "answered_by_hop": len(block) - len(landed),
                "confident_right": int((confident & correct).sum()),
                "confident_wrong": int((confident & ~correct).sum()),
                "still_low_right": int((~confident & correct).sum()),
                "still_low_wrong": int((~confident & ~correct).sum()),
                "verdict_changed": int((was_correct != correct).sum()),
                "accuracy_after": (round(float(correct.mean()), 4)
                                   if len(landed) else np.nan),
            })
    return pd.DataFrame(rows)


def hop_summary_row(patients: pd.DataFrame, incoming_n, source, target, args) -> dict:
    became = patients["became_confident"].to_numpy() == 1
    s_correct = patients[f"{source}_correct"].to_numpy() == 1
    t_correct = patients[f"{target}_correct"].to_numpy() == 1
    n = len(patients)
    return {
        "hop": f"{name(source)} -> {name(target)}",
        "source": name(source),
        "target": name(target),
        "incoming_n": incoming_n,
        "source_gate": args.hi_source,
        "target_band": args.hi,
        "kept_by_source": incoming_n - n,
        "handed_off": n,
        "handed_off_pct": round(100.0 * n / incoming_n, 1) if incoming_n else np.nan,
        "converter_rate_handed": round(float((patients["y"] == 1).mean()), 3) if n else np.nan,
        "source_acc_on_handed": round(float(s_correct.mean()), 4) if n else np.nan,
        "target_acc_on_handed": round(float(t_correct.mean()), 4) if n else np.nan,
        "became_confident": int(became.sum()),
        "became_confident_pct": round(100.0 * float(became.mean()), 1) if n else np.nan,
        "still_unsure": int((~became).sum()),
        "confidently_right": int((became & t_correct).sum()),
        "confidently_wrong": int((became & ~t_correct).sum()),
        "fixed_wrong_to_right": int((~s_correct & t_correct).sum()),
        "broken_right_to_wrong": int((s_correct & ~t_correct).sum()),
        "net_correct_change": int((~s_correct & t_correct).sum())
                              - int((s_correct & ~t_correct).sum()),
        "unchanged": int((s_correct == t_correct).sum()),
        "mean_confidence_gain": round(float(patients["confidence_gain"].mean()), 4) if n else np.nan,
    }


def plot_hop(patients: pd.DataFrame, source, target, args, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    became = patients["became_confident"].to_numpy() == 1
    s_correct = patients[f"{source}_correct"].to_numpy() == 1
    t_correct = patients[f"{target}_correct"].to_numpy() == 1
    n = len(patients)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    fig.patch.set_facecolor(SURFACE)

    # left: where the handed-off patients land under the target model
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    left_cells = [
        (f"{name(source)}\n(all LOW conf)", [
            ("right", int(s_correct.sum()), CELL_COLOURS["low-correct"]),
            ("wrong", int((~s_correct).sum()), CELL_COLOURS["low-incorrect"]),
        ]),
        (f"{name(target)}", [
            ("HIGH & right", int((became & t_correct).sum()), CELL_COLOURS["high-correct"]),
            ("HIGH & WRONG", int((became & ~t_correct).sum()), CELL_COLOURS["high-incorrect"]),
            ("low & right", int((~became & t_correct).sum()), CELL_COLOURS["low-correct"]),
            ("low & wrong", int((~became & ~t_correct).sum()), CELL_COLOURS["low-incorrect"]),
        ]),
    ]
    for x, (_column_label, segments) in enumerate(left_cells):
        bottom = 0
        for label, value, colour in segments:
            if not value:
                continue
            ax.bar(x, value, bottom=bottom, width=0.55, color=colour, edgecolor=SURFACE,
                   linewidth=2)
            dark = colour in (CELL_COLOURS["high-correct"], CELL_COLOURS["high-incorrect"])
            if value < 0.08 * n:  # too thin to hold two lines of text
                ax.text(x + 0.31, bottom + value / 2.0, f"{label}  {value}", ha="left",
                        va="center", fontsize=9, color=INK)
            else:
                ax.text(x, bottom + value / 2.0, f"{label}\n{value}", ha="center",
                        va="center", fontsize=9.5, color=("#ffffff" if dark else INK))
            bottom += value
    ax.set_xticks([0, 1])
    ax.set_xlim(-0.5, 1.7)
    ax.set_xticklabels([c[0] for c in left_cells], fontsize=11, color=INK)
    ax.set_ylabel("patients", fontsize=10.5, color=INK)
    ax.set_title(f"the {n} patients {name(source)} could not call", fontsize=11.5,
                 color=INK, pad=12)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color="#ececec", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # right: correctness transition 2x2
    ax = axes[1]
    ax.set_facecolor(SURFACE)
    counts = np.array([
        [int((s_correct & t_correct).sum()), int((s_correct & ~t_correct).sum())],
        [int((~s_correct & t_correct).sum()), int((~s_correct & ~t_correct).sum())],
    ], dtype=float)
    vmax = counts.max() if counts.max() else 1.0
    mesh = ax.pcolormesh(counts[::-1], cmap=plt.cm.Blues, vmin=0, vmax=vmax,
                         edgecolors=SURFACE, linewidth=4)
    notes = [["stayed right", "BROKEN"], ["RESCUED", "stayed wrong"]]
    for row in range(2):
        for col in range(2):
            value = counts[row, col]
            dark = value > 0.5 * vmax
            colour = "#ffffff" if dark else INK
            ax.text(col + 0.5, (1 - row) + 0.58, f"{int(value)}", ha="center", va="center",
                    fontsize=26, fontweight="bold", color=colour)
            ax.text(col + 0.5, (1 - row) + 0.32, notes[row][col], ha="center", va="center",
                    fontsize=10, style="italic",
                    color=("#eaeaea" if dark else MUTED))
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels([f"{name(target)} right", f"{name(target)} wrong"],
                       fontsize=11, color=INK)
    ax.set_yticklabels([f"{name(source)}\nwrong", f"{name(source)}\nright"],
                       fontsize=11, color=INK)
    ax.tick_params(length=0)
    ax.xaxis.set_ticks_position("top")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.03)
    cbar.ax.tick_params(labelsize=9, color=MUTED, labelcolor=MUTED)
    cbar.outline.set_visible(False)

    fig.text(0.5, 0.965,
             f"Handing the unsure patients from {name(source)} to {name(target)}",
             ha="center", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.915,
             f"{n} of the test half had {name(source)} confidence < {args.hi_source:.2f}"
             f"   -   {int(became.sum())} reach confidence >= {args.hi:.2f} under "
             f"{name(target)}   -   "
             f"{int((~s_correct & t_correct).sum())} fixed / "
             f"{int((s_correct & ~t_correct).sum())} broken",
             ha="center", fontsize=9.5, color=MUTED)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def build_parser(prog: str, description: str, default_chain: list) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--chain", nargs="+", default=default_chain,
                        choices=list(MODELS),
                        help="handoff order; each hop takes the patients the previous "
                             "model could not call confidently")
    parser.add_argument("--test-size", type=float, default=0.50,
                        help="held-out fraction of the 776 tri-modal patients")
    parser.add_argument("--seed", type=int, default=42, help="seed for the 50-50 split")
    parser.add_argument("--hi", type=float, default=0.80,
                        help="confidence at or above which a patient is HIGH band")
    parser.add_argument("--hi-source", type=float, nargs="+", default=None,
                        help="gate the source model must clear to keep a patient - one "
                             "value, or one per hop. Defaults to the band threshold for "
                             "every model except MMSE, which cannot exceed ~0.74 and so "
                             "defaults to 0.70")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    return parser


def resolve_gates(parser: argparse.ArgumentParser, args) -> list:
    """One source gate per hop, defaulting per source model."""
    if len(args.chain) < 2:
        parser.error("--chain needs at least two models")
    if len(set(args.chain)) != len(args.chain):
        parser.error(f"--chain repeats a model: {args.chain}")

    n_hops = len(args.chain) - 1
    if args.hi_source is None:
        source_gates = [SOURCE_GATE.get(s, args.hi) for s in args.chain[:-1]]
    elif len(args.hi_source) == 1:
        source_gates = args.hi_source * n_hops
    else:
        source_gates = args.hi_source
    if len(source_gates) != n_hops:
        parser.error(f"--hi-source needs 1 or {n_hops} values, got {len(source_gates)}")
    # every hop reads args.hi_source; the loop re-points it at that hop's gate
    args.hi_source = source_gates[0]
    return source_gates


HANDOFF_PATHS = {
    "dir": HANDOFF_DIR,
    "split_csv": HANDOFF_SPLIT_CSV,
    "all_patients_csv": HANDOFF_ALL_PATIENTS_CSV,
    "predictions_csv": HANDOFF_PREDICTIONS_CSV,
    "results_csv": HANDOFF_RESULTS_CSV,
    "xlsx": HANDOFF_XLSX,
}


def main(argv: list | None = None) -> None:
    parser = build_parser(
        "s13_modality_handoff",
        "Train MMSE / MRI / CSF on a 50-50 split and hand the patients one model is "
        "unsure about to the next.",
        ["mmse", "mri", "csf"],
    )
    args = parser.parse_args(argv)
    source_gates = resolve_gates(parser, args)
    run_handoff(args, source_gates, HANDOFF_PATHS,
                "MODALITY HANDOFF  -  three single-modality classifiers on a 50-50 split")


def run_handoff(args, source_gates, paths, title: str) -> None:
    ensure_output_dirs()
    paths["dir"].mkdir(parents=True, exist_ok=True)
    modalities = [m for m in MODALITIES
                  if any(m in MODELS[k]["modalities"] for k in args.chain)]
    require(*[conversion_csv(m) for m in modalities])

    frames = {m: pd.read_csv(conversion_csv(m), low_memory=False) for m in modalities}
    rids = frames[modalities[0]]["RID"].tolist()
    for m in modalities[1:]:
        assert frames[m]["RID"].tolist() == rids, \
            f"the {m} conversion file is not row-aligned with {modalities[0]}"

    base = frames[modalities[0]]
    y = np.where(base["y"].to_numpy() == 1, 1.0, -1.0)
    rid = base["RID"].to_numpy()
    train, test = make_split(y, args.test_size, args.seed)
    gold = (y[test] > 0).astype(int)
    majority = max(gold.mean(), 1 - gold.mean())

    print("=" * 92)
    print(title)
    print("=" * 92)
    print(f"  cohort   : {len(base)} tri-modal patients   "
          f"{int((y > 0).sum())} converters ({100 * (y > 0).mean():.1f}%)")
    print(f"  split    : {len(train)} train / {len(test)} test  "
          f"(stratified {100 * (1 - args.test_size):.0f}/{100 * args.test_size:.0f}, "
          f"seed {args.seed}) - independent of the canonical 70/30")
    print(f"  test half: {int(gold.sum())} converters ({100 * gold.mean():.1f}%)   "
          f"majority {majority:.4f}")
    print(f"  bands    : target band {args.hi:.2f}   source gates "
          + ", ".join(f"{name(s)}>={g:.2f}"
                      for s, g in zip(args.chain, source_gates)))
    print(f"  chain    : " + " -> ".join(name(m) for m in args.chain))

    print(f"\n  fitting {len(args.chain)} classifiers "
          f"(lambda searched inside the train half only):")
    fits = {k: fit_model(k, frames, y, train, test, args) for k in args.chain}
    keys = list(args.chain)

    predictions = pd.DataFrame({
        "RID": rid[test],
        "gold": np.where(gold == 1, "converter", "stable"),
        "y": gold,
        "conv_label": base["conv_label"].to_numpy()[test],
        "time_to_conv_months": base["time_to_conv_months"].to_numpy()[test],
    })
    for m in keys:
        f = fits[m]
        predictions[f"{m}_p_converter"] = np.round(f["proba"], 6)
        predictions[f"{m}_confidence"] = np.round(f["confidence"], 6)
        predictions[f"{m}_band"] = np.where(f["confidence"] >= args.hi, "high", "low")
        predictions[f"{m}_pred"] = np.where(f["pred"] == 1, "converter", "stable")
        predictions[f"{m}_correct"] = f["correct"]

    model_rows = []
    for m in keys:
        f = fits[m]
        row = {
            "model": f"{name(m)} (50-50 split)",
            "modality": name(m),
            "n_columns": f["n_columns"],
            "n_features": f["n_features"],
            "n_nonzero": f["n_nonzero"],
            "lam": f["lam"],
            "cohort_n": len(base),
            "train_n": len(train),
            "test_n": len(test),
            "max_confidence": round(float(f["confidence"].max()), 4),
            **result_row(f["metrics"], f["ci"]),
        }
        model_rows.append(row)
        upsert_row(row, paths["results_csv"])
    models = pd.DataFrame(model_rows)

    train_mask = np.isin(np.arange(len(rid)), train)
    all_patients = all_patients_sheet(fits, base, rid, y, keys, train,
                                      source_gates, args)
    counts = confidence_counts(fits, keys, train_mask, source_gates, args)

    print("\n" + "=" * 92)
    print(f"ALL {len(base)} PATIENTS  -  confident vs underconfident, per classifier")
    print("=" * 92)
    for gate in sorted({args.hi} | set(source_gates)):
        print(f"\n  at confidence >= {gate:.2f}:")
        print(f"    {'model':<7}{'CONFIDENT':>16}{'acc':>8}{'conf.wrong':>12}   "
              f"{'UNDERCONFIDENT':>16}{'acc':>8}{'max conf':>10}")
        for m in keys:
            r = counts[(counts["model"] == name(m)) & (counts["split"] == "all")
                       & (counts["gate"] == gate)].iloc[0]
            print(f"    {r.model:<7}{f'{r.confident_n} ({r.confident_pct}%)':>16}"
                  f"{r.acc_confident:>8.4f}{r.confidently_wrong:>12}   "
                  f"{f'{r.underconfident_n} ({r.underconfident_pct}%)':>16}"
                  f"{r.acc_underconfident:>8.4f}{r.max_confidence:>10.3f}")

    sheets = {
        "all_patients": all_patients,
        "confidence_counts": counts,
        "test_predictions": predictions,
        "model_summary": models,
    }
    hop_rows, figures, hop_frames = [], [], []
    incoming = np.ones(len(test), dtype=bool)

    for i, (source, target) in enumerate(zip(args.chain, args.chain[1:]), start=1):
        args.hi_source = source_gates[i - 1]
        incoming_n = int(incoming.sum())
        handed = handed_mask(fits, incoming, source, args)
        patients = hop_patients(fits, rid[test], gold, handed, source, target, args)
        # sheet names are capped at 31 chars by Excel, and fusion keys are long, so the
        # sheets are numbered and hop_summary carries the source/target names
        tag = f"hop{i}"
        figure_tag = f"hop{i}_{source}_to_{target}"

        print("\n" + "-" * 92)
        print(f"HOP {i}  -  {name(source)} -> {name(target)}")
        print("-" * 92)
        if not len(patients):
            print(f"  {name(source)} was confident (>= {args.hi_source:.2f}) about all "
                  f"{incoming_n} incoming patients - nothing to hand off.")
            incoming = np.zeros(len(test), dtype=bool)
            continue

        summary = hop_summary_row(patients, incoming_n, source, target, args)
        hop_rows.append(summary)
        sheets[f"{tag}_patients"] = patients
        hop_frames.append((i, source, target, patients))
        sheets[f"{tag}_conf_move"] = confidence_transition(
            fits, incoming, source, target, args)
        sheets[f"{tag}_correct_move"] = correctness_transition(patients, source, target)
        sheets[f"{tag}_sweep"] = threshold_sweep(fits, incoming, source, target, args)

        png = paths["dir"] / f"{figure_tag}.png"
        plot_hop(patients, source, target, args, png)
        figures.append(png)

        print(f"  incoming            : {incoming_n}")
        print(f"  {name(source)} kept (conf >= {args.hi_source:.2f}): "
              f"{summary['kept_by_source']}")
        print(f"  handed to {name(target):<9}: {summary['handed_off']} "
              f"({summary['handed_off_pct']}%)   converter rate "
              f"{summary['converter_rate_handed']}")
        if not summary["kept_by_source"]:
            ceiling = float(fits[source]["confidence"].max())
            reason = (
                f"{name(source)} never reaches confidence {args.hi_source:.2f} "
                f"(max {ceiling:.3f})"
                if ceiling < args.hi_source else
                f"every incoming patient was already below {args.hi_source:.2f} by "
                f"construction - hop {i - 1} passed on exactly the patients "
                f"{name(source)} left under the {args.hi:.2f} band"
            )
            print(f"  [note] {reason}, so this hop forwards the whole incoming set "
                  f"rather than selecting a subset of it - see the {tag}_sweep sheet "
                  f"for gates that split it")
        print(f"\n  DID {name(target)} BECOME CONFIDENT ABOUT THEM?")
        print(f"    reached conf >= {args.hi:.2f} : {summary['became_confident']} "
              f"({summary['became_confident_pct']}%)   "
              f"of those, {summary['confidently_right']} right / "
              f"{summary['confidently_wrong']} CONFIDENTLY WRONG")
        print(f"    still below the gate  : {summary['still_unsure']}")
        print(f"    mean confidence gain  : {summary['mean_confidence_gain']:+.4f}")
        print(f"\n  DID THE ANSWER CHANGE?  (accuracy on these {summary['handed_off']}: "
              f"{name(source)} {summary['source_acc_on_handed']:.4f} -> "
              f"{name(target)} {summary['target_acc_on_handed']:.4f})")
        print(f"    wrong -> right (fixed) : {summary['fixed_wrong_to_right']}")
        print(f"    right -> wrong (broken): {summary['broken_right_to_wrong']}")
        print(f"    unchanged              : {summary['unchanged']}   "
              f"net {summary['net_correct_change']:+d} patients")
        print("\n  correctness transition:")
        print(sheets[f"{tag}_correct_move"].to_string(index=False))

        # the next hop only sees the patients the target model is still unsure about
        incoming = handed & (fits[target]["confidence"] < args.hi)

    if hop_rows:
        sheets["hop_summary"] = pd.DataFrame(hop_rows)
    trajectory = hop_trajectory(hop_frames, args) if len(hop_frames) > 1 else None
    if trajectory is not None and len(trajectory):
        sheets["hop_trajectory"] = trajectory
        print("\n" + "=" * 92)
        print("TRAJECTORY  -  what the next model does to each group the last one created")
        print("=" * 92)
        print(trajectory.to_string(index=False))

    config = pd.DataFrame({
        "field": ["cohort", "split", "split_seed", "train_n", "test_n", "chain",
                  "source_gate", "target_band", "lambda_selection", "preprocessing",
                  "coverage_floor",
                  *[f"lambda_{k}" for k in keys],
                  *[f"max_confidence_{k}" for k in keys],
                  "note_split", "note_confidence", "note_all_patients", "note_gate",
                  "note_chain"],
        "value": [
            f"{len(base)} patients with MMSE + MRI + CSF at t=0",
            f"stratified {100 * (1 - args.test_size):.0f}/{100 * args.test_size:.0f} "
            f"on the conversion label",
            args.seed, len(train), len(test),
            " -> ".join(name(m) for m in args.chain),
            " / ".join(
                f"{name(s)}>={g:.2f}"
                for s, g in zip(args.chain, source_gates)
            ),
            args.hi,
            f"{args.folds}-fold CV inside the train half only (seed {args.cv_seed})",
            "median imputation + standardisation fitted on the train half only",
            args.min_coverage,
            *[fits[k]["lam"] for k in keys],
            *[round(float(fits[k]["confidence"].max()), 4) for k in keys],
            "this is a fresh 50-50 split, NOT the canonical 70/30 split used by stages "
            "04-12. Numbers here are not comparable to the headline table: the models "
            "see ~30% fewer training patients and are scored on a different test set.",
            "every model scores the same held-out patients with a model fitted on the "
            "disjoint train half, so every test-side confidence is out-of-sample",
            f"the all_patients sheet covers all {len(base)} patients: the {len(test)} "
            f"test rows carry the fitted model's probability (as in deployment) and the "
            f"{len(train)} train rows carry a {args.folds}-fold out-of-fold probability, "
            "because a model's confidence about a patient it was fitted on is optimistic. "
            "The conf_source column records which. Compare bands within a split, not "
            "across splits; the handoff analysis uses the test half only.",
            "the source gate is separate from the band threshold because the MMSE model "
            "never exceeds ~0.74 confidence: at a 0.80 gate it hands off every patient "
            "and the hop stops being a selection. The default 0.70 is the same value "
            "stage 05 uses for the same reason. Every other gate is tabulated in the "
            "sweep sheet.",
            "each hop is fed only the patients the previous model left below the gate, "
            "so hop 2 sees the intersection of both models' low-confidence sets",
        ],
    })
    sheets["run_config"] = config

    write_csv(all_patients, paths["all_patients_csv"])
    write_csv(predictions, paths["predictions_csv"])
    write_csv(pd.DataFrame({
        "RID": rid,
        "split": np.where(np.isin(np.arange(len(rid)), train), "train", "test"),
        "y": (y > 0).astype(int),
        "conv_label": base["conv_label"],
    }), paths["split_csv"])
    xlsx = write_excel(sheets, paths["xlsx"])

    print("\n" + "=" * 92)
    print(f"THREE MODELS ON THE SAME {len(test)} HELD-OUT PATIENTS  "
          f"(majority {majority:.4f})")
    print("=" * 92)
    print(f"  {'model':<8}{'feat':>6}{'nz':>5}{'lam':>7}{'AUC':>8}{'95% CI':>17}"
          f"{'PR-AUC':>9}{'bal.acc':>9}{'acc':>8}{'max conf':>10}{'%HIGH':>8}")
    for r in models.sort_values("roc_auc", ascending=False).itertuples(index=False):
        print(f"  {r.modality:<8}{r.n_features:>6}{r.n_nonzero:>5}{r.lam:>7g}"
              f"{r.roc_auc:>8.4f}  [{r.auc_ci_lo:.3f},{r.auc_ci_hi:.3f}]"
              f"{r.pr_auc:>9.4f}{r.balanced_accuracy:>9.4f}{r.accuracy:>8.4f}"
              f"{r.max_confidence:>10.3f}{100 * r.frac_high_conf:>7.1f}%")

    if hop_rows:
        print("\n  handoff summary:")
        print(pd.DataFrame(hop_rows).to_string(index=False))

    print(f"\n  written to {paths["dir"]}:")
    for path in [paths["split_csv"], paths["all_patients_csv"], paths["predictions_csv"],
                 paths["results_csv"], xlsx, *figures]:
        print(f"    {Path(path).name}")


if __name__ == "__main__":
    main()
