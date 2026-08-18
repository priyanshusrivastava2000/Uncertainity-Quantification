from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import write_csv, write_excel
from ..paths import (
    CONFIDENT_AUDIT_DIR,
    confident_audit_path,
    conversion_csv,
    ensure_output_dirs,
    require,
)
from .s13_modality_handoff import (
    CELL_COLOURS,
    INK,
    MODALITIES,
    MODELS,
    MUTED,
    SOURCE_GATE,
    SURFACE,
    SWEEP,
    fit_model,
    make_split,
    name,
    transition_label,
)

# stage 13 routes the patients the source model could NOT call. This is the other half of
# that question: the patients it COULD call are kept and never see a second opinion, so
# nobody knows what that second opinion would have been. Here they get one anyway - the
# already-fitted target model scores them - and the two failure modes are counted:
#   rescued = source was WRONG about a patient it felt sure of, target gets it right
#   broken  = source was RIGHT, target would have overturned a good call
# Nothing is re-fitted: same 50-50 split, same seeds, same lambda grid as stage 13, so the
# kept set here is exactly the complement of stage 13's hop-1 handed-off set.

# the kept set can sit in a very narrow confidence range - MMSE tops out around 0.74, so
# fixed bins put every kept patient in one bucket. Quartiles of the kept set itself keep
# the resolution whatever the source model's ceiling is.
CONF_QUANTILES = [0.0, 0.25, 0.50, 0.75, 1.0]


def kept_patients(fits, rid, gold, kept, source, target, args) -> pd.DataFrame:
    """One row per patient the source model was confident enough to keep."""
    s, t = fits[source], fits[target]
    idx = np.flatnonzero(kept)

    t_high = t["confidence"][idx] >= args.hi
    s_correct = s["correct"][idx].astype(bool)
    t_correct = t["correct"][idx].astype(bool)
    disagree = s["pred"][idx] != t["pred"][idx]

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
        f"{target}_band": np.where(t_high, "high", "low"),
        f"{target}_pred": np.where(t["pred"][idx] == 1, "converter", "stable"),
        f"{target}_correct": t_correct.astype(int),
        "confidence_change": np.round(t["confidence"][idx] - s["confidence"][idx], 6),
        "models_disagree": disagree.astype(int),
        "correctness_move": transition_label(s_correct, t_correct),
        "rescued": (~s_correct & t_correct).astype(int),
        "broken": (s_correct & ~t_correct).astype(int),
        # a disagreement only costs anything if a policy would act on it, and a policy
        # that overturns a confident call on a shaky second opinion is not worth running
        "rescued_by_confident_target": (~s_correct & t_correct & t_high).astype(int),
        "broken_by_confident_target": (s_correct & ~t_correct & t_high).astype(int),
        f"{source}_confidently_wrong": (~s_correct).astype(int),
    }).sort_values([f"{source}_correct", f"{source}_confidence"],
                   ascending=[True, False])


def correctness_move(patients, source, target) -> pd.DataFrame:
    """Source right/wrong x target right/wrong, on the kept patients only."""
    s_correct = patients[f"{source}_correct"].to_numpy() == 1
    t_correct = patients[f"{target}_correct"].to_numpy() == 1
    t_high = patients[f"{target}_band"].to_numpy() == "high"
    meanings = {
        ("right", "right"): "agreed, stayed right",
        ("right", "wrong"): "BROKEN",
        ("wrong", "right"): "RESCUED",
        ("wrong", "wrong"): "both wrong",
    }
    n = len(patients)
    rows = []
    for s_label, s_mask in (("right", s_correct), ("wrong", ~s_correct)):
        for t_label, t_mask in (("right", t_correct), ("wrong", ~t_correct)):
            cell = s_mask & t_mask
            rows.append({
                f"{source}_was": s_label,
                f"{target}_is": t_label,
                "meaning": meanings[(s_label, t_label)],
                "n": int(cell.sum()),
                "pct_of_kept": round(100.0 * float(cell.mean()), 1) if n else np.nan,
                f"{target}_confident": int((cell & t_high).sum()),
                f"{target}_unsure": int((cell & ~t_high).sum()),
                "converters": int(((patients["y"].to_numpy() == 1) & cell).sum()),
            })
    return pd.DataFrame(rows)


def target_band_breakdown(patients, source, target, args) -> pd.DataFrame:
    """Does the target model at least know when it is about to overturn a good call?"""
    t_high = patients[f"{target}_band"].to_numpy() == "high"
    s_correct = patients[f"{source}_correct"].to_numpy() == 1
    t_correct = patients[f"{target}_correct"].to_numpy() == 1
    disagree = patients["models_disagree"].to_numpy() == 1
    rows = []
    for label, mask in ((f"high (>= {args.hi:.2f})", t_high),
                        (f"low (< {args.hi:.2f})", ~t_high)):
        rescued = int((mask & ~s_correct & t_correct).sum())
        broken = int((mask & s_correct & ~t_correct).sum())
        rows.append({
            f"{target}_band": label,
            "n": int(mask.sum()),
            "pct_of_kept": round(100.0 * float(mask.mean()), 1) if len(mask) else np.nan,
            f"{source}_accuracy": (round(float(s_correct[mask].mean()), 4)
                                   if mask.any() else np.nan),
            f"{target}_accuracy": (round(float(t_correct[mask].mean()), 4)
                                   if mask.any() else np.nan),
            "rescued": rescued,
            "broken": broken,
            "net": rescued - broken,
            "disagreements": int((mask & disagree).sum()),
        })
    return pd.DataFrame(rows)


def confidence_bins(patients, source, target) -> pd.DataFrame:
    """Where in the kept set the damage sits - just above the gate, or right at the top."""
    conf = patients[f"{source}_confidence"].to_numpy()
    s_correct = patients[f"{source}_correct"].to_numpy() == 1
    t_correct = patients[f"{target}_correct"].to_numpy() == 1
    edges = np.unique(np.quantile(conf, CONF_QUANTILES))
    rows = []
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        last = i == len(edges) - 2
        cell = (conf >= lo) & (conf <= hi if last else conf < hi)
        if not cell.any():
            continue
        rescued = int((cell & ~s_correct & t_correct).sum())
        broken = int((cell & s_correct & ~t_correct).sum())
        rows.append({
            f"{source}_confidence_bin": f"[{lo:.4f}, {hi:.4f}{']' if last else ')'}",
            "n": int(cell.sum()),
            f"{source}_accuracy": round(float(s_correct[cell].mean()), 4),
            f"{target}_accuracy": round(float(t_correct[cell].mean()), 4),
            "rescued": rescued,
            "broken": broken,
            "net": rescued - broken,
        })
    return pd.DataFrame(rows)


def policy_comparison(patients, source, target, args) -> pd.DataFrame:
    """Three ways to answer the kept patients, scored on the same rows."""
    s_correct = patients[f"{source}_correct"].to_numpy() == 1
    t_correct = patients[f"{target}_correct"].to_numpy() == 1
    t_high = patients[f"{target}_band"].to_numpy() == "high"
    n = len(patients)

    # override only where the second opinion is itself above the band
    selective = np.where(t_high, t_correct, s_correct)

    policies = [
        (f"keep {name(source)} (what stage 13 does)", s_correct,
         "the confident call stands, no second opinion"),
        (f"always defer to {name(target)}", t_correct,
         f"{name(target)} answers every kept patient"),
        (f"defer to {name(target)} only when it is confident", selective,
         f"{name(target)} overrides only at confidence >= {args.hi:.2f}"),
    ]
    return pd.DataFrame([{
        "policy": label,
        "n": n,
        "correct": int(correct.sum()),
        "accuracy": round(float(correct.mean()), 4) if n else np.nan,
        "vs_keep_source": int(correct.sum()) - int(s_correct.sum()),
        "note": note,
    } for label, correct, note in policies])


def gate_sweep(fits, source, target, args, incoming) -> pd.DataFrame:
    """The kept set is a function of the gate - how the trade moves as the gate moves."""
    s, t = fits[source], fits[target]
    s_correct = s["correct"] == 1
    t_correct = t["correct"] == 1
    t_high = t["confidence"] >= args.hi
    rows = []
    for gate in sorted({args.gate} | set(SWEEP)):
        kept = incoming & (s["confidence"] >= gate)
        rescued = int((kept & ~s_correct & t_correct).sum())
        broken = int((kept & s_correct & ~t_correct).sum())
        # a gate above the source model's ceiling keeps nobody; the row stays so the
        # sweep shows where the model runs out of confidence rather than skipping it
        rows.append({
            "source_gate": gate,
            "kept_n": int(kept.sum()),
            "kept_pct": round(100.0 * kept.sum() / max(int(incoming.sum()), 1), 1),
            "handed_off_n": int((incoming & ~kept).sum()),
            f"{source}_acc_on_kept": (round(float(s_correct[kept].mean()), 4)
                                      if kept.any() else np.nan),
            f"{target}_acc_on_kept": (round(float(t_correct[kept].mean()), 4)
                                      if kept.any() else np.nan),
            "rescued": rescued,
            "broken": broken,
            "net_change": rescued - broken,
            "rescued_confident_target": int((kept & ~s_correct & t_correct & t_high).sum()),
            "broken_confident_target": int((kept & s_correct & ~t_correct & t_high).sum()),
            f"{source}_confidently_wrong": int((kept & ~s_correct).sum()),
        })
    return pd.DataFrame(rows)


def split_check(fits, train_mask, source, target, args, incoming_all) -> pd.DataFrame:
    """The same audit on train (out-of-fold) and on all 776, as a stability check."""
    s, t = fits[source], fits[target]
    s_correct = s["correct_all"] == 1
    t_correct = t["correct_all"] == 1
    rows = []
    for split, mask in (("train (out-of-fold)", train_mask),
                        ("test (held out)", ~train_mask),
                        ("all", np.ones(len(train_mask), dtype=bool))):
        mask = mask & incoming_all
        kept = mask & (s["confidence_all"] >= args.gate)
        rescued = int((kept & ~s_correct & t_correct).sum())
        broken = int((kept & s_correct & ~t_correct).sum())
        rows.append({
            "split": split,
            "incoming_n": int(mask.sum()),
            "kept_n": int(kept.sum()),
            "kept_pct": round(100.0 * kept.sum() / max(int(mask.sum()), 1), 1),
            f"{source}_acc_on_kept": (round(float(s_correct[kept].mean()), 4)
                                      if kept.any() else np.nan),
            f"{target}_acc_on_kept": (round(float(t_correct[kept].mean()), 4)
                                      if kept.any() else np.nan),
            "rescued": rescued,
            "broken": broken,
            "net_change": rescued - broken,
        })
    return pd.DataFrame(rows)


def summary_row(patients, incoming_n, source, target, args) -> dict:
    s_correct = patients[f"{source}_correct"].to_numpy() == 1
    t_correct = patients[f"{target}_correct"].to_numpy() == 1
    t_high = patients[f"{target}_band"].to_numpy() == "high"
    n = len(patients)
    return {
        "audit": f"{name(source)} confident -> {name(target)}",
        "source": name(source),
        "target": name(target),
        "upstream": " -> ".join(name(u) for u in args.upstream) or "none (whole test half)",
        "incoming_n": incoming_n,
        "source_gate": args.gate,
        "target_band": args.hi,
        "kept_by_source": n,
        "kept_pct": round(100.0 * n / incoming_n, 1) if incoming_n else np.nan,
        "handed_off_by_stage13": incoming_n - n,
        "converter_rate_kept": (round(float((patients["y"] == 1).mean()), 3)
                                if n else np.nan),
        "source_acc_on_kept": round(float(s_correct.mean()), 4) if n else np.nan,
        "target_acc_on_kept": round(float(t_correct.mean()), 4) if n else np.nan,
        "source_confidently_wrong": int((~s_correct).sum()),
        "rescued_wrong_to_right": int((~s_correct & t_correct).sum()),
        "broken_right_to_wrong": int((s_correct & ~t_correct).sum()),
        "net_correct_change": int((~s_correct & t_correct).sum())
                              - int((s_correct & ~t_correct).sum()),
        "unchanged": int((s_correct == t_correct).sum()),
        "models_disagree": int((patients["models_disagree"] == 1).sum()),
        "target_confident": int(t_high.sum()),
        "target_confidently_wrong": int((t_high & ~t_correct).sum()),
        "rescued_by_confident_target": int((~s_correct & t_correct & t_high).sum()),
        "broken_by_confident_target": int((s_correct & ~t_correct & t_high).sum()),
        "mean_confidence_change": (round(float(patients["confidence_change"].mean()), 4)
                                   if n else np.nan),
    }


def plot_audit(patients, source, target, args, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s_correct = patients[f"{source}_correct"].to_numpy() == 1
    t_correct = patients[f"{target}_correct"].to_numpy() == 1
    t_high = patients[f"{target}_band"].to_numpy() == "high"
    n = len(patients)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    fig.patch.set_facecolor(SURFACE)

    # left: what the kept patients look like before and after the second opinion
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    columns = [
        (f"{name(source)}\n(all kept, conf >= {args.gate:.2f})", [
            ("right", int(s_correct.sum()), CELL_COLOURS["high-correct"]),
            # the column header already says every one of these was above the gate, so
            # the segment only has to carry right/wrong - the long label overruns the bar
            ("WRONG", int((~s_correct).sum()), CELL_COLOURS["high-incorrect"]),
        ]),
        (f"{name(target)}\nsecond opinion", [
            ("HIGH & right", int((t_high & t_correct).sum()), CELL_COLOURS["high-correct"]),
            ("HIGH & WRONG", int((t_high & ~t_correct).sum()),
             CELL_COLOURS["high-incorrect"]),
            ("low & right", int((~t_high & t_correct).sum()), CELL_COLOURS["low-correct"]),
            ("low & wrong", int((~t_high & ~t_correct).sum()),
             CELL_COLOURS["low-incorrect"]),
        ]),
    ]
    for x, (_column_label, segments) in enumerate(columns):
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
    ax.set_xticklabels([c[0] for c in columns], fontsize=11, color=INK)
    ax.set_ylabel("patients", fontsize=10.5, color=INK)
    ax.set_title(f"the {n} patients {name(source)} was sure about", fontsize=11.5,
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
    notes = [["stayed right", "BROKEN"], ["RESCUED", "both wrong"]]
    for row in range(2):
        for col in range(2):
            value = counts[row, col]
            dark = value > 0.5 * vmax
            colour = "#ffffff" if dark else INK
            ax.text(col + 0.5, (1 - row) + 0.58, f"{int(value)}", ha="center", va="center",
                    fontsize=26, fontweight="bold", color=colour)
            ax.text(col + 0.5, (1 - row) + 0.32, notes[row][col], ha="center", va="center",
                    fontsize=10, style="italic", color=("#eaeaea" if dark else MUTED))
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
             f"Second opinion on the patients {name(source)} kept: "
             f"{name(source)} confident -> {name(target)}",
             ha="center", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.915,
             f"{n} of the test half had {name(source)} confidence >= {args.gate:.2f}"
             f"   -   accuracy {name(source)} {float(s_correct.mean()):.3f} vs "
             f"{name(target)} {float(t_correct.mean()):.3f}   -   "
             f"{int((~s_correct & t_correct).sum())} rescued / "
             f"{int((s_correct & ~t_correct).sum())} broken",
             ha="center", fontsize=9.5, color=MUTED)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s15_confident_audit",
        description="Take the patients the source model was CONFIDENT about - the ones "
                    "stage 13 keeps and never routes - and score them with the "
                    "already-fitted target model to count rescues and breakages.")
    parser.add_argument("--source", default="mmse", choices=list(MODELS),
                        help="the model whose confident patients are audited")
    parser.add_argument("--target", default="mri", choices=list(MODELS),
                        help="the already-trained model asked for a second opinion")
    parser.add_argument("--upstream", nargs="*", default=[], choices=list(MODELS),
                        help="models that ran before the source in the stage 13 chain. "
                             "The audit is then restricted to the patients those models "
                             "handed on, so the kept set is exactly the one the source "
                             "model keeps mid-chain rather than over the whole test half. "
                             "Each upstream model gates at its stage 13 default")
    parser.add_argument("--gate", type=float, default=None,
                        help="confidence the source must reach to keep a patient; "
                             "defaults to the stage 13 gate for that model "
                             "(0.70 for MMSE, otherwise the band threshold)")
    parser.add_argument("--test-size", type=float, default=0.50,
                        help="held-out fraction of the 776 tri-modal patients")
    parser.add_argument("--seed", type=int, default=42, help="seed for the 50-50 split")
    parser.add_argument("--hi", type=float, default=0.80,
                        help="confidence at or above which a patient is HIGH band")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    return parser


def main(argv: list | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source == args.target:
        parser.error("--source and --target must be different models")
    chain = list(args.upstream) + [args.source, args.target]
    if len(set(chain)) != len(chain):
        parser.error(f"--upstream repeats the source, the target or itself: {chain}")
    if args.gate is None:
        args.gate = SOURCE_GATE.get(args.source, args.hi)
    # stage 13's fit_model prints against args.hi_source; keep the two in step
    args.hi_source = args.gate
    upstream_gates = [SOURCE_GATE.get(u, args.hi) for u in args.upstream]

    source, target = args.source, args.target
    ensure_output_dirs()
    CONFIDENT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    needed = sum((MODELS[k]["modalities"] for k in chain), ())
    modalities = [m for m in MODALITIES if m in needed]
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

    print("=" * 92)
    print(f"CONFIDENT-SET AUDIT  -  {name(source)} keeps them, {name(target)} "
          f"gets a look anyway")
    print("=" * 92)
    print(f"  cohort   : {len(base)} tri-modal patients   "
          f"{int((y > 0).sum())} converters ({100 * (y > 0).mean():.1f}%)")
    print(f"  split    : {len(train)} train / {len(test)} test  "
          f"(stratified, seed {args.seed}) - the same split stage 13 uses")
    print(f"  gate     : {name(source)} confidence >= {args.gate:.2f}  "
          f"(these are the patients stage 13 does NOT hand off)")
    print(f"  band     : {name(target)} counts as confident at >= {args.hi:.2f}")
    if args.upstream:
        print(f"  upstream : " + " -> ".join(
            f"{name(u)}<{g:.2f}" for u, g in zip(args.upstream, upstream_gates))
            + f" -> {name(source)}   (only the patients those models handed on are audited)")
    else:
        print(f"  upstream : none - the pool is the whole test half")

    print(f"\n  fitting {len(chain)} classifiers "
          f"(lambda searched inside the train half only):")
    fits = {k: fit_model(k, frames, y, train, test, args) for k in chain}

    # stage 13 routing: a patient reaches the source model only if every upstream model
    # left it below that model's own gate
    incoming = np.ones(len(test), dtype=bool)
    incoming_all = np.ones(len(rid), dtype=bool)
    for u, g in zip(args.upstream, upstream_gates):
        incoming &= fits[u]["confidence"] < g
        incoming_all &= fits[u]["confidence_all"] < g

    kept = incoming & (fits[source]["confidence"] >= args.gate)
    if not kept.any():
        print(f"\n  none of the {int(incoming.sum())} patients reaching {name(source)} "
              f"clear {args.gate:.2f} (max {fits[source]['confidence'][incoming].max():.3f} "
              f"on that pool) - nothing was kept, so there is nothing to audit.")
        return

    patients = kept_patients(fits, rid[test], gold, kept, source, target, args)
    summary = summary_row(patients, int(incoming.sum()), source, target, args)

    sheets = {
        "summary": pd.DataFrame([summary]),
        "kept_patients": patients,
        "correctness_move": correctness_move(patients, source, target),
        "target_band": target_band_breakdown(patients, source, target, args),
        "source_confidence_bins": confidence_bins(patients, source, target),
        "policy_comparison": policy_comparison(patients, source, target, args),
        "gate_sweep": gate_sweep(fits, source, target, args, incoming),
        "split_check": split_check(fits, np.isin(np.arange(len(rid)), train),
                                   source, target, args, incoming_all),
        "confidently_wrong": patients[patients[f"{source}_correct"] == 0],
    }

    print("\n" + "-" * 92)
    print(f"THE KEPT SET  -  {name(source)} confidence >= {args.gate:.2f}")
    print("-" * 92)
    print(f"  reaching {name(source):<16}: {summary['incoming_n']} of {len(test)} "
          f"({summary['upstream']})")
    print(f"  kept by {name(source):<17}: {summary['kept_by_source']} "
          f"({summary['kept_pct']}%)   handed off by stage 13: "
          f"{summary['handed_off_by_stage13']}")
    print(f"  converter rate           : {summary['converter_rate_kept']}")
    print(f"  {name(source)} accuracy on them   : {summary['source_acc_on_kept']:.4f}   "
          f"({summary['source_confidently_wrong']} confidently wrong)")
    print(f"  {name(target)} accuracy on them    : {summary['target_acc_on_kept']:.4f}")

    print("\n  RESCUED OR BROKEN?")
    print(f"    RESCUED (wrong -> right): {summary['rescued_wrong_to_right']}   "
          f"({name(target)} confident about "
          f"{summary['rescued_by_confident_target']} of them)")
    print(f"    BROKEN  (right -> wrong): {summary['broken_right_to_wrong']}   "
          f"({name(target)} confident about "
          f"{summary['broken_by_confident_target']} of them)")
    print(f"    unchanged               : {summary['unchanged']}   "
          f"net {summary['net_correct_change']:+d} patients")
    print(f"    the two models disagree on {summary['models_disagree']} of "
          f"{summary['kept_by_source']}")
    print(f"    {name(target)} was confident about {summary['target_confident']} of them "
          f"and wrong on {summary['target_confidently_wrong']} of those")

    print("\n  correctness transition:")
    print(sheets["correctness_move"].to_string(index=False))
    print(f"\n  by {name(target)} band:")
    print(sheets["target_band"].to_string(index=False))
    print(f"\n  by {name(source)} confidence:")
    print(sheets["source_confidence_bins"].to_string(index=False))
    print("\n  what each policy would have scored on these patients:")
    print(sheets["policy_comparison"].to_string(index=False))
    print(f"\n  as the {name(source)} gate moves (the kept set grows as the gate falls):")
    print(sheets["gate_sweep"].to_string(index=False))
    print("\n  the same audit on the other splits:")
    print(sheets["split_check"].to_string(index=False))

    sheets["run_config"] = pd.DataFrame({
        "field": ["question", "cohort", "split", "split_seed", "train_n", "test_n",
                  "source", "target", "upstream", "incoming_n",
                  "source_gate", "target_band",
                  "lambda_selection", "preprocessing", "coverage_floor",
                  f"lambda_{source}", f"lambda_{target}",
                  f"max_confidence_{source}", f"max_confidence_{target}",
                  "note_relation_to_stage13", "note_refit", "note_reading"],
        "value": [
            f"of the patients {name(source)} was confident enough to keep, how many "
            f"would {name(target)} rescue and how many would it break?",
            f"{len(base)} patients with MMSE + MRI + CSF at t=0",
            f"stratified {100 * (1 - args.test_size):.0f}/{100 * args.test_size:.0f} "
            f"on the conversion label",
            args.seed, len(train), len(test),
            name(source), name(target), summary["upstream"], summary["incoming_n"],
            args.gate, args.hi,
            f"{args.folds}-fold CV inside the train half only (seed {args.cv_seed})",
            "median imputation + standardisation fitted on the train half only",
            args.min_coverage,
            fits[source]["lam"], fits[target]["lam"],
            round(float(fits[source]["confidence"].max()), 4),
            round(float(fits[target]["confidence"].max()), 4),
            (f"the pool is the {summary['incoming_n']} patients "
             + (f"{summary['upstream']} handed on"
                if args.upstream else "of the whole test half")
             + f", and this audits the {summary['kept_by_source']} of them "
               f"{name(source)} kept at {args.gate:.2f} - the exact complement of the "
               f"set stage 13 routes onward. Same split, same seeds, same lambda grid, "
               f"so the audited and routed sets partition that pool."),
            f"{name(target)} is not re-fitted for this audit - it is the same model "
            "stage 13 fits on the train half, applied to test rows it never saw.",
            "rescued = the source model was confidently WRONG and the target model gets "
            "it right; broken = the source model was right and the target model would "
            "have overturned a good call. Net is rescued minus broken: negative means "
            "routing the confident patients as well would cost accuracy.",
        ],
    })

    # the chain-conditioned audit is a different population from the standalone one, so it
    # gets its own filenames rather than overwriting them
    tag = f"{source}_to_{target}" + (f"_after_{'_'.join(args.upstream)}"
                                     if args.upstream else "")
    patients_csv = write_csv(patients, confident_audit_path(tag, "patients.csv"))
    summary_csv = write_csv(pd.DataFrame([summary]),
                            confident_audit_path(tag, "summary.csv"))
    png = confident_audit_path(tag, "audit.png")
    plot_audit(patients, source, target, args, png)
    xlsx = write_excel(sheets, confident_audit_path(tag, "audit.xlsx"))

    print(f"\n  written to {CONFIDENT_AUDIT_DIR}:")
    for path in [patients_csv, summary_csv, xlsx, png]:
        print(f"    {Path(path).name}")


if __name__ == "__main__":
    main()
