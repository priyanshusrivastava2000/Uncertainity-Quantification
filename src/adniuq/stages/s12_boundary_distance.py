from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import modality_of
from ..io import write_csv, write_excel
from ..modeling import out_of_fold_predictions
from ..paths import (
    BOUNDARY_COMPARISON_PNG,
    BOUNDARY_DIR,
    BOUNDARY_DISTANCES_CSV,
    BOUNDARY_DISTANCES_XLSX,
    INDIVIDUAL_RESULTS_CSV,
    ensure_output_dirs,
    require,
)
from .s04_individual_models import MODELS as S04_MODELS
from .s04_individual_models import assemble

INK, MUTED, SURFACE = "#1a1a1a", "#6b6b6b", "#ffffff"
CONVERTER, STABLE = "#c0392b", "#2c6fbb"

# the two fusion models fitted by stage 04
MODELS = {key: S04_MODELS[key] for key in ("mmse_mri", "mmse_mri_csf")}


def boundary_png(key: str) -> Path:
    return BOUNDARY_DIR / f"{key}_boundary_distance.png"


def logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def score_model(key: str, args) -> pd.DataFrame:
    spec = MODELS[key]
    require(INDIVIDUAL_RESULTS_CSV)

    results = pd.read_csv(INDIVIDUAL_RESULTS_CSV).set_index("model")
    if spec["display"] not in results.index:
        raise SystemExit(
            f"[error] no result row for {spec['display']} - run s04_individual_models first"
        )
    lam = float(results.at[spec["display"], "lam"])

    df, X, feats, _, _, _ = assemble(key, args.min_coverage)
    counts = pd.Series([modality_of(c) for c in feats]).value_counts()
    y = np.where(df["y"].to_numpy() == 1, 1.0, -1.0)
    gold = (y > 0).astype(int)
    train = np.flatnonzero((df["split"] == "train").to_numpy())
    test = np.flatnonzero((df["split"] == "test").to_numpy())

    fitted = out_of_fold_predictions(X, y, train, test, lam, args.folds, args.seed)
    proba, scores = fitted["proba"], fitted["scores"]
    beta_norm = fitted["beta_norm"]

    assert np.allclose(1.0 / (1.0 + np.exp(-scores)), proba, atol=1e-9), \
        "decision scores and probabilities disagree - they must be the same object"

    distance = np.where(beta_norm > 0, scores / np.where(beta_norm > 0, beta_norm, 1.0),
                        np.nan)
    pred = (scores >= 0).astype(int)
    confidence = np.maximum(proba, 1.0 - proba)

    print(f"\n  {spec['display']}: {len(feats)} features  ("
          + " + ".join(f"{counts.get(m, 0)} {m}" for m in ("MMSE", "MRI", "CSF")
                       if counts.get(m, 0))
          + f")   lam={lam:g}   ||beta|| = "
          + " / ".join(f"{v:.3f}" for v in sorted(set(np.round(beta_norm, 3)))))

    return pd.DataFrame({
        "RID": df["RID"],
        "split": df["split"],
        "conv_label": df["conv_label"],
        "y": df["y"],
        "model": spec["display"],
        "model_key": key,
        "lam": lam,
        "score": np.round(scores, 6),
        "abs_score": np.round(np.abs(scores), 6),
        "norm_distance": np.round(distance, 6),
        "beta_norm": np.round(beta_norm, 6),
        "p_converter": np.round(proba, 6),
        "confidence": np.round(confidence, 6),
        "band": np.where(confidence >= args.hi, "high", "low"),
        "predicted": np.where(pred == 1, "converter", "stable"),
        "correct": (pred == gold).astype(int),
        "confidence_source": fitted["source"],
    })


def summarise(scored: pd.DataFrame, args) -> pd.DataFrame:
    hi_cut, lo_cut = logit(args.hi), logit(args.mid)
    rows = []
    for model in scored["model"].unique():
        block = scored[scored["model"] == model]
        for label in ("converter", "stable", "all"):
            cell = block if label == "all" else block[block["conv_label"] == label]
            rows.append({
                "model": model,
                "class": label,
                "n": len(cell),
                "median_abs_score": round(float(cell["abs_score"].median()), 4),
                "mean_abs_score": round(float(cell["abs_score"].mean()), 4),
                "median_abs_distance": round(float(cell["norm_distance"].abs().median()), 4),
                "pct_beyond_p70": round(100.0 * float((cell["abs_score"] >= lo_cut).mean()), 1),
                "pct_beyond_p80": round(100.0 * float((cell["abs_score"] >= hi_cut).mean()), 1),
                "pct_within_0.5": round(100.0 * float((cell["abs_score"] < 0.5).mean()), 1),
                "accuracy": round(float(cell["correct"].mean()), 4),
            })
    return pd.DataFrame(rows)


def pair_frame(scored: pd.DataFrame) -> pd.DataFrame:
    keys = list(MODELS)
    left = scored[scored["model_key"] == keys[0]].reset_index(drop=True)
    right = scored[scored["model_key"] == keys[1]].reset_index(drop=True)
    assert left["RID"].tolist() == right["RID"].tolist(), \
        "the two fusion datasets are not row-aligned"

    suffix = {keys[0]: "mmse_mri", keys[1]: "mmse_mri_csf"}
    paired = pd.DataFrame({
        "RID": left["RID"],
        "split": left["split"],
        "conv_label": left["conv_label"],
        "y": left["y"],
    })
    for frame, key in ((left, keys[0]), (right, keys[1])):
        tag = suffix[key]
        for column in ("score", "abs_score", "norm_distance", "beta_norm", "p_converter",
                       "confidence", "band", "predicted", "correct", "confidence_source"):
            paired[f"{tag}_{column}"] = frame[column]

    paired["score_delta"] = (paired["mmse_mri_csf_score"] - paired["mmse_mri_score"]).round(6)
    paired["norm_distance_delta"] = (
        paired["mmse_mri_csf_norm_distance"] - paired["mmse_mri_norm_distance"]
    ).round(6)
    paired["crossed_boundary"] = (
        np.sign(paired["mmse_mri_score"]) != np.sign(paired["mmse_mri_csf_score"])
    ).astype(int)
    return paired


def plot_model(block: pd.DataFrame, display: str, args, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hi_cut, lo_cut = logit(args.hi), logit(args.mid)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    fig.patch.set_facecolor(SURFACE)

    panels = [
        ("score", "signed log-odds   s = β·x + b", True),
        ("norm_distance", "geometric distance   s / ||β||", False),
    ]
    for ax, (column, xlabel, with_bands) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        values = block[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        bins = np.linspace(finite.min(), finite.max(), 36)

        for label, colour in (("stable", STABLE), ("converter", CONVERTER)):
            cell = block.loc[block["conv_label"] == label, column].to_numpy(dtype=float)
            ax.hist(cell[np.isfinite(cell)], bins=bins, alpha=0.55, color=colour,
                    label=label, edgecolor=SURFACE, linewidth=0.6)

        ax.axvline(0.0, color=INK, linewidth=1.6)
        if with_bands:
            top = ax.get_ylim()[1]
            for (cut, p), height in (((lo_cut, args.mid), 0.99), ((hi_cut, args.hi), 0.90)):
                for sign in (-1, 1):
                    ax.axvline(sign * cut, color=MUTED, linewidth=1.0, linestyle="--")
                ax.text(cut, top * height, f" p={p:.2f}", fontsize=8.5, color=MUTED,
                        ha="left", va="top")

        ax.set_xlabel(xlabel, fontsize=10.5, color=INK)
        ax.set_ylabel("patients", fontsize=10.5, color=INK)
        ax.tick_params(colors=MUTED, length=0)
        ax.grid(axis="y", color="#ececec", linewidth=1)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")
    beyond = 100.0 * float((block["abs_score"] >= hi_cut).mean())
    fig.text(0.5, 0.965, f"Distance to the decision boundary  -  {display}", ha="center",
             fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.915,
             f"{len(block)} patients   -   {beyond:.1f}% sit beyond the "
             f"p={args.hi:.2f} confidence band   -   solid line is the boundary",
             ha="center", fontsize=9.5, color=MUTED)

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_comparison(paired: pd.DataFrame, args, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0))
    fig.patch.set_facecolor(SURFACE)

    panels = [
        ("score", "signed log-odds"),
        ("norm_distance", "geometric distance  s / ||β||"),
    ]
    for ax, (column, label) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        x = paired[f"mmse_mri_{column}"].to_numpy(dtype=float)
        y = paired[f"mmse_mri_csf_{column}"].to_numpy(dtype=float)

        for name, colour in (("stable", STABLE), ("converter", CONVERTER)):
            mask = (paired["conv_label"] == name).to_numpy()
            ax.plot(x[mask], y[mask], "o", markersize=4.5, alpha=0.55, color=colour,
                    markeredgewidth=0, label=name)

        finite = np.isfinite(x) & np.isfinite(y)
        lim = max(np.abs(x[finite]).max(), np.abs(y[finite]).max()) * 1.08
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=1.0, linestyle=":")
        ax.axvline(0.0, color=INK, linewidth=1.2)
        ax.axhline(0.0, color=INK, linewidth=1.2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"MMSE + MRI   {label}", fontsize=10.5, color=INK)
        ax.set_ylabel(f"MMSE + MRI + CSF   {label}", fontsize=10.5, color=INK)
        ax.tick_params(colors=MUTED, length=0)
        ax.grid(color="#ececec", linewidth=1)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")
    crossed = int(paired["crossed_boundary"].sum())
    fig.text(0.5, 0.965, "Where each patient sits relative to the boundary, before and "
                         "after adding CSF", ha="center", fontsize=13.5,
             fontweight="bold", color=INK)
    fig.text(0.5, 0.915,
             f"{len(paired)} patients   -   {crossed} changed which side of the boundary "
             f"they fall on   -   dotted line is y = x",
             ha="center", fontsize=9.5, color=MUTED)

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s12_boundary_distance",
        description="Plot how far every patient sits from each fusion model's boundary.",
    )
    parser.add_argument("--hi", type=float, default=0.80,
                        help="confidence band drawn on the log-odds panel")
    parser.add_argument("--mid", type=float, default=0.70,
                        help="second reference band")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    ensure_output_dirs()

    print("=" * 92)
    print("DISTANCE TO THE DECISION BOUNDARY  -  MMSE+MRI  vs  MMSE+MRI+CSF")
    print("=" * 92)
    print("  boundary  : s = 0   (p = 0.50)")
    print(f"  bands     : |s| = {logit(args.mid):.4f} is p={args.mid:.2f}, "
          f"|s| = {logit(args.hi):.4f} is p={args.hi:.2f}")

    scored = pd.concat([score_model(key, args) for key in MODELS], ignore_index=True)
    summary = summarise(scored, args)
    paired = pair_frame(scored)

    for key, spec in MODELS.items():
        block = scored[scored["model_key"] == key]
        plot_model(block, spec["display"], args, boundary_png(key))
    plot_comparison(paired, args, BOUNDARY_COMPARISON_PNG)

    crossings = paired[paired["crossed_boundary"] == 1][[
        "RID", "split", "conv_label", "mmse_mri_score", "mmse_mri_predicted",
        "mmse_mri_correct", "mmse_mri_csf_score", "mmse_mri_csf_predicted",
        "mmse_mri_csf_correct", "score_delta",
    ]].sort_values("score_delta")

    config = pd.DataFrame({
        "field": ["patients", "models", "lambda_mmse_mri", "lambda_mmse_mri_csf",
                  "coverage_floor", "folds", "seed", "band_threshold", "boundary",
                  "band_in_log_odds", "train_scores", "test_scores", "caveat_beta_norm"],
        "value": [
            len(paired), " / ".join(m["display"] for m in MODELS.values()),
            float(scored[scored.model_key == "mmse_mri"]["lam"].iloc[0]),
            float(scored[scored.model_key == "mmse_mri_csf"]["lam"].iloc[0]),
            args.min_coverage, args.folds, args.seed, args.hi, "s = 0 is p = 0.50",
            f"|s| = {logit(args.hi):.4f}",
            f"{args.folds}-fold out-of-fold", "model fitted on all training rows",
            "out-of-fold scores come from five fold-models, each with its own beta and "
            "scaler, so ||beta|| is per-fold; the geometric distance is comparable within "
            "a model but the log-odds is the quantity that drives the reported bands",
        ],
    })

    write_csv(paired, BOUNDARY_DISTANCES_CSV)
    xlsx = write_excel({
        "distances": paired,
        "per_model_long": scored,
        "summary": summary,
        "crossings": crossings,
        "run_config": config,
    }, BOUNDARY_DISTANCES_XLSX)

    print("\n" + "=" * 92)
    print("DISTANCE SUMMARY")
    print("=" * 92)
    print(summary.to_string(index=False))

    print(f"\n  patients that changed side of the boundary when CSF was added: "
          f"{int(paired['crossed_boundary'].sum())} of {len(paired)}")
    both = paired[paired["crossed_boundary"] == 1]
    if len(both):
        gained = int((both["mmse_mri_csf_correct"] > both["mmse_mri_correct"]).sum())
        lost = int((both["mmse_mri_csf_correct"] < both["mmse_mri_correct"]).sum())
        print(f"    of those, CSF fixed {gained} and broke {lost}")

    print("\n  written:")
    for key in MODELS:
        print(f"    {boundary_png(key).name}")
    print(f"    {BOUNDARY_COMPARISON_PNG}")
    print(f"    {BOUNDARY_DISTANCES_CSV.name}\n    {xlsx.name}")


if __name__ == "__main__":
    main()
