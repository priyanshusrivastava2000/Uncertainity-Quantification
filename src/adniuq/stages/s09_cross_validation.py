from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from ..features import all_features, covered
from ..io import write_csv, write_excel
from ..modeling import compute_metrics, fit_clg, predict_proba, preprocess, select_lam
from ..paths import (
    CV_AUC_PNG,
    CV_REPEATS_CSV,
    CV_SUMMARY_CSV,
    CV_XLSX,
    conversion_csv,
    ensure_output_dirs,
    require,
)

NARROW_GRID = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
WIDE_GRID = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0]

# A model is one or more modalities. Fusion feature blocks are concatenated here from the
# per-modality conversion tables, which are row-aligned on RID and label, so no separate
# fusion dataset stage is needed.
MODELS = {
    "MMSE": {"modalities": ("mmse",), "grid": NARROW_GRID},
    "MRI": {"modalities": ("mri",), "grid": WIDE_GRID},
    "CSF": {"modalities": ("csf",), "grid": NARROW_GRID},
    "MMSE+MRI": {"modalities": ("mmse", "mri"), "grid": WIDE_GRID},
    "MMSE+MRI+CSF": {"modalities": ("mmse", "mri", "csf"), "grid": WIDE_GRID},
}

REPORTED = [
    "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity",
    "specificity", "f1_converter", "acc_high_conf", "frac_high_conf",
    "confidently_wrong", "n_nonzero", "lam",
]


def load_model_data(name: str, min_coverage: float):
    """Feature block for one model, concatenated across its modalities.

    The per-modality conversion tables share a row order (same RIDs, same labels, same
    canonical split), so the blocks can be stacked horizontally without a join.
    """
    modalities = MODELS[name]["modalities"]
    require(*[conversion_csv(m) for m in modalities])

    frames = {m: pd.read_csv(conversion_csv(m), low_memory=False) for m in modalities}
    first = frames[modalities[0]]
    for m in modalities[1:]:
        assert frames[m]["RID"].tolist() == first["RID"].tolist(), (
            f"conversion tables are not aligned on RID: {modalities[0]} vs {m}"
        )
        assert frames[m]["y"].tolist() == first["y"].tolist(), (
            f"labels disagree between conversion tables: {modalities[0]} vs {m}"
        )

    feats, blocks = [], []
    for m in modalities:
        df = frames[m]
        kept, _, _ = covered(df, all_features(m, df), min_coverage)
        overlap = set(kept) & set(feats)
        assert not overlap, f"feature name collision across modalities: {overlap}"
        feats += kept
        blocks.append(df[kept].to_numpy(dtype=float))

    X = np.hstack(blocks)
    y = np.where(first["y"].to_numpy() == 1, 1.0, -1.0)
    return X, y, feats


def splitter(scheme: str, repeats: int, folds: int, test_size: float, seed: int):
    if scheme == "shuffle":
        return StratifiedShuffleSplit(
            n_splits=repeats, test_size=test_size, random_state=seed
        )
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def evaluate_model(name: str, args) -> list:
    X, y, feats = load_model_data(name, args.min_coverage)
    gold_all = (y > 0).astype(int)
    grid = MODELS[name]["grid"]

    rows = []
    started = time.time()
    print(f"\n{'-' * 92}")
    print(f"[{name}]  {X.shape[0]} patients x {len(feats)} features   "
          f"{args.scheme} resampling")

    for i, (train, test) in enumerate(
        splitter(args.scheme, args.repeats, args.folds, args.test_size,
                 args.seed).split(X, gold_all), start=1
    ):
        lam, _ = select_lam(X[train], y[train], grid, k=args.inner_folds,
                            seed=args.seed + i, verbose=False)
        X_train, X_test = preprocess(X[train], X[test])
        clf = fit_clg(X_train, y[train], lam)
        proba = predict_proba(clf, X_test)
        gold = gold_all[test]

        metrics = compute_metrics(proba, gold, args.hi)
        rows.append({
            "model": name,
            "split": i,
            "train_n": len(train),
            "test_n": len(test),
            "n_features": len(feats),
            "lam": lam,
            "n_nonzero": int(clf.n_nonzero),
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in metrics.items()},
        })
        print(f"    split {i:>2}/{args.repeats if args.scheme == 'shuffle' else args.folds}"
              f"   lam={lam:<5g} nz={clf.n_nonzero:<4} AUC={metrics['roc_auc']:.4f}   "
              f"acc={metrics['accuracy']:.4f}")

    print(f"  {name} done in {time.time() - started:.0f}s")
    return rows


def summarise(repeats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in repeats["model"].unique():
        block = repeats[repeats["model"] == name]
        row = {
            "model": name,
            "n_splits": len(block),
            "n_features": int(block["n_features"].iloc[0]),
            "train_n": int(block["train_n"].iloc[0]),
            "test_n": int(block["test_n"].iloc[0]),
        }
        for metric in REPORTED:
            values = block[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = round(float(values.mean()), 4)
            row[f"{metric}_sd"] = round(float(values.std(ddof=1)), 4)
            row[f"{metric}_min"] = round(float(values.min()), 4)
            row[f"{metric}_max"] = round(float(values.max()), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def lambda_stability(repeats: pd.DataFrame) -> pd.DataFrame:
    counts = (
        repeats.groupby(["model", "lam"]).size().rename("n_splits").reset_index()
    )
    totals = counts.groupby("model")["n_splits"].transform("sum")
    counts["pct_of_splits"] = (100 * counts["n_splits"] / totals).round(1)
    return counts.sort_values(["model", "n_splits"], ascending=[True, False])


def plot_distribution(repeats: pd.DataFrame, order: list, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, surface, accent = "#1a1a1a", "#6b6b6b", "#ffffff", "#2c6fbb"
    data = [repeats.loc[repeats["model"] == name, "roc_auc"].to_numpy() for name in order]

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    box = ax.boxplot(data, vert=True, widths=0.55, patch_artist=True,
                     medianprops={"color": ink, "linewidth": 1.6},
                     whiskerprops={"color": muted}, capprops={"color": muted},
                     flierprops={"markeredgecolor": muted, "markersize": 4})
    for patch in box["boxes"]:
        patch.set_facecolor("#d6e4f2")
        patch.set_edgecolor(muted)

    rng = np.random.default_rng(0)
    for i, values in enumerate(data, start=1):
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.plot(np.full(len(values), i) + jitter, values, "o", markersize=4.5,
                color=accent, alpha=0.55, markeredgewidth=0)

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(
        [f"{name}\n{values.mean():.3f} ± {values.std(ddof=1):.3f}"
         for name, values in zip(order, data)],
        fontsize=9.5, color=ink,
    )
    ax.set_ylabel("ROC-AUC on the held-out split", fontsize=10.5, color=ink)
    ax.tick_params(colors=muted, length=0)
    ax.grid(axis="y", color="#ececec", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    n_splits = len(data[0]) if data else 0
    fig.text(0.5, 0.955, "ROC-AUC across resampled train/test splits", ha="center",
             fontsize=13.5, fontweight="bold", color=ink)
    fig.text(0.5, 0.905, f"{n_splits} splits per model, lambda re-selected inside every "
                         f"training set", ha="center", fontsize=10, color=muted)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=150, facecolor=surface)
    plt.close(fig)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s09_cross_validation",
        description="Resample the train/test split and report mean +- sd per model.",
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--scheme", default="shuffle", choices=["shuffle", "kfold"],
                        help="repeated stratified holdout, or stratified k-fold")
    parser.add_argument("--repeats", type=int, default=20,
                        help="number of resampled splits (shuffle scheme)")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--folds", type=int, default=5, help="outer folds (kfold scheme)")
    parser.add_argument("--inner-folds", type=int, default=5,
                        help="folds of the lambda search inside each training set")
    parser.add_argument("--hi", type=float, default=0.80)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    ensure_output_dirs()

    n_splits = args.repeats if args.scheme == "shuffle" else args.folds
    print("=" * 92)
    print("CROSS-VALIDATED COMPARISON  -  every model on the same resampled splits")
    print("=" * 92)
    print(f"  scheme        : {args.scheme}"
          + (f" ({args.repeats} x stratified {100 * (1 - args.test_size):.0f}/"
             f"{100 * args.test_size:.0f} holdout)" if args.scheme == "shuffle"
             else f" (stratified {args.folds}-fold)"))
    print(f"  splits        : {n_splits} per model, identical across models (seed "
          f"{args.seed})")
    print(f"  lambda        : re-selected by {args.inner_folds}-fold CV inside every "
          f"training set")
    print(f"  models        : {', '.join(args.models)}")

    rows = []
    for name in args.models:
        rows += evaluate_model(name, args)

    repeats = pd.DataFrame(rows)
    summary = summarise(repeats)
    stability = lambda_stability(repeats)

    write_csv(repeats, CV_REPEATS_CSV)
    write_csv(summary, CV_SUMMARY_CSV)

    config = pd.DataFrame({
        "field": ["scheme", "n_splits", "test_size", "inner_folds", "seed",
                  "min_coverage", "confidence_band", "spread", "caveat"],
        "value": [
            args.scheme, n_splits, args.test_size, args.inner_folds, args.seed,
            args.min_coverage, args.hi,
            "mean +- sd across splits",
            "training sets overlap between splits, so the sd measures sensitivity to the "
            "split, not an independent-sample standard error; it is narrower than a true "
            "out-of-sample interval",
        ],
    })

    xlsx = write_excel({
        "summary": summary,
        "per_split": repeats,
        "lambda_stability": stability,
        "run_config": config,
    }, CV_XLSX)

    order = summary["model"].tolist()
    plot_distribution(repeats, order, CV_AUC_PNG)

    print("\n" + "=" * 92)
    print(f"RESULTS  -  mean +- sd over {n_splits} resampled splits")
    print("=" * 92)
    print(f"  {'model':<16}{'AUC':>17}{'PR-AUC':>17}{'bal.acc':>17}{'accuracy':>17}")
    for r in summary.itertuples(index=False):
        print(f"  {r.model:<16}"
              f"{f'{r.roc_auc_mean:.3f}+-{r.roc_auc_sd:.3f}':>17}"
              f"{f'{r.pr_auc_mean:.3f}+-{r.pr_auc_sd:.3f}':>17}"
              f"{f'{r.balanced_accuracy_mean:.3f}+-{r.balanced_accuracy_sd:.3f}':>17}"
              f"{f'{r.accuracy_mean:.3f}+-{r.accuracy_sd:.3f}':>17}")

    print(f"\n  {'model':<16}{'AUC range':>18}{'sens':>16}{'spec':>16}{'nonzero':>16}")
    for r in summary.itertuples(index=False):
        print(f"  {r.model:<16}"
              f"{f'[{r.roc_auc_min:.3f},{r.roc_auc_max:.3f}]':>18}"
              f"{f'{r.sensitivity_mean:.3f}+-{r.sensitivity_sd:.3f}':>16}"
              f"{f'{r.specificity_mean:.3f}+-{r.specificity_sd:.3f}':>16}"
              f"{f'{r.n_nonzero_mean:.0f}+-{r.n_nonzero_sd:.0f}':>16}")

    print("\n  lambda selected across splits:")
    for name in order:
        block = stability[stability["model"] == name]
        picks = ", ".join(f"{row.lam:g} ({row.n_splits}x)"
                          for row in block.itertuples(index=False))
        print(f"    {name:<16}{picks}")

    print(f"\n  written:\n    {CV_REPEATS_CSV.name}\n    {CV_SUMMARY_CSV.name}"
          f"\n    {xlsx.name}\n    {CV_AUC_PNG.name}")


if __name__ == "__main__":
    main()
