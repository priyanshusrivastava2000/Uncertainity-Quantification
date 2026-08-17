from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .clg_lasso import CLGLasso
from .io import write_excel


def to_columns(X: np.ndarray):
    n, d = X.shape
    rows = np.arange(n, dtype=np.int64)
    columns = [
        (rows, X[:, j].astype(np.float64), (X[:, j] ** 2).astype(np.float64))
        for j in range(d)
    ]
    return columns, n


def preprocess(X_train: np.ndarray, X_test: np.ndarray):
    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))
    return (
        scaler.transform(imputer.transform(X_train)),
        scaler.transform(imputer.transform(X_test)),
    )


def fit_clg(X_train: np.ndarray, y_train: np.ndarray, lam: float) -> CLGLasso:
    columns, n = to_columns(X_train)
    return CLGLasso(lam=lam).fit(columns, y_train, n)


def predict_proba(clf: CLGLasso, X: np.ndarray) -> np.ndarray:
    columns, n = to_columns(X)
    return clf.predict_proba(columns, n)


def decision_scores(clf: CLGLasso, X: np.ndarray) -> np.ndarray:
    columns, n = to_columns(X)
    return clf.decision_function(columns, n)


def out_of_fold_predictions(X, y, train, test, lam, folds: int = 5, seed: int = 0):
    n = len(y)
    proba = np.full(n, np.nan)
    scores = np.full(n, np.nan)
    beta_norm = np.full(n, np.nan)
    source = np.empty(n, dtype=object)

    gold_train = (y[train] > 0).astype(int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fit_idx, hold_idx in splitter.split(X[train], gold_train):
        X_fit, X_hold = preprocess(X[train][fit_idx], X[train][hold_idx])
        clf = fit_clg(X_fit, y[train][fit_idx], lam)
        held = train[hold_idx]
        proba[held] = predict_proba(clf, X_hold)
        scores[held] = decision_scores(clf, X_hold)
        beta_norm[held] = float(np.linalg.norm(clf.beta_))
    source[train] = "out_of_fold"

    X_train, X_test = preprocess(X[train], X[test])
    clf = fit_clg(X_train, y[train], lam)
    proba[test] = predict_proba(clf, X_test)
    scores[test] = decision_scores(clf, X_test)
    beta_norm[test] = float(np.linalg.norm(clf.beta_))
    source[test] = "fitted_model"

    return {
        "proba": proba,
        "scores": scores,
        "beta_norm": beta_norm,
        "source": source,
        "model": clf,
    }


def select_lam(
    X: np.ndarray,
    y: np.ndarray,
    lams: Sequence[float],
    k: int = 5,
    seed: int = 0,
    verbose: bool = True,
):
    gold = (y > 0).astype(int)
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, gold))
    rows = []

    for lam in lams:
        aucs, nonzeros = [], []
        for train_idx, val_idx in folds:
            X_train, X_val = preprocess(X[train_idx], X[val_idx])
            clf = fit_clg(X_train, y[train_idx], lam)
            proba = predict_proba(clf, X_val)
            g = gold[val_idx]
            aucs.append(roc_auc_score(g, proba) if len(np.unique(g)) > 1 else np.nan)
            nonzeros.append(clf.n_nonzero)
        rows.append({
            "lam": lam,
            "cv_auc": float(np.nanmean(aucs)),
            "cv_auc_sd": float(np.nanstd(aucs)),
            "mean_nonzero": float(np.mean(nonzeros)),
        })

    curve = pd.DataFrame(rows)
    best = float(curve.loc[curve["cv_auc"].idxmax(), "lam"])
    if verbose:
        print(f"    lambda search ({k}-fold CV on train, maximise AUC):")
        for r in curve.itertuples(index=False):
            mark = "  <- selected" if r.lam == best else ""
            print(
                f"      lam={r.lam:<6g} AUC {r.cv_auc:.4f} +-{r.cv_auc_sd:.4f}"
                f"   nz~{r.mean_nonzero:.0f}{mark}"
            )
    curve["selected"] = (curve["lam"] == best).astype(int)
    return best, curve


def compute_metrics(proba: np.ndarray, gold: np.ndarray, hi: float) -> dict:
    pred = (proba >= 0.5).astype(int)
    correct = pred == gold
    confidence = np.maximum(proba, 1.0 - proba)
    high = confidence >= hi

    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    tn = int(((pred == 0) & (gold == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    majority = max(gold.mean(), 1.0 - gold.mean())

    return {
        "n_test": len(gold),
        "accuracy": float(correct.mean()),
        "majority_baseline": float(majority),
        "lift_over_majority": float(correct.mean() - majority),
        "roc_auc": float(roc_auc_score(gold, proba)) if len(np.unique(gold)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(gold, proba)),
        "pr_auc_baseline": float(gold.mean()),
        "balanced_accuracy": float((sens + spec) / 2.0),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1_converter": float(f1),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "acc_high_conf": float(correct[high].mean()) if high.any() else np.nan,
        "acc_low_conf": float(correct[~high].mean()) if (~high).any() else np.nan,
        "frac_high_conf": float(high.mean()),
        "confidently_wrong": int((~correct & high).sum()),
    }


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    stat: str = "mean",
    gold: np.ndarray = None,
):
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    if stat == "mean":
        stats = values[idx].mean(axis=1)
    else:
        stats = np.asarray([
            roc_auc_score(gold[row], values[row])
            for row in idx
            if len(np.unique(gold[row])) > 1
        ])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def evaluate(proba: np.ndarray, gold: np.ndarray, hi: float, n_boot: int, seed: int):
    metrics = compute_metrics(proba, gold, hi)
    correct = ((proba >= 0.5).astype(int) == gold).astype(float)
    acc_lo, acc_hi = bootstrap_ci(correct, n_boot, seed)
    auc_lo, auc_hi = bootstrap_ci(proba, n_boot, seed, stat="auc", gold=gold)
    ci = {
        "acc_ci_lo": acc_lo,
        "acc_ci_hi": acc_hi,
        "auc_ci_lo": auc_lo,
        "auc_ci_hi": auc_hi,
    }
    return metrics, ci


def result_row(metrics: dict, ci: dict) -> dict:
    row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
    row.update({k: round(v, 4) for k, v in ci.items()})
    return row


def confidence_cells(proba: np.ndarray, gold: np.ndarray, hi: float) -> list:
    correct = (proba >= 0.5).astype(int) == gold
    high = np.maximum(proba, 1.0 - proba) >= hi
    return [
        [int((correct & high).sum()), int((correct & ~high).sum())],
        [int((~correct & high).sum()), int((~correct & ~high).sum())],
    ]


def plot_matrix(counts, display_name, hi, n, acc, acc_hi, acc_lo, out_path, auc=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(counts, dtype=float)
    ink, muted, surface = "#1a1a1a", "#6b6b6b", "#ffffff"
    critical = "#c0392b"

    fig = plt.figure(figsize=(7.8, 6.4))
    fig.patch.set_facecolor(surface)
    ax = fig.add_axes([0.155, 0.15, 0.68, 0.60])
    cax = fig.add_axes([0.875, 0.15, 0.028, 0.60])
    ax.set_facecolor(surface)

    vmax = counts.max()
    mesh = ax.pcolormesh(
        counts[::-1], cmap=plt.cm.Blues, vmin=0, vmax=vmax, edgecolors=surface, linewidth=4
    )

    labels = [
        ["confident & right", "unsure but right"],
        ["confidently WRONG", "unsure & wrong"],
    ]
    for row in range(2):
        for col in range(2):
            value = counts[row, col]
            y = 1 - row
            dark = value > 0.5 * vmax
            text_colour = "#ffffff" if dark else ink
            ax.text(col + 0.5, y + 0.60, f"{int(value)}", ha="center", va="center",
                    fontsize=30, fontweight="bold", color=text_colour)
            ax.text(col + 0.5, y + 0.36, f"{100.0 * value / n:.1f}% of test",
                    ha="center", va="center", fontsize=10.5,
                    color=("#eaeaea" if dark else muted))
            ax.text(col + 0.5, y + 0.20, labels[row][col], ha="center", va="center",
                    fontsize=10.5, style="italic",
                    color=(critical if (row == 1 and col == 0)
                           else ("#eaeaea" if dark else muted)))

    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels([f"HIGH conf\n(>= {hi:.2f})", f"LOW conf\n(< {hi:.2f})"],
                       fontsize=11, color=ink)
    ax.set_yticklabels(["INCORRECT", "CORRECT"], fontsize=11, color=ink)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.35, 2)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    title = f"Confidence x correctness  -  {display_name}  (TEST, n={n})"
    title_size = 14 if len(title) <= 58 else (12 if len(title) <= 72 else
                                              (11 if len(title) <= 84 else 10))
    fig.text(0.5, 0.94, title, ha="center", fontsize=title_size, fontweight="bold", color=ink)

    def fmt(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        return f"{value:.3f}"

    parts = ([] if auc is None else [f"AUC {fmt(auc)}"]) + [
        f"accuracy {fmt(acc)}",
        f"acc | HIGH conf {fmt(acc_hi)}",
        f"acc | LOW conf {fmt(acc_lo)}",
    ]
    subtitle = "      -      ".join(parts)
    fig.text(0.5, 0.875, subtitle, ha="center",
             fontsize=(9.5 if len(subtitle) > 78 else 10.5), color=muted)

    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("patient count", fontsize=10, color=muted)
    cbar.ax.tick_params(labelsize=9, color=muted, labelcolor=muted)
    cbar.outline.set_visible(False)

    fig.savefig(out_path, dpi=150, facecolor=surface)
    plt.close(fig)


def report(
    clf,
    proba,
    gold,
    rid,
    feats,
    metrics,
    display_name,
    task_label,
    hi,
    out_dir: Path,
    lam,
    lam_curve,
    pos_label: str = "converter",
    neg_label: str = "stable",
    ci: dict = None,
):
    out_dir = Path(out_dir)
    pred = (proba >= 0.5).astype(int)
    correct = pred == gold
    confidence = np.maximum(proba, 1.0 - proba)
    high = confidence >= hi
    n = len(gold)
    cells = confidence_cells(proba, gold, hi)
    (correct_hi, correct_lo), (wrong_hi, wrong_lo) = cells

    print("\n" + "=" * 64)
    print(f"CONFIDENCE x CORRECTNESS  ({display_name}, TEST n={n})")
    print("=" * 64)
    print(f"  {'':<12}{'HIGH conf':>12}{'LOW conf':>12}")
    print(f"  {'CORRECT':<12}{correct_hi:>12}{correct_lo:>12}")
    print(f"  {'INCORRECT':<12}{wrong_hi:>12}{wrong_lo:>12}")
    print(f"\n  accuracy {metrics['accuracy']:.4f}   "
          f"(majority {metrics['majority_baseline']:.4f}, "
          f"lift {metrics['lift_over_majority']:+.4f})")
    print(f"  ROC-AUC  {metrics['roc_auc']:.4f}    PR-AUC {metrics['pr_auc']:.4f} "
          f"(baseline {metrics['pr_auc_baseline']:.4f})")
    print(f"  balanced acc {metrics['balanced_accuracy']:.4f}   "
          f"sens {metrics['sensitivity']:.4f}   spec {metrics['specificity']:.4f}   "
          f"F1 {metrics['f1_converter']:.4f}")
    print(f"  TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']} TN={metrics['TN']}")
    print(f"  confidently WRONG: {metrics['confidently_wrong']}  "
          f"(on the {100 * metrics['frac_high_conf']:.0f}% it is confident about, "
          f"accuracy is {metrics['acc_high_conf']:.4f})")

    nonzero = [j for j in np.argsort(-np.abs(clf.beta_)) if clf.beta_[j] != 0.0]
    print(f"\n  non-zero features: {len(nonzero)} of {len(feats)}"
          + ("  (top 15 shown)" if len(nonzero) > 15 else ""))
    for j in nonzero[:15]:
        weight = float(clf.beta_[j])
        print(f"    {feats[j]:<20}{weight:+.4f}  -> "
              f"{pos_label if weight > 0 else neg_label}")

    png = out_dir / f"confidence_matrix_{task_label}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_matrix(cells, display_name, hi, n, metrics["accuracy"],
                metrics["acc_high_conf"], metrics["acc_low_conf"], png,
                auc=metrics.get("roc_auc"))

    predictions = pd.DataFrame({
        "RID": rid,
        "gold": np.where(gold == 1, pos_label, neg_label),
        "p_converter": np.round(proba, 6),
        "confidence": np.round(confidence, 6),
        "predicted": np.where(pred == 1, pos_label, neg_label),
        "correct": correct.astype(int),
        "band": np.where(high, "high", "low"),
        "cell": [
            f"{'correct' if c else 'incorrect'}-{'high' if h else 'low'}"
            for c, h in zip(correct, high)
        ],
    }).sort_values("p_converter", ascending=False)

    matrix = pd.DataFrame({
        "outcome": ["CORRECT", "INCORRECT"],
        f"HIGH_conf(>={hi})": [correct_hi, wrong_hi],
        f"LOW_conf(<{hi})": [correct_lo, wrong_lo],
    })

    metric_rows = [
        {"metric": k, "value": (round(v, 4) if isinstance(v, float) else v)}
        for k, v in metrics.items()
    ]
    if ci:
        metric_rows += [{"metric": k, "value": round(v, 4)} for k, v in ci.items()]
    metric_rows += [
        {"metric": "lambda", "value": lam},
        {"metric": "n_features_kept", "value": len(feats)},
        {"metric": "n_nonzero_weights", "value": clf.n_nonzero},
        {"metric": "intercept", "value": round(float(clf.intercept_), 4)},
    ]

    order = np.argsort(-np.abs(clf.beta_))
    weights = pd.DataFrame({
        "rank": np.arange(1, len(feats) + 1),
        "feature": [feats[j] for j in order],
        "weight": [round(float(clf.beta_[j]), 6) for j in order],
        "direction": [
            f"-> {pos_label}" if clf.beta_[j] > 0
            else (f"-> {neg_label}" if clf.beta_[j] < 0 else "(zero)")
            for j in order
        ],
    })

    xlsx = write_excel({
        "test_predictions": predictions,
        "confidence_matrix": matrix,
        "metrics": pd.DataFrame(metric_rows),
        "model_weights": weights,
        "lambda_cv_curve": lam_curve,
    }, out_dir / f"{task_label}_results.xlsx")

    print(f"\n  heatmap -> {png.name}")
    print(f"  excel   -> {xlsx.name}")
    return png, xlsx
