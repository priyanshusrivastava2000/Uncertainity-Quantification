from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..features import DISPLAY, all_features, covered
from ..modeling import fit_clg, preprocess
from ..paths import (
    INDIVIDUAL_RESULTS_CSV,
    conversion_csv,
    ensure_output_dirs,
    feature_selection_xlsx,
    require,
)
from ..selection import write_feature_selection


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s04_feature_selection",
        description="Export what each modality model kept and what the lasso zeroed.",
    )
    parser.add_argument("--min-coverage", type=float, default=0.5)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    require(INDIVIDUAL_RESULTS_CSV)
    results = pd.read_csv(INDIVIDUAL_RESULTS_CSV).set_index("modality")

    print("=" * 90)
    print("FEATURE SELECTION EXPORT  -  what each model kept, what it zeroed")
    print("=" * 90)

    overview = []
    for modality in ("mmse", "mri", "csf"):
        display = DISPLAY[modality]
        if display not in results.index:
            print(f"  [skip] no result row for {display}")
            continue
        lam = float(results.at[display, "lam"])

        require(conversion_csv(modality))
        df = pd.read_csv(conversion_csv(modality), low_memory=False)
        y = np.where(df["y"].to_numpy() == 1, 1.0, -1.0)
        train = np.flatnonzero((df["split"] == "train").to_numpy())
        test = np.flatnonzero((df["split"] == "test").to_numpy())

        feats, coverage, _ = covered(df, all_features(modality, df), args.min_coverage)
        X = df[feats].to_numpy(dtype=float)
        X_train, _ = preprocess(X[train], X[test])
        clf = fit_clg(X_train, y[train], lam)

        assert int(clf.n_nonzero) == int(results.at[display, "n_nonzero"]), (
            f"{display}: refit gave {clf.n_nonzero} non-zero, results table says "
            f"{int(results.at[display, 'n_nonzero'])} - the fit is not reproducing"
        )

        path, _, by_group, _ = write_feature_selection(
            clf, feats, coverage, lam, float(results.at[display, "roc_auc"]),
            feature_selection_xlsx(modality), f"{display} only", args.min_coverage,
        )

        n_nonzero = int(clf.n_nonzero)
        print(f"\n{display}  lam={lam:g}   {len(feats)} features -> {n_nonzero} kept, "
              f"{len(feats) - n_nonzero} zeroed "
              f"({100 * (1 - n_nonzero / len(feats)):.1f}% sparse)")
        print(by_group.to_string(index=False))
        print(f"  -> {path.name}")

        overview.append({
            "modality": display,
            "lam": lam,
            "n_features": len(feats),
            "non_zero": n_nonzero,
            "zero": len(feats) - n_nonzero,
            "sparsity_pct": round(100 * (1 - n_nonzero / len(feats)), 1),
        })

    print("\n" + "=" * 90)
    print(pd.DataFrame(overview).to_string(index=False))


if __name__ == "__main__":
    main()
