"""Baseline logistic regression, XGBoost, and an isotonic calibration layer.

Target handling
---------------
One source row = `session_seen` Bernoulli trials with `session_correct`
successes. We expand each row into at most two weighted rows:

    (y=1, weight=session_correct)  and  (y=0, weight=session_seen-session_correct)

This is an exact representation of the trials -- no thresholding, no information
discarded -- and keeps the model's output on the per-review probability scale a
scheduler consumes.

Split discipline
----------------
    train/fit  -> fits the model
    train/val  -> early stopping only
    calib      -> fits isotonic regression only
    test       -> reported numbers only

All four are carved by user_id, so no learner crosses a boundary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluate.metrics import evaluate, format_table, reliability_curve
from src.features.build_features import BASELINE_COLUMNS, FEATURE_COLUMNS
from src.spark_session import PROJECT_ROOT

FIGDIR = PROJECT_ROOT / "reports" / "figures"
REPORTS = PROJECT_ROOT / "reports"


def expand_trials(df: pd.DataFrame, cols: list[str]):
    """Row -> up to two weighted rows. Exact Bernoulli-trial representation."""
    correct = df["session_correct"].to_numpy()
    wrong = (df["session_seen"] - df["session_correct"]).to_numpy()
    X = df[cols].to_numpy(dtype=np.float32)

    pos, neg = correct > 0, wrong > 0
    Xe = np.vstack([X[pos], X[neg]])
    ye = np.concatenate([np.ones(pos.sum()), np.zeros(neg.sum())])
    we = np.concatenate([correct[pos], wrong[neg]]).astype(np.float64)
    return Xe, ye, we


def main(path: Path) -> None:
    # Read only what we use. At full scale the id columns are the memory hazard:
    # 12.85M rows of 32-char lexeme_id as python objects is several GB, so
    # lexeme_id is left on disk entirely and user_id is loaded as a category.
    cols = ["split", "fit_role", "user_id", "session_seen", "session_correct"] + FEATURE_COLUMNS
    df = pd.read_parquet(path, columns=cols)
    df["user_id"] = df["user_id"].astype("category")
    print(f"loaded {len(df):,} rows from {path.name} "
          f"({df.memory_usage(deep=True).sum()/1e9:.2f} GB in memory)")
    parts = {k: df[df["split"] == k] for k in ("train", "calib", "test")}
    fit_df = parts["train"][parts["train"]["fit_role"] == "fit"]
    val_df = parts["train"][parts["train"]["fit_role"] == "val"]

    print("\nSPLIT SIZES (rows / users / trials)")
    for name, d in [("train/fit", fit_df), ("train/val", val_df),
                    ("calib", parts["calib"]), ("test", parts["test"])]:
        print(f"  {name:10} {len(d):>8,} / {d['user_id'].nunique():>5,} / {int(d['session_seen'].sum()):>9,}")

    results, curves = [], {}

    # ---- baseline: LR on recency + frequency only -------------------------
    # Deliberately excludes the engineered item/learner features so the XGBoost
    # gap is attributable to them rather than to the model class.
    Xb, yb, wb = expand_trials(fit_df, BASELINE_COLUMNS)
    Xbt, ybt, wbt = expand_trials(parts["test"], BASELINE_COLUMNS)
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    base.fit(Xb, yb, logisticregression__sample_weight=wb)
    pb = base.predict_proba(Xbt)[:, 1]
    results.append(evaluate(ybt, pb, wbt, "LR baseline (recency+freq)"))
    curves["LR baseline"] = reliability_curve(ybt, pb, wbt)
    del Xb, yb, wb, Xbt, ybt, wbt  # free before building the 18-feature matrices

    # ---- XGBoost on all 18 features ---------------------------------------
    Xf, yf, wf = expand_trials(fit_df, FEATURE_COLUMNS)
    Xv, yv, wv = expand_trials(val_df, FEATURE_COLUMNS)
    Xc, yc, wc = expand_trials(parts["calib"], FEATURE_COLUMNS)
    Xt, yt, wt = expand_trials(parts["test"], FEATURE_COLUMNS)

    xgb = XGBClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        reg_lambda=1.0, tree_method="hist", n_jobs=28,
        eval_metric="logloss", early_stopping_rounds=50, random_state=0,
    )
    xgb.fit(Xf, yf, sample_weight=wf, eval_set=[(Xv, yv)],
            sample_weight_eval_set=[wv], verbose=False)
    print(f"\nXGBoost stopped at {xgb.best_iteration} trees (of 3000)")

    pt = xgb.predict_proba(Xt)[:, 1]
    results.append(evaluate(yt, pt, wt, "XGBoost (18 feats, raw)"))
    curves["XGBoost raw"] = reliability_curve(yt, pt, wt)

    # ---- isotonic calibration layer ---------------------------------------
    # Fit on `calib` -- data neither the model nor early stopping ever saw.
    pc = xgb.predict_proba(Xc)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(pc, yc, sample_weight=wc)
    pt_cal = iso.predict(pt)
    results.append(evaluate(yt, pt_cal, wt, "XGBoost + isotonic"))
    curves["XGBoost calibrated"] = reliability_curve(yt, pt_cal, wt)

    print("\n" + format_table(results))

    # feature importance
    imp = sorted(zip(FEATURE_COLUMNS, xgb.feature_importances_), key=lambda x: -x[1])
    print("\nTOP FEATURES (gain)")
    for n, v in imp[:8]:
        print(f"  {n:24} {v:.4f}")

    _plot(curves, results)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {REPORTS/'metrics.json'} and {FIGDIR/'calibration.png'}")


def _plot(curves: dict, results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.plot([0.4, 1], [0.4, 1], "k--", lw=1, label="perfect calibration")
    for name, (pred, obs, _) in curves.items():
        ax.plot(pred, obs, "o-", ms=4, label=name)
    ax.set_xlabel("predicted P(recall)"); ax.set_ylabel("observed recall rate")
    ax.set_title("Reliability curve (held-out learners)")
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_xlim(0.4, 1); ax.set_ylim(0.4, 1)

    ax = axes[1]
    names = [r["model"] for r in results]
    ax.barh(names, [r["ece"] for r in results], color=["#888", "#d95f02", "#1b9e77"])
    ax.set_xlabel("Expected Calibration Error (lower is better)")
    ax.set_title("Calibration error"); ax.grid(alpha=.3, axis="x")
    for i, r in enumerate(results):
        ax.text(r["ece"], i, f"  {r['ece']:.4f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGDIR / "calibration.png", dpi=140)


if __name__ == "__main__":
    p = PROJECT_ROOT / "data/processed" / (sys.argv[1] if len(sys.argv) > 1
                                           else "features_user_sample.parquet")
    main(p)
