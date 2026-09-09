"""Probability-quality metrics for the recall model.

Everything here takes sample weights, because our target is per-TRIAL: one row
of the source data represents `session_seen` Bernoulli trials, so an unweighted
metric would treat a 1-trial session and a 9-trial session as equally important.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(y, p, w, n_bins: int = 20) -> float:
    """Weighted ECE: mean |confidence - accuracy| across equal-width bins.

    Complements the reliability curve with a single number. Equal-width bins
    (not quantile) so the sparsely-populated low-probability region still shows
    up rather than being merged away.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = w.sum()
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        wb = w[m].sum()
        if wb == 0:
            continue
        ece += (wb / total) * abs(np.average(p[m], weights=w[m]) - np.average(y[m], weights=w[m]))
    return float(ece)


def reliability_curve(y, p, w, n_bins: int = 20):
    """Weighted reliability curve: (mean predicted, observed rate, weight) per bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    pred, obs, wt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any() or w[m].sum() == 0:
            continue
        pred.append(np.average(p[m], weights=w[m]))
        obs.append(np.average(y[m], weights=w[m]))
        wt.append(w[m].sum())
    return np.array(pred), np.array(obs), np.array(wt)


def evaluate(y, p, w, label: str = "") -> dict:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "model": label,
        "auc": roc_auc_score(y, p, sample_weight=w),
        "brier": brier_score_loss(y, p, sample_weight=w),
        "log_loss": log_loss(y, p, sample_weight=w, labels=[0, 1]),
        "ece": expected_calibration_error(y, p, w),
        "trials": float(w.sum()),
    }


def format_table(rows: list[dict]) -> str:
    hdr = f"{'model':<28} {'AUC':>7} {'Brier':>9} {'LogLoss':>9} {'ECE':>8}"
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        out.append(f"{r['model']:<28} {r['auc']:>7.4f} {r['brier']:>9.5f} "
                   f"{r['log_loss']:>9.5f} {r['ece']:>8.5f}")
    return "\n".join(out)
