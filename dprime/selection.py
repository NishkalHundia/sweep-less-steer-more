import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from .geometry import rank_layers
from .judge import SCALES

MIN_FLUENT = 28


def prepare(df, behavior):
    """Score in [0, 1], binary behavior label and fluency flag for every response."""
    df = df.copy()
    if "match" in df:
        df["score"] = df["match"].astype(float)
        df["fluent"] = True
    else:
        lo, hi = SCALES[behavior]
        df["score"] = (df["behavior"] - lo) / (hi - lo)
        df["fluent"] = df["fluency"] >= 1
    df["label"] = df["score"] > 0.5
    return df


def load_run(run_dir, behavior):
    run_dir = Path(run_dir)
    dprimes = {int(k): v for k, v in json.loads((run_dir / "dprime.json").read_text()).items()}
    val, test = (prepare(pd.read_csv(run_dir / f"{s}.csv"), behavior) for s in ("val", "test"))
    return dprimes, val, test


def score_table(df, min_n=0):
    """Mean score over fluent responses per layer x alpha; NaN where fewer than min_n are fluent."""
    g = df[df.fluent].groupby(["layer", "alpha"]).score
    return g.mean().where(g.size() >= min_n).unstack("alpha")


def select(dprimes, val, test, select_on="val", min_n=MIN_FLUENT):
    """Per-layer alpha*, and the best / top-1 d' / top-3 d' layers."""
    choose = score_table(val if select_on == "val" else test, min_n)
    report = score_table(test, 0 if select_on == "val" else min_n)
    steered = choose.drop(columns=0.0).dropna(how="all")
    alpha = steered.idxmax(axis=1)
    chosen = steered.max(axis=1)
    test_score = pd.Series(
        {l: report.at[l, a] if l in report.index and a in report.columns else np.nan for l, a in alpha.items()},
        dtype=float)
    layers = pd.DataFrame({"dprime": pd.Series(dprimes), "alpha": alpha,
                           "selection_score": chosen, "test_score": test_score}).sort_index()
    ranked = rank_layers(dprimes)
    top3 = [l for l in ranked[:3] if l in chosen.index]
    return dict(
        layers=layers,
        baseline=float(report[0.0].mean()),
        best=int(chosen.idxmax()),
        top1=ranked[0],
        top3=max(top3, key=lambda l: (chosen[l], -l)) if top3 else ranked[0],
        top3_layers=ranked[:3],
    )


def readout(test, layer, alpha):
    """MCC of sign(kappa) against the behavior label. If every response has the same label
    MCC is undefined and accuracy is returned instead (flag=True)."""
    d = test[(test.layer == layer) & (test.alpha == alpha) & test.fluent]
    pred, label = d.kappa > 0, d.label
    if label.nunique() < 2:
        return float((pred == label).mean()), True
    return float(matthews_corrcoef(label, pred)), False


def recovered(score, baseline, best):
    """Share of the best layer's improvement over baseline recovered by `score`."""
    gain = best - baseline
    return (score - baseline) / gain if gain > 0 else np.nan


def summarize(dprimes, val, test, select_on="val", min_n=MIN_FLUENT):
    s = select(dprimes, val, test, select_on, min_n)
    layers, base = s["layers"], s["baseline"]
    at = lambda l: layers.test_score.get(l, np.nan)
    best, top1, top3 = at(s["best"]), at(s["top1"]), at(s["top3"])
    mcc_top1, dag_top1 = readout(test, s["top1"], layers.alpha.get(s["top1"], np.nan))
    mcc_best, dag_best = readout(test, s["best"], layers.alpha.get(s["best"], np.nan))
    return dict(
        baseline=base, top1=top1, top3=top3, best=best,
        top1_layer=s["top1"], top3_layer=s["top3"], best_layer=s["best"],
        top1_recovered=recovered(top1, base, best), top3_recovered=recovered(top3, base, best),
        mcc_top1=mcc_top1, mcc_top1_all_same_label=dag_top1,
        mcc_best=mcc_best, mcc_best_all_same_label=dag_best,
        max_dprime=max(dprimes.values()),
    )
