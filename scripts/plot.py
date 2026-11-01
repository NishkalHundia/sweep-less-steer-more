import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dprime.cli import BEHAVIORS, SETTINGS, read_json
from dprime.selection import MIN_FLUENT, load_run, select

TOP1, TOP3, BEST, DPRIME, MATCH, NONMATCH = "#e66101", "#f4c430", "#1f5fbf", "#2b5c8a", "#4a90c8", "#e06060"


def plot_layers(model_dir, select_on, min_n, out):
    """Per-layer test score at alpha* (top) and training d' (bottom) for every behavior."""
    setting = model_dir.parent.name
    runs = [(b, model_dir / b) for b in BEHAVIORS if (model_dir / b / "test.csv").exists()]
    fig, axes = plt.subplots(2, len(runs), figsize=(max(2.6 * len(runs), 10), 4.2), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    for col, (behavior, run) in enumerate(runs):
        s = select(*load_run(run, behavior), select_on=select_on, min_n=min_n)
        layers = s["layers"]
        top, bottom = axes[0, col], axes[1, col]
        top.plot(layers.index, layers.test_score, color="black", lw=1.4)
        top.axhline(s["baseline"], color="gray", ls=":", lw=1.2)
        top.scatter(s["best"], layers.test_score[s["best"]], marker="*", s=120, color=BEST, zorder=5)
        bottom.plot(layers.index, layers.dprime, color=DPRIME, lw=1.4)
        for ax, y in ((top, layers.test_score), (bottom, layers.dprime)):
            ax.axvline(s["top1"], color=TOP1, lw=1.6, zorder=1)
            ax.scatter(s["top3_layers"], y[s["top3_layers"]], s=28, color=TOP3, edgecolor="black",
                       lw=0.5, zorder=4)
            ax.spines[["top", "right"]].set_visible(False)
        top.set_ylim(0, 1.05)
        bottom.set_ylim(0, None)
        top.set_title(behavior, fontsize=9)
        bottom.set_xlabel("layer")
    axes[0, 0].set_ylabel("test accuracy (best α)" if setting == "mcqa" else "test score (best α)")
    axes[1, 0].set_ylabel("train d'")
    handles = [Line2D([], [], color="black", label="steering"),
               Line2D([], [], color="gray", ls=":", label="baseline"),
               Line2D([], [], color=DPRIME, label="d'"),
               Line2D([], [], marker="o", ls="", color=TOP3, markeredgecolor="black", label="top-3 d' layers"),
               Line2D([], [], color=TOP1, label="top-1 d' layer"),
               Line2D([], [], marker="*", ls="", color=BEST, markersize=10, label="best layer")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_kappa(run, behavior, select_on, min_n, layer, alpha, out):
    """Training kappa vs test-time kappa of steered outputs, split by whether they show the behavior."""
    dprimes, val, test = load_run(run, behavior)
    s = select(dprimes, val, test, select_on=select_on, min_n=min_n)
    layer = s["top1"] if layer is None else layer
    alpha = s["layers"].alpha[layer] if alpha is None else alpha
    train = read_json(Path(run) / "train_kappa.json")[str(layer)]
    d = test[(test.layer == layer) & (test.alpha == alpha) & test.fluent]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.8))
    for ax, pos, neg, title in ((a, train["pos"], train["neg"], "Train"),
                                (b, d.kappa[d.label], d.kappa[~d.label], f"α={alpha:g}")):
        values = np.concatenate([np.asarray(pos, float), np.asarray(neg, float), [0.0]])
        bins = np.linspace(values.min() - 0.2, values.max() + 0.2, 25)
        ax.hist(pos, bins=bins, color=MATCH, alpha=0.8, label="Matching")
        ax.hist(neg, bins=bins, color=NONMATCH, alpha=0.8, label="Non-matching")
        ax.axvline(0, color="black", ls="--", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    a.set_xlabel("κ_a")
    b.set_xlabel("κ_a (post-gen)")
    a.set_ylabel("# prompts")
    a.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{behavior}, layer {layer}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Paper figures from sweep results.")
    p.add_argument("--results", default="results")
    p.add_argument("--select_on", default="val", choices=["val", "test"])
    p.add_argument("--min_fluent", type=int, default=MIN_FLUENT)
    p.add_argument("--kappa", nargs=3, metavar=("SETTING", "MODEL", "BEHAVIOR"),
                   help="plot the kappa histogram for one run instead of the layer figures")
    p.add_argument("--layer", type=int, help="kappa plot layer (default: top-1 d')")
    p.add_argument("--alpha", type=float, help="kappa plot alpha (default: alpha* at that layer)")
    args = p.parse_args()

    out = Path(args.results) / f"figures_{args.select_on}"
    out.mkdir(parents=True, exist_ok=True)
    if args.kappa:
        setting, model, behavior = args.kappa
        path = out / f"kappa_{setting}_{model}_{behavior}.png"
        plot_kappa(Path(args.results) / setting / model / behavior, behavior, args.select_on,
                   args.min_fluent, args.layer, args.alpha, path)
        print(f"wrote {path}")
        return
    for setting in SETTINGS:
        for model_dir in sorted((Path(args.results) / setting).glob("*")):
            path = out / f"layers_{setting}_{model_dir.name}.png"
            plot_layers(model_dir, args.select_on, args.min_fluent, path)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
