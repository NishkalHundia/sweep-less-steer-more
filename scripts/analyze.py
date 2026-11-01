import argparse
from pathlib import Path

import pandas as pd

from dprime.cli import BEHAVIORS, SETTINGS
from dprime.selection import MIN_FLUENT, load_run, summarize

TITLES = {
    "mcqa": "MCQA steering accuracy",
    "oe-mean": "Mean-pooled steering score",
    "oe-prompt": "Prompt-only steering score",
    "oe-last": "Last-token steering score",
}


def collect(results, select_on, min_n):
    rows = []
    for setting in SETTINGS:
        for model_dir in sorted((Path(results) / setting).glob("*")):
            for behavior in BEHAVIORS:
                run = model_dir / behavior
                if (run / "test.csv").exists():
                    s = summarize(*load_run(run, behavior), select_on=select_on, min_n=min_n)
                    rows.append(dict(setting=setting, model=model_dir.name, behavior=behavior, **s))
    return pd.DataFrame(rows)


def steering_table(d):
    models = list(dict.fromkeys(d.model))
    lines = ["| Behavior | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
    for b in [b for b in BEHAVIORS if b in set(d.behavior)]:
        cells = []
        for m in models:
            r = d[(d.model == m) & (d.behavior == b)]
            cells.append("" if r.empty else
                         "{baseline:.2f} → {top1:.2f} / {top3:.2f} ({best:.2f})".format(**r.iloc[0]))
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    pct = lambda x: f"{x.median():.0%}" if x.notna().any() else "n/a"
    med = f"{pct(d.top1_recovered)} / {pct(d.top3_recovered)}"
    lines.append("| Median % of improvement (top-1 / top-3) | " + med + " |" + " |" * (len(models) - 1))
    return lines


def readout_table(d):
    models = list(dict.fromkeys(d.model))
    lines = ["| Behavior | Max d' | " + " | ".join(models) + " |", "|---" * (len(models) + 2) + "|"]
    for b in BEHAVIORS:
        db = d[d.behavior == b]
        if db.empty:
            continue
        cells = []
        for m in models:
            r = db[db.model == m]
            if r.empty:
                cells.append("")
                continue
            r = r.iloc[0]
            cells.append(f"{r.mcc_top1:.2f}{'†' if r.mcc_top1_all_same_label else ''} "
                         f"({r.mcc_best:.2f}{'†' if r.mcc_best_all_same_label else ''})")
        lines.append(f"| {b} | {db.max_dprime.min():.1f}–{db.max_dprime.max():.1f} | " + " | ".join(cells) + " |")
    return lines


def main():
    p = argparse.ArgumentParser(description="Layer-selection and readout tables from sweep results.")
    p.add_argument("--results", default="results")
    p.add_argument("--select_on", default="val", choices=["val", "test"],
                   help="split used to pick alpha per layer and the best / top-3 layer")
    p.add_argument("--min_fluent", type=int, default=MIN_FLUENT)
    args = p.parse_args()

    df = collect(args.results, args.select_on, args.min_fluent)
    out = Path(args.results) / f"summary_{args.select_on}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)

    md = ["Steering: baseline → top-1 / top-3 (best). Readout: MCC of sign(κ) at the top-1 d' layer "
          "(best layer); † = every response has the same label, accuracy reported.", ""]
    for setting in SETTINGS:
        d = df[df.setting == setting]
        if d.empty:
            continue
        md += [f"## {TITLES[setting]}", ""] + steering_table(d) + ["", "Readout", ""] + readout_table(d) + [""]
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out / 'tables.md'}")


if __name__ == "__main__":
    main()
