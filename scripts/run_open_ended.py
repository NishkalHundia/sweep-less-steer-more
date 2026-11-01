import argparse
from pathlib import Path

import pandas as pd
import torch

from dprime import geometry, judge, open_ended
from dprime.cli import BEHAVIORS, parse_floats, parse_layers, read_json, run_dir
from dprime.models import load, num_layers


def add_scores(path, behavior, judge_model):
    df = pd.read_csv(path)
    df["generation"] = df["generation"].fillna("")
    todo = df[["question", "generation"]].drop_duplicates()
    scores = judge.score(behavior, list(todo.itertuples(index=False, name=None)), judge_model)
    todo["behavior"], todo["fluency"] = zip(*scores)
    df = df.drop(columns=["behavior", "fluency"], errors="ignore").merge(todo, on=["question", "generation"])
    df.to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser(description="Open-ended layer sweep: d' per layer, steered generations, LLM judge.")
    p.add_argument("--model", required=True)
    p.add_argument("--behavior", required=True, choices=BEHAVIORS)
    p.add_argument("--regime", required=True, choices=open_ended.REGIMES)
    p.add_argument("--layers", default="10-31")
    p.add_argument("--alphas", default="1,2,3,5,10")
    p.add_argument("--data_dir", default="data/open_ended/generated")
    p.add_argument("--results", default="results")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--skip_judge", action="store_true")
    p.add_argument("--judge_only", action="store_true", help="score existing generations without a GPU")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    out = run_dir(args.results, f"oe-{args.regime}", args.model, args.behavior)
    if not args.judge_only:
        data = Path(args.data_dir) / args.model.split("/")[-1] / args.behavior
        model, tok = load(args.model)
        layers = parse_layers(args.layers, num_layers(model))
        fits = open_ended.fit(model, tok, read_json(data / "train.json"), layers, args.regime,
                              args.behavior, args.batch_size)
        geometry.save(fits, out)
        alphas = parse_floats(args.alphas)
        for split in ("val", "test"):
            questions = [x["question"] for x in read_json(data / f"{split}.json")]
            df = open_ended.evaluate(model, tok, questions, fits, args.regime, alphas,
                                     args.batch_size, args.max_new_tokens)
            df.to_csv(out / f"{split}.csv", index=False)

    if not args.skip_judge:
        for split in ("val", "test"):
            add_scores(out / f"{split}.csv", args.behavior, args.judge_model)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
