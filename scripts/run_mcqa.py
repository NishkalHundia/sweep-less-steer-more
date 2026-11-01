import argparse
from pathlib import Path

import torch

from dprime import geometry, mcqa
from dprime.cli import BEHAVIORS, parse_floats, parse_layers, read_json, run_dir
from dprime.models import load, num_layers


def main():
    p = argparse.ArgumentParser(description="MCQA layer sweep: d' per layer and steered accuracy.")
    p.add_argument("--model", required=True)
    p.add_argument("--behavior", required=True, choices=BEHAVIORS)
    p.add_argument("--layers", default="10-31")
    p.add_argument("--alphas", default="1,2,3,5,10")
    p.add_argument("--data_dir", default="data/mcqa")
    p.add_argument("--results", default="results")
    p.add_argument("--n_train", type=int, default=300)
    p.add_argument("--n_val", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    data = Path(args.data_dir) / args.behavior
    raw, test = read_json(data / "train.json"), read_json(data / "test.json")
    train_idx, val_idx = mcqa.split_indices(len(raw), args.n_train, args.n_val, args.seed)
    out = run_dir(args.results, "mcqa", args.model, args.behavior)

    model, tok = load(args.model)
    layers = parse_layers(args.layers, num_layers(model))
    fits = mcqa.fit(model, tok, [raw[i] for i in train_idx], layers, args.batch_size)
    geometry.save(fits, out)

    alphas = parse_floats(args.alphas)
    for split, items in (("val", [raw[i] for i in val_idx]), ("test", test)):
        df = mcqa.evaluate(model, tok, items, fits, alphas, args.batch_size)
        df.to_csv(out / f"{split}.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
