"""Rank layers by d' for your own contrastive pairs, then steer at the top-ranked layer.

pairs.json: [{"question": ..., "answer_matching_behavior": ..., "answer_not_matching_behavior": ...}, ...]
"""
import argparse

import torch

from dprime import open_ended
from dprime.cli import read_json
from dprime.geometry import rank_layers
from dprime.models import chat_ids, load, num_layers, steer, user


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--regime", default="mean", choices=["mean", "last"])
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--prompt", help="optional prompt to generate from with steering at the top layer")
    p.add_argument("--alpha", type=float, default=2.0)
    args = p.parse_args()

    model, tok = load(args.model)
    fits = open_ended.fit(model, tok, read_json(args.pairs), list(range(num_layers(model))), args.regime)
    dprimes = {layer: f.dprime for layer, f in fits.items()}
    ranked = rank_layers(dprimes)
    for layer in ranked[:args.top_k]:
        print(f"layer {layer:2d}  d'={dprimes[layer]:.2f}")

    if args.prompt:
        layer = ranked[0]
        ids = torch.tensor([chat_ids(tok, user(args.prompt), True)], device=model.device)
        with steer(model, layer, fits[layer].direction.v.to(model.device), args.alpha):
            out = model.generate(ids, max_new_tokens=200, do_sample=False)
        print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
