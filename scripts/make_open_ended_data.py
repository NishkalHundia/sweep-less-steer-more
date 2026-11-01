import argparse
from pathlib import Path

import torch

from dprime import open_ended
from dprime.cli import BEHAVIORS, read_json, write_json
from dprime.models import load


def main():
    p = argparse.ArgumentParser(description="Build open-ended contrastive pairs from the model's own rollouts.")
    p.add_argument("--model", required=True)
    p.add_argument("--behavior", required=True, choices=BEHAVIORS)
    p.add_argument("--data_dir", default="data/open_ended")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    src = Path(args.data_dir) / args.behavior
    splits = open_ended.split_prompts(
        read_json(src / "open_ended_prompts.json"), read_json(src / "mcqa_prompts.json"), args.seed)
    out = Path(args.data_dir) / "generated" / args.model.split("/")[-1] / args.behavior

    model, tok = load(args.model)
    for split, questions in splits.items():
        pos, neg = (open_ended.rollouts(model, tok, questions, args.behavior, matching,
                                        args.batch_size, args.max_new_tokens) for matching in (True, False))
        pairs = [{"question": q, "answer_matching_behavior": a, "answer_not_matching_behavior": b}
                 for q, a, b in zip(questions, pos, neg)]
        write_json(pairs, out / f"{split}.json")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
