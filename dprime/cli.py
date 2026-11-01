import json
from pathlib import Path

BEHAVIORS = ["myopic-reward", "corrigible-neutral-HHH", "hallucination", "survival-instinct", "sycophancy"]
SETTINGS = ["mcqa", "oe-mean", "oe-prompt", "oe-last"]


def parse_layers(spec, n_layers):
    if "-" in spec:
        lo, hi = map(int, spec.split("-"))
        layers = range(lo, hi + 1)
    else:
        layers = map(int, spec.split(","))
    return [l for l in layers if 0 <= l < n_layers]


def parse_floats(spec):
    return [float(x) for x in spec.split(",")]


def run_dir(root, setting, model, behavior):
    return Path(root) / setting / model.split("/")[-1] / behavior


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
