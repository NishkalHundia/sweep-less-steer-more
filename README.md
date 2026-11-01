# Sweep Less, Steer More

Code for *Sweep Less, Steer More: Using Activation Steering Geometry for Reduced Hyperparameter Search*.

Difference-in-means (DIM) steering usually needs a sweep over layers and steering strengths. We rank layers by the discriminability index d′ of the training activations, a quantity you get for free from the data used to build the steering vector, and steer only at the top-k layers. This cuts the number of steered runs from 110 (22 layers × 5 α) to 5 (top-1) or 15 (top-3).

## Setup

```bash
pip install -e .
export OPENAI_API_KEY=...   # open-ended judging only
```

Models used in the paper: `google/gemma-2-9b-it`, `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3.5-9B`. Each run fits on one A100 (40GB).

## Using d′ on your own behavior

```python
from dprime import open_ended
from dprime.geometry import rank_layers
from dprime.models import load, num_layers

model, tok = load("google/gemma-2-9b-it")
pairs = [{"question": ..., "answer_matching_behavior": ..., "answer_not_matching_behavior": ...}, ...]
fits = open_ended.fit(model, tok, pairs, list(range(num_layers(model))), regime="mean")
ranked = rank_layers({layer: f.dprime for layer, f in fits.items()})
top3 = ranked[:3]
```

Each `fits[layer]` holds the DIM `direction` (with `.v` for steering and `.kappa(acts)` for projecting activations onto the DIM line), the training `dprime`, and the training projections. `examples/rank_layers.py` does this from a JSON file and optionally steers a prompt at the top layer.

## Reproducing the paper

Behaviors: `myopic-reward`, `corrigible-neutral-HHH`, `hallucination`, `survival-instinct`, `sycophancy`. All commands below run one model × behavior; loop over behaviors and models for the full set.

**MCQA (§4).** 300 training pairs and 50 validation questions sampled from `data/mcqa/<behavior>/train.json`, 50 test questions from `test.json`.

```bash
python scripts/run_mcqa.py --model google/gemma-2-9b-it --behavior myopic-reward
```

**Open-ended (§5).** First build contrastive pairs from the model's own rollouts under behavior-matching and opposite cues (128 train / 32 val / 32 test), then sweep each training regime:

```bash
python scripts/make_open_ended_data.py --model google/gemma-2-9b-it --behavior myopic-reward
for regime in mean prompt last; do
  python scripts/run_open_ended.py --model google/gemma-2-9b-it --behavior myopic-reward --regime $regime
done
```

`--skip_judge` saves generations without scoring; `--judge_only` scores them later without a GPU.

**Tables and figures.**

```bash
python scripts/analyze.py                  # Tables 1, 2, 3, 5 -> results/summary_val/tables.md
python scripts/plot.py                     # Figures 2, 4-7 -> results/figures_val/
python scripts/plot.py --kappa mcqa gemma-2-9b-it myopic-reward   # Figure 3
```

`--select_on test` picks α and the best / top-3 layer on the test split instead of validation.

Every run writes to `results/<setting>/<model>/<behavior>/`:

| File | Contents |
|---|---|
| `dprime.json` | training d′ per layer |
| `train_kappa.json` | training projections κ per layer |
| `directions.pt` | class means μ⁺, μ⁻ per layer |
| `val.csv`, `test.csv` | one row per (layer, α, prompt): output, κ of the output, and judge scores (open-ended) |

## Layout

```
dprime/
  geometry.py     DIM direction, κ, d', layer ranking
  models.py       model loading, chat templates, activation capture, steering hook
  mcqa.py         MCQA encoding, fitting, evaluation
  open_ended.py   open-ended fitting (mean / last / prompt), generation, data construction
  judge.py        GPT-4o-mini behavior and fluency judges
  selection.py    α selection, top-k layer selection, recovered gain, sign(κ) MCC
scripts/          entry points for the sweeps, tables and figures
examples/         d' layer ranking on your own contrastive pairs
data/             MCQA datasets and open-ended prompt pools
```

## Data

The behavior datasets are from [Rimsky et al. (2024)](https://github.com/nrimsky/CAA) (MIT License). The fluency judge prompt is from [AxBench](https://github.com/stanfordnlp/axbench) (Apache 2.0).
