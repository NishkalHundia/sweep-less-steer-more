import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Direction:
    mu_pos: torch.Tensor
    mu_neg: torch.Tensor

    @property
    def v(self):
        return self.mu_pos - self.mu_neg

    def kappa(self, acts):
        """Position along the DIM line: -1 at mu_neg, +1 at mu_pos (Eq. 1)."""
        mu = 0.5 * (self.mu_pos + self.mu_neg)
        v = self.v
        return (2.0 * (acts.float() - mu) @ v / (v @ v)).numpy()


@dataclass
class LayerFit:
    direction: Direction
    dprime: float
    kappa_pos: np.ndarray
    kappa_neg: np.ndarray


def dprime(kappa_pos, kappa_neg):
    """Discriminability index (Eq. 2)."""
    pooled = np.sqrt(0.5 * (np.var(kappa_pos) + np.var(kappa_neg)))
    if pooled == 0:
        return 0.0
    return float(abs(np.mean(kappa_pos) - np.mean(kappa_neg)) / pooled)


def fit(pos, neg):
    """Fit a difference-in-means direction from [N, H] positive and negative activations."""
    direction = Direction(pos.float().mean(0), neg.float().mean(0))
    k_pos, k_neg = direction.kappa(pos), direction.kappa(neg)
    return LayerFit(direction, dprime(k_pos, k_neg), k_pos, k_neg)


def fit_layers(acts, labels):
    """acts: {layer: [N, H]}, labels: [N] bool (True = behavior-matching)."""
    labels = torch.as_tensor(labels, dtype=torch.bool)
    return {layer: fit(a[labels], a[~labels]) for layer, a in acts.items()}


def rank_layers(dprimes):
    """Layers ordered by d', highest first; ties go to the lower layer."""
    return sorted(dprimes, key=lambda layer: (-dprimes[layer], layer))


def save(fits, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dprime.json").write_text(json.dumps({l: f.dprime for l, f in fits.items()}, indent=2))
    kappas = {l: {"pos": f.kappa_pos.tolist(), "neg": f.kappa_neg.tolist()} for l, f in fits.items()}
    (out / "train_kappa.json").write_text(json.dumps(kappas))
    torch.save({l: {"mu_pos": f.direction.mu_pos, "mu_neg": f.direction.mu_neg} for l, f in fits.items()},
               out / "directions.pt")
