import random
import re

import pandas as pd
import torch

from .geometry import fit_layers
from .models import (activations, batches, chat_ids, hidden_at, is_qwen35, maybe_steer,
                     next_tokens, pad, shared_prefix_len, user)


def answer_letter(answer):
    return re.search(r"\(([A-Z])\)", answer.upper()).group(1)


def split_indices(n, n_train=300, n_val=50, seed=42):
    """Train and validation indices sampled from the raw behavior dataset."""
    rng = random.Random(seed)
    train = sorted(rng.sample(range(n), min(n_train, n)))
    taken = set(train)
    rest = [i for i in range(n) if i not in taken]
    val = sorted(rng.sample(rest, min(n_val, len(rest))))
    return train, val


def encode(tok, question, letter):
    """Token ids for the question answered with ' (letter)', and the index of the letter token."""
    prompt = chat_ids(tok, user(question), True)
    answer = f" ({letter})"
    candidates = [prompt + tok.encode(answer, add_special_tokens=False)]
    if not is_qwen35(tok.name_or_path):
        turn = user(question) + [{"role": "assistant", "content": answer}]
        candidates.insert(0, chat_ids(tok, turn, False))
    for ids in candidates:
        if ids[:len(prompt)] != prompt:
            continue
        for i in range(len(ids) - 1, len(prompt) - 1, -1):
            if tok.decode([ids[i]]).strip() == letter:
                return ids, i
    raise ValueError(f"answer letter {letter!r} not found after prompt")


def fit(model, tok, pairs, layer_ids, batch_size=16):
    seqs, sites, labels = [], [], []
    for pair in pairs:
        for key, label in (("answer_matching_behavior", True), ("answer_not_matching_behavior", False)):
            ids, site = encode(tok, pair["question"], answer_letter(pair[key]))
            seqs.append(ids)
            sites.append(site)
            labels.append(label)
    acts = activations(model, tok, seqs, sites, layer_ids, batch_size)
    return fit_layers(acts, labels)


@torch.no_grad()
def evaluate(model, tok, items, fits, alphas, batch_size=16):
    """Greedy answer at '(' for alpha=0 and every (layer, alpha), with kappa of the chosen
    letter from a second, unsteered forward pass."""
    layer_ids = list(fits)
    vectors = {l: fits[l].direction.v.to(model.device) for l in layer_ids}
    start = shared_prefix_len(tok)
    prompts = []
    for idx, item in enumerate(items):
        letter = answer_letter(item["answer_matching_behavior"])
        ids, site = encode(tok, item["question"], letter)
        prompts.append((idx, letter, ids[:site]))

    rows = []

    def run(read_layers, alpha, steer_layer=None):
        for batch in batches(prompts, batch_size):
            prefixes = [p for _, _, p in batch]
            ids, mask = pad(prefixes, tok.pad_token_id, model.device)
            with maybe_steer(model, steer_layer, vectors.get(steer_layer), alpha, start):
                chosen = next_tokens(model, ids, mask, [len(p) - 1 for p in prefixes])
            seqs = [p + [t] for p, t in zip(prefixes, chosen)]
            ids, mask = pad(seqs, tok.pad_token_id, model.device)
            h = hidden_at(model, ids, mask, read_layers, [len(s) - 1 for s in seqs])
            for layer in read_layers:
                kappas = fits[layer].direction.kappa(h[layer])
                for (idx, letter, _), t, k in zip(batch, chosen, kappas):
                    token = tok.decode([t]).strip()
                    rows.append(dict(layer=layer, alpha=alpha, idx=idx, token=token,
                                     match=token == letter, kappa=float(k)))

    run(layer_ids, 0.0)
    for layer in layer_ids:
        for alpha in alphas:
            run([layer], alpha, layer)
    return pd.DataFrame(rows)
