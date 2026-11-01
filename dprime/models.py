from contextlib import contextmanager, nullcontext

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def is_qwen35(name):
    return "qwen3.5" in name.lower().replace("_", ".")


def load(model_name, device="cuda", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type == "qwen3_5":
        model = _load_qwen35(model_name, config, device, dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device)
    return model.eval(), tok


def _load_qwen35(model_name, config, device, dtype):
    # Qwen3.5 is published as a vision-language checkpoint; we only need the text stack.
    try:
        from transformers import Qwen3_5ForCausalLM
        return Qwen3_5ForCausalLM.from_pretrained(
            model_name, config=config.text_config, dtype=dtype, device_map=device)
    except Exception:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype, device_map=device)


def decoder(model):
    inner = model.model
    return inner if hasattr(inner, "layers") else inner.language_model


def num_layers(model):
    return len(decoder(model).layers)


def chat_ids(tok, messages, generation_prompt):
    kwargs = {"enable_thinking": False} if is_qwen35(tok.name_or_path) else {}
    out = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=generation_prompt, **kwargs)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(t) for t in out]


def user(text):
    return [{"role": "user", "content": text}]


def eos_ids(tok):
    eos = tok.eos_token_id
    ids = set(eos if isinstance(eos, list) else [eos])
    names = getattr(tok, "additional_special_tokens", []) or []
    tids = getattr(tok, "additional_special_tokens_ids", []) or []
    ids.update(i for name, i in zip(names, tids) if "end_of_turn" in name.lower())
    return ids


def shared_prefix_len(tok):
    """Number of leading chat-template tokens shared by every prompt (e.g. BOS + turn header)."""
    a = chat_ids(tok, user("1"), False)
    b = chat_ids(tok, user("2"), False)
    return next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)


def pad(seqs, pad_id, device, left=False):
    n = max(map(len, seqs))
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        span = slice(n - len(s), n) if left else slice(0, len(s))
        ids[i, span] = torch.tensor(s)
        mask[i, span] = 1
    return ids.to(device), mask.to(device)


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


@contextmanager
def capture(model, layer_ids):
    store = {}

    def make_hook(layer):
        def hook(module, inputs, output):
            store[layer] = output[0] if isinstance(output, tuple) else output
        return hook

    layers = decoder(model).layers
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in layer_ids]
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def steer(model, layer, vector, alpha, start=0):
    """Add alpha * vector to the residual stream after `layer`, at positions >= start."""
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, start:] += alpha * vector.to(h.device, h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    handle = decoder(model).layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def maybe_steer(model, layer, vector, alpha, start=0):
    return steer(model, layer, vector, alpha, start) if alpha else nullcontext()


@torch.no_grad()
def hidden_at(model, ids, mask, layer_ids, positions):
    """Residual stream at positions[i] (int, or inclusive (start, end) span to mean-pool)."""
    with capture(model, layer_ids) as store:
        decoder(model)(input_ids=ids, attention_mask=mask)
    out = {}
    for layer in layer_ids:
        h = store[layer]
        rows = [h[i, p[0]:p[1] + 1].float().mean(0) if isinstance(p, tuple) else h[i, p].float()
                for i, p in enumerate(positions)]
        out[layer] = torch.stack(rows).cpu()
    return out


@torch.no_grad()
def next_tokens(model, ids, mask, positions):
    """Greedy next token predicted at each row's position."""
    h = decoder(model)(input_ids=ids, attention_mask=mask).last_hidden_state
    h = h[torch.arange(len(positions), device=h.device), torch.tensor(positions, device=h.device)]
    return model.get_output_embeddings()(h).argmax(-1).tolist()


@torch.no_grad()
def activations(model, tok, seqs, positions, layer_ids, batch_size):
    out = {layer: [] for layer in layer_ids}
    for i in range(0, len(seqs), batch_size):
        ids, mask = pad(seqs[i:i + batch_size], tok.pad_token_id, model.device)
        h = hidden_at(model, ids, mask, layer_ids, positions[i:i + batch_size])
        for layer in layer_ids:
            out[layer].append(h[layer])
    return {layer: torch.cat(v) for layer, v in out.items()}
