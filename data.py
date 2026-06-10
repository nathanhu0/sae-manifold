"""
Manifold datasets + Llama-3.1-8B activation extraction (via nnsight).

Each manifold is a dataset of prompts with ground-truth labels that define a
known low-dimensional structure. We extract last-token activations at a single
layer and cache to disk so the LLM only needs to run once.

Manifolds (public data sources only):
  colors        HEX codes -> hue/lightness/saturation (paraboloid)
  temperature   "today the temp is X F" -> fahrenheit/celsius (line)
  age           "they are X years old" -> age (line)
  geography     world cities -> lat/long/country/continent (hierarchical tree)
  days          day x time-of-day grid -> day_idx, time_idx (2D grid, 7x8 = 56 prompts)
  years         "the date is {year}" -> year/decade (helix)
  formality     Pavlick formality scores (line)
  sent_length   WikiText sentences -> token count (line)

Usage:
  uv run data.py                   # extract all manifolds, cache to cache/
  uv run data.py --manifold colors # extract a single manifold
"""
import json
import os
import colorsys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Llama-3.1-8B"
LAYER = 19
D_MODEL = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Large outputs go to scratch (override with MANIFOLD_CACHE_DIR).
CACHE_DIR = Path(os.environ.get("MANIFOLD_CACHE_DIR",
                                "/nlp/scr/nathu/sae-manifold/cache"))


def mpl_colorscale(name, n=11):
    """Sample a matplotlib colormap as a plotly-compatible list."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(name)
    return [[i / (n - 1),
             f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})']
            for i, c in enumerate(cmap(np.linspace(0, 1, n)))]


# ── Dataset builders ─────────────────────────────────────────────────────────
# Each returns (prompts: list[str], labels: list[dict]).

def build_colors():
    from datasets import load_dataset
    ds = load_dataset("burkelibbey/colors", split="train")
    prompts, labels = [], []
    for i, row in enumerate(ds):
        if i % 10 != 0:
            continue
        h_code = row['color']
        r, g, b = (int(h_code.lstrip('#')[j:j+2], 16) / 255 for j in (0, 2, 4))
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        if l < 0.2 or l > 0.8 or s < 0.2 or s > 0.8:
            continue
        prompts.append(f"The hex code {h_code} is for the color")
        labels.append(dict(hex=h_code, hue=h, lightness=l, saturation=s))
    return prompts, labels


def build_age():
    prompts, labels = [], []
    for age in range(1, 100):
        prompts.append(f"They are {age} years old. ")
        labels.append(dict(age=float(age)))
    return prompts, labels


def build_temperature():
    prompts, labels = [], []
    for f in range(-30, 120):
        prompts.append(f"Today it's {f} degrees Fahrenheit outside")
        labels.append(dict(fahrenheit=float(f), celsius=(f - 32) * 5 / 9))
    return prompts, labels


def build_geography():
    from datasets import load_dataset
    ds = load_dataset("jamescalam/world-cities-geo", split="train")
    prompts, labels = [], []
    for i, row in enumerate(ds):
        if i % 3 != 0:
            continue
        prompts.append(
            f"The geographical coordinates latitude {row['latitude']} "
            f"and longitude {row['longitude']} are in the country of"
        )
        labels.append(dict(
            latitude=row['latitude'], longitude=row['longitude'],
            country=row['country'], region=row['region'],
            continent=row['continent'],
        ))
    return prompts, labels


def build_days():
    """Day × time-of-day grid — 7 days × 8 times = 56 prompts.

    Single template ``"It's {time} on {day}"`` so the resulting manifold is a
    clean 2D discrete grid: ``day_idx`` (cyclic) × ``time_idx`` (ordinal).
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]
    times = [
        "the crack of dawn", "early morning", "late morning", "noon",
        "afternoon", "evening", "night", "midnight",
    ]
    prompts, labels = [], []
    for di, day in enumerate(days):
        for ti, time in enumerate(times):
            prompts.append(f"It's {time} on {day}")
            labels.append(dict(day=day, day_idx=di,
                               time=time, time_idx=ti))
    return prompts, labels


def build_years():
    """Years 1800-1999 — helical manifold (linear century + periodic decade)."""
    prompts, labels = [], []
    for year in range(1800, 1999):
        prompts.append(f"The date is {year}")
        labels.append(dict(
            year=float(year),
            decade_digit=float(year % 10),
            decade=float((year % 100) // 10),
        ))
    return prompts, labels


def build_formality():
    from datasets import load_dataset
    ds = load_dataset("osyvokon/pavlick-formality-scores", split="train")
    prompts, labels = [], []
    for row in ds:
        text = row['sentence']
        score = row['avg_score']
        if not text or not text.strip() or len(text.strip()) < 3:
            continue
        prompts.append(text.strip())
        labels.append(dict(formality=float(score)))
        if len(prompts) >= 1000:
            break
    return prompts, labels


def build_sent_length():
    """WikiText sentences labeled by whitespace token count (3-80)."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    prompts, labels = [], []
    seen = set()
    for row in ds:
        text = row['text'].strip()
        if not text or len(text) < 10 or text.startswith('='):
            continue
        for sent in text.replace('? ', '.\n').replace('! ', '.\n').split('.\n'):
            sent = sent.strip()
            if not sent or len(sent) < 10:
                continue
            if not sent.endswith(('.', '?', '!')):
                sent = sent + '.'
            n_tokens = len(sent.split())
            if n_tokens < 3 or n_tokens > 80 or sent in seen:
                continue
            seen.add(sent)
            prompts.append(sent)
            labels.append(dict(n_tokens=float(n_tokens), n_chars=float(len(sent))))
            if len(prompts) >= 5000:
                break
        if len(prompts) >= 5000:
            break
    return prompts, labels


# ── Registry ─────────────────────────────────────────────────────────────────

MANIFOLDS = dict(
    colors=build_colors,
    age=build_age,
    temperature=build_temperature,
    geography=build_geography,
    days=build_days,
    years=build_years,
    formality=build_formality,
    sent_length=build_sent_length,
)


def get_all_manifold_names():
    return list(MANIFOLDS)


# ── LLM loading + extraction ────────────────────────────────────────────────

def load_llm():
    import nnsight
    from transformers import AutoTokenizer
    print(f"Loading {MODEL_NAME}...")
    model = nnsight.LanguageModel(MODEL_NAME, device_map="auto", dispatch=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    for p in model.model.parameters():
        p.requires_grad = False
    return model, tokenizer


@torch.no_grad()
def _extract_one(model, tokenizer, prompt):
    """Last-token activation for a single prompt. Returns [d_model] or None."""
    toks = tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt")
    if toks["input_ids"].shape[1] == 0:
        return None
    with model.trace(toks) as tracer:
        acts = model.model.layers[LAYER].output[0].clone().save()
        tracer.stop()
    acts = acts.squeeze(0)  # [seq_len, d_model]
    if acts.dim() == 1:
        return acts
    return acts[-1]


def _cache(name, activations, labels, prompts):
    if not isinstance(activations, torch.Tensor):
        activations = torch.as_tensor(activations)
    result = dict(activations=activations.cpu(), labels=labels, prompts=prompts)
    path = CACHE_DIR / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    print(f"  Cached {name}: {activations.shape}")
    return result


def extract_standard(name, model, tokenizer):
    """Extract last-token activations for a standard prompt-based manifold."""
    cache = CACHE_DIR / f"{name}.pt"
    if cache.exists():
        print(f"  {name}: already cached")
        return torch.load(cache, weights_only=False)
    prompts, labels = MANIFOLDS[name]()
    print(f"  Extracting {name} ({len(prompts)} samples)...")
    results = [_extract_one(model, tokenizer, p)
               for p in tqdm(prompts, desc=f"  {name}")]
    good = [(i, r) for i, r in enumerate(results)
            if r is not None and r.dim() == 1 and r.shape[0] == D_MODEL]
    if len(good) < len(results):
        print(f"  Dropped {len(results) - len(good)} bad samples")
    idx = [i for i, _ in good]
    acts = torch.stack([r for _, r in good])
    labels = [labels[i] for i in idx]
    prompts = [prompts[i] for i in idx]
    return _cache(name, acts, labels, prompts)


def filter_norm_outliers(data, n_std=6.0):
    """Remove samples with abnormally-large activation norms (attention sinks)."""
    acts = data['activations']
    norms = acts.float().norm(dim=1)
    cutoff = norms.mean() + n_std * norms.std()
    mask = norms <= cutoff
    n_removed = (~mask).sum().item()
    if n_removed == 0:
        return data
    idx = torch.where(mask)[0]
    filtered = dict(
        activations=acts[idx],
        labels=[data['labels'][i] for i in idx.tolist()],
    )
    if 'prompts' in data and data['prompts'] is not None:
        filtered['prompts'] = [data['prompts'][i] for i in idx.tolist()]
    print(f"  Filtered {n_removed} norm outliers (cutoff={cutoff:.1f})")
    return filtered


def load_manifold_data(name, filter_outliers=True, n_std=3.0):
    """Load cached activations + labels, optionally filtering norm outliers."""
    cache = CACHE_DIR / f"{name}.pt"
    if not cache.exists():
        return None
    data = torch.load(cache, weights_only=False)
    if filter_outliers:
        data = filter_norm_outliers(data, n_std=n_std)
    return data


# ── Main ─────────────────────────────────────────────────────────────────────

def run_extraction(manifolds=None):
    model, tokenizer = load_llm()
    names = manifolds or get_all_manifold_names()
    for name in names:
        print(f"\n{name}")
        if name in MANIFOLDS:
            extract_standard(name, model, tokenizer)
        else:
            print(f"  Unknown manifold: {name}")
    print("\nDone.")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description=f"Extract {MODEL_NAME} layer {LAYER} activations for manifold datasets")
    p.add_argument('--manifold', nargs='*', default=None,
                   help='Specific manifolds to extract (default: all)')
    a = p.parse_args()
    run_extraction(a.manifold)
