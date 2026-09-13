# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Build fixed probe prompt pools so every probe period (and every demo stage) replays the same
requests.

Usage: make_probe_pool.py OUT_PREFIX --n 4 --batch 64 --in-len 128 [--source sharegpt|random]
Writes OUT_PREFIX_<i>.json (a list of `batch` token-id lists of `in_len` tokens), i in 0..n-1.
sharegpt: human turns of ShareGPT_V3 (HF cache), tokenized with the model tokenizer, cut to in_len.
"""

import argparse
import glob
import json
import os
import random
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ap = argparse.ArgumentParser()
ap.add_argument("out_prefix")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--in-len", type=int, default=128)
ap.add_argument("--source", choices=["sharegpt", "random"], default="sharegpt")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--vocab", type=int, default=150000)
ap.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B-FP8")
ap.add_argument("--sharegpt", default="auto")
a = ap.parse_args()
rng = random.Random(a.seed)
need = a.n * a.batch
if a.source == "random":
    pool = [[rng.randrange(1000, a.vocab) for _ in range(a.in_len)] for _ in range(need)]
else:
    from transformers import AutoTokenizer

    path = a.sharegpt
    if path == "auto":
        cands = glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/*/ShareGPT_V3_unfiltered_cleaned_split.json"
            )
        )
        if not cands:
            sys.exit("ShareGPT json not found in the HF cache")
        path = cands[0]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    with open(path) as fh:
        data = json.load(fh)
    rng.shuffle(data)
    pool = []
    for conv in data:
        text = " ".join(
            t.get("value", "") for t in conv.get("conversations", []) if t.get("from") == "human"
        )
        if len(text) < 4 * a.in_len:
            continue  # cheap pre-filter before tokenizing
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) >= a.in_len:
            pool.append(ids[: a.in_len])
        if len(pool) == need:
            break
    if len(pool) < need:
        sys.exit(f"only {len(pool)} ShareGPT prompts with >= {a.in_len} tokens")
for i in range(a.n):
    with open(f"{a.out_prefix}_{i}.json", "w") as fh:
        json.dump(pool[i * a.batch : (i + 1) * a.batch], fh)
print(
    f"wrote {a.n} pools x {a.batch} prompts x {a.in_len} tokens ({a.source}) "
    f"-> {a.out_prefix}_<i>.json"
)
