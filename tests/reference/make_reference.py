"""Builds test fixtures from the official HuggingFace GPT-2 implementation.

The C++ tests compare minigpt against these files, so a mismatch means a real bug:
  tokens.json   tokenizer output for tricky strings, plus a greedy continuation
  logits.bin    float32 logits for every position of a short prompt
  model.safetensors / config.json   the model those logits came from

By default it creates a small randomly initialised GPT-2 (2 layers, fast for CI).
With --full-size it creates a randomly initialised model with GPT-2 small's exact shape
(124M parameters), which is useful for benchmarking when the real weights are unavailable.

Usage:
  pip install torch==2.5.1 transformers==4.46.3 safetensors numpy
  python tests/reference/make_reference.py --dir models/test
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

TEXTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces and trailing spaces   ",
    "I'm sure they'll say we've done what you'd expect, isn't it?",
    "'s at the start, and 'RE in caps",
    "Numbers: 12345, 3.14159 and 1,000,000",
    "a\n\nb",
    "tabs\tand\r\nwindows newlines\n\n\n  then text",
    "multiple   spaces   here",
    "Emoji 😀🎉 and accents: café, naïve, Ünïcödé",
    "Dashes — and – quotes “like this” … done",
    "日本語のテキストと中文",
    "Привет, мир! Γειά σου κόσμε",
    "code: for (int i = 0; i < 10; i++) { x += i; }",
    "https://github.com/code-with-sen?tab=repositories#readme",
]

PROMPT = "In a shocking finding, scientists discovered a herd of unicorns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="models/test", help="folder with vocab.json and merges.txt; outputs go here")
    ap.add_argument("--full-size", action="store_true", help="GPT-2 small shape instead of a tiny model")
    args = ap.parse_args()

    tok = GPT2Tokenizer(os.path.join(args.dir, "vocab.json"), os.path.join(args.dir, "merges.txt"))
    tokenized = [{"text": t, "ids": tok.encode(t)} for t in TEXTS]

    torch.manual_seed(0)
    if args.full_size:
        cfg = GPT2Config(vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12)
    else:
        cfg = GPT2Config(vocab_size=50257, n_positions=128, n_embd=128, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(cfg).eval()

    # HuggingFace initialises biases to 0 and LayerNorm to 1/0, which would hide bugs in how
    # we load them. Perturb every parameter so each one actually affects the output.
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.add_(torch.randn_like(p) * 0.02)

    model.save_pretrained(args.dir, safe_serialization=True)

    prompt_ids = tok.encode(PROMPT)
    with torch.no_grad():
        logits = model(torch.tensor([prompt_ids])).logits[0].numpy().astype(np.float32)
        out = model.generate(torch.tensor([prompt_ids]), max_new_tokens=16, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    logits.tofile(os.path.join(args.dir, "logits.bin"))

    with open(os.path.join(args.dir, "tokens.json"), "w", encoding="utf-8") as f:
        json.dump({
            "tokenizer": tokenized,
            "prompt_ids": prompt_ids,
            "greedy_ids": out[0, len(prompt_ids):].tolist(),
            "n_vocab": cfg.vocab_size,
        }, f, ensure_ascii=False, indent=1)
    print(f"wrote fixtures to {args.dir}: {len(prompt_ids)}-token prompt, logits shape {logits.shape}")


if __name__ == "__main__":
    main()
