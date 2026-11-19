# Design notes

## Goal

Run GPT-2 text generation on a CPU with no dependencies beyond a C++17 compiler, produce the
same numbers as the reference implementation, and make each speed-up measurable on its own.

## How a token is generated

```
text ──► tokenizer ──► token ids ──► [embedding + position]
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  × 12 layers                        │
                       │   LayerNorm → Q,K,V → attention     │◄── KV cache (keys/values of
                       │   over all earlier tokens → add     │    every earlier token)
                       │   LayerNorm → MLP (GELU) → add      │
                       └──────────────────┬──────────────────┘
                                          ▼
                     LayerNorm → logits over 50,257 tokens → sampler → next token
```

Generation is one token at a time, so almost all of the work is matrix–vector products
(a 768-value vector times weight matrices). That shapes every decision below.

## Decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| **KV cache** | Without it, every new token reprocesses the whole sequence, so cost grows with length squared. With it, each layer stores keys/values once. | Memory: 2 × layers × context × 768 floats (~75 MB for 1,024 tokens). |
| **Weights stored as `[out][in]`** | Each output is one dot product over contiguous memory, which is cache- and SIMD-friendly. HuggingFace stores `[in][out]`, so we transpose once at load. | Slower load (~1 s). |
| **Persistent thread pool, split by output rows** | ~50 matrix–vector products per token; spawning threads each time would cost more than the work. Rows are independent, so no locking. | Little gain beyond memory bandwidth limits. |
| **int8 in blocks of 32 (Q8_0)** | Decoding is limited by how fast weights stream from memory; int8 moves 3.5× fewer bytes. A scale per 32 values keeps outliers from hurting accuracy. Activations are quantized too, so the inner loop is integer multiply-adds. | Accuracy loss: with the real GPT-2 weights, the same next token as fp32 84% of the time (differences mostly where two words are nearly tied). |
| **Hand-written AVX2 / NEON for int8 only** | The compiler vectorized the fp32 dot product well, but not the int8 one: auto-vectorized int8 was *slower* than fp32. Explicit SIMD made it 2.2× (AVX2) to 2.4× (Apple NEON) faster than fp32. | Two extra code paths, each tested in CI on its own hardware. |
| **No `-ffast-math`** | Keeps results deterministic and comparable to PyTorch to ~1e-6. Dot products use 8 accumulators so they still vectorize. | Slightly slower fp32. |
| **Hand-written pre-tokenizer instead of a regex library** | `std::regex` has no Unicode classes; adding ICU or PCRE would add a dependency. | Unicode classes are approximated by block ranges (exact for ASCII; tested against HuggingFace on mixed scripts and emoji). |

## Verification strategy

Correctness is checked against HuggingFace `transformers` rather than against hand-picked
outputs. `tests/reference/make_reference.py` builds a GPT-2 with randomly perturbed weights
(so every bias and LayerNorm parameter matters), then saves its logits, greedy output and
tokenizer output. The C++ tests require:

- logits within 1e-3 of PyTorch at every position (observed: ~1e-6),
- identical greedy generations,
- identical tokenization on tricky strings (contractions, whitespace runs, emoji, CJK, Cyrillic),
- bit-identical logits with and without KV cache, and with 1 vs 3 threads,
- int8 logits within 5% of the largest logit.

CI runs this on x86 (AVX2), x86 without SIMD, Apple Silicon (NEON), and under
AddressSanitizer + UBSan.

## Not done (yet)

- Batched prompt processing (prefill is token by token; a matrix–matrix path would be faster).
- Keeping the output layer in fp32 when using int8, to trade some memory for accuracy.
- 4-bit quantization.
- GPU support.
