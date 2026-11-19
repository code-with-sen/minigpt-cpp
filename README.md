# minigpt-cpp

[![CI](https://github.com/code-with-sen/minigpt-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/code-with-sen/minigpt-cpp/actions/workflows/ci.yml)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

**GPT-2 inference from scratch in C++17**: tokenizer, transformer, sampling, KV cache, a thread pool and int8 SIMD kernels, in about 1,450 lines of C++ with no dependencies. It runs OpenAI's GPT-2 weights on a CPU, matches the official PyTorch implementation to within **5e-6**, and generates text **3.3× faster than PyTorch** on the same machine (1.5× faster at the same fp32 precision). On an Apple Silicon MacBook Air it reaches **240 tokens/sec**.

```
$ ./minigpt -p "The future of AI is" -n 50 --int8
loaded GPT-2: 12 layers, 12 heads, 768 dims, 143 MB of weights (int8), 8 threads (0.8s)
The future of AI is being explored. But in the meantime, we can't say goodbye to people. The future of AI
is being explored. But in the meantime, we can't say goodbye to people.

What we have to do is identify the problem and address
prompt: 5 tokens in 0.03s | generated: 50 tokens in 0.23s (221.1 tokens/s)
```

*Real output from OpenAI's GPT-2 (124M) on a MacBook Air. Repeating itself is typical of GPT-2 small with greedy-ish sampling; raise `-t` for more variety.*

## Results

Greedy generation of 32 tokens after an 18-token prompt. Each row adds one optimization.

**Apple Silicon MacBook Air, 8 threads (NEON), real GPT-2 weights:**

| Configuration | Tokens/sec | Speed-up | Weights in memory |
|---|---:|---:|---:|
| Baseline: 1 thread, no KV cache, fp32 | 2.3 | 1.0× | 498 MB |
| + KV cache | 57.6 | 25.3× | 498 MB |
| + multithreading (8 threads) | 99.1 | 43.5× | 498 MB |
| + int8 weights (NEON) | **239.8** | **105.2×** | **143 MB** |

**2-core Intel Xeon cloud VM (AVX2)**, compared with PyTorch on the same machine. This run used a model with GPT-2 small's exact shape but random weights (the VM could not download the real ones), which takes exactly the same time per token:

| Configuration | Tokens/sec | Speed-up |
|---|---:|---:|
| Baseline: 1 thread, no KV cache, fp32 | 1.2 | 1.0× |
| + KV cache | 31.9 | 25.7× |
| + multithreading (2 threads) | 49.6 | 39.9× |
| + int8 weights (AVX2) | **109.1** | **87.9×** |
| *PyTorch 2.5 + HuggingFace `generate`, 2 threads, fp32* | *32.8* | |

- **KV cache and threads change nothing in the output:** logits are bit-identical with and without them.
- **int8 is an accuracy trade-off.** With the real GPT-2 weights, int8 picks the same next token as fp32 at 84% of positions (42 of 50, each fed the same history). Most differences are where the top two words are almost tied. Use fp32 (no `--int8`) when exact output matters.
- Reproduce with `./build/minigpt-bench models/gpt2`.

## Correctness

Tests compare against HuggingFace `transformers` as the reference, not against hand-picked outputs:

| Check | Result |
|---|---|
| Logits vs PyTorch at every position (2-layer and full 12-layer models) | max difference **8.9e-7** and **4.6e-6** |
| Greedy generation vs `model.generate()` | identical tokens |
| Tokenizer vs `GPT2Tokenizer` on 15 tricky strings (contractions, whitespace runs, emoji, CJK, Cyrillic, URLs, code) | 15 / 15 identical |
| KV cache vs no cache, 1 vs 3 threads | bit-identical logits |
| Sanitizers (AddressSanitizer + UBSan) | clean |

CI runs the full suite on **x86 with AVX2**, **x86 without SIMD**, **Apple Silicon (NEON)** and under sanitizers.

## How it works

```
"Hello world" ──► tokenizer ──► [15496, 995] ──► embeddings
                                                     │
                            ┌────────────────────────▼───────────────┐
                            │ × 12 transformer layers                │
                            │   LayerNorm → attention ◄── KV cache   │
                            │   LayerNorm → MLP (GELU)               │
                            └────────────────────────┬───────────────┘
                                                     ▼
                          logits for 50,257 tokens ──► sampler ──► next token
```

| File | What it does |
|---|---|
| [`src/tokenizer.cpp`](src/tokenizer.cpp) | GPT-2's byte-level BPE: a hand-written version of its pre-tokenizer regex, the byte↔Unicode table, and merging by rank |
| [`src/model.cpp`](src/model.cpp) | Loads weights and runs the transformer one token at a time, with the KV cache |
| [`src/ops.hpp`](src/ops.hpp) | Math kernels: dot products, LayerNorm, GELU, softmax, and Q8_0 int8 quantization with AVX2 / NEON paths |
| [`src/thread_pool.hpp`](src/thread_pool.hpp) | Persistent workers with `parallel_for`, reused across ~50 matrix products per token |
| [`src/safetensors.hpp`](src/safetensors.hpp) | Reads HuggingFace `.safetensors` files (fp32, fp16, bf16) |
| [`src/json.hpp`](src/json.hpp) | Small JSON parser for the safetensors header and `vocab.json` |
| [`src/sampler.hpp`](src/sampler.hpp) | Greedy, temperature and top-k sampling |
| [`tests/`](tests) | C++ tests plus the script that builds reference outputs with PyTorch |

Design decisions and trade-offs are in **[docs/DESIGN.md](docs/DESIGN.md)**. The main ones:

1. **KV cache.** Each layer keeps the keys and values of earlier tokens, so a new token costs one pass instead of reprocessing the whole sequence (**~26× faster** here).
2. **Thread pool.** Matrix rows are split across persistent threads with no locking, since rows are independent.
3. **int8, in blocks of 32 values with one scale each.** Generating text is limited by how fast weights stream from memory, and int8 moves 3.5× fewer bytes. The first version, left to the compiler, was *slower* than fp32. Hand-written AVX2 (`maddubs`/`madd`) and NEON (`vdotq_s32`) kernels made it **2.2–2.4× faster** than fp32.

## Build and run

Needs a C++17 compiler (GCC or Clang; both are tested in CI). There are no other dependencies.

```bash
git clone https://github.com/code-with-sen/minigpt-cpp && cd minigpt-cpp
cmake -S . -B build && cmake --build build -j
sh scripts/download_model.sh                       # or: gpt2-medium, gpt2-large, gpt2-xl
./build/minigpt -p "Once upon a time" -n 100 --int8
```

No CMake? One command builds it (use `-march=native` instead of `-mcpu=apple-m1` on Intel/AMD):

```bash
clang++ -std=c++17 -O3 -mcpu=apple-m1 -Isrc src/main.cpp src/model.cpp src/tokenizer.cpp -o minigpt
```

| Option | Meaning |
|---|---|
| `-p, --prompt` | Text to continue |
| `-n, --tokens` | How many tokens to generate (default 64) |
| `-t, --temperature` | 0 = always the most likely token (default 0.8) |
| `-k, --top-k` | Sample from the k most likely tokens (default 40) |
| `-j, --threads` | CPU threads (default: all cores) |
| `--int8` | 8-bit weights: ~3.5× less memory, 2–2.5× faster, slightly different output |
| `--no-kv-cache` | Disable the cache, for comparison |

### Run the tests

```bash
pip install torch==2.5.1 transformers==4.46.3 safetensors numpy
mkdir -p models/test && cp models/gpt2/vocab.json models/gpt2/merges.txt models/test/
python tests/reference/make_reference.py --dir models/test   # reference outputs from PyTorch
./build/minigpt-tests models/test
```

## Limitations

- **Prompts are processed one token at a time.** A batched matrix–matrix prefill would be faster for long prompts.
- **Tokenizer Unicode classes are approximated** by Unicode block ranges instead of a full Unicode database. They are exact for ASCII and tested on mixed scripts, but rare symbols could split differently.
- **CPU only:** no GPU, and no 4-bit quantization yet.

## Credits

Model weights: [OpenAI GPT-2](https://github.com/openai/gpt-2) (MIT), downloaded from Hugging Face. Inspired by Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) and Georgi Gerganov's [llama.cpp](https://github.com/ggml-org/llama.cpp) (the Q8_0 block format). The code here is written from scratch.

## License

MIT © Animesh Sen
