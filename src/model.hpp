// GPT-2 inference, one token at a time, on the CPU.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ops.hpp"
#include "thread_pool.hpp"

namespace minigpt {

struct Config {
    int n_vocab = 0, n_ctx = 0, n_embd = 0, n_layer = 0, n_head = 0;
    int head_dim() const { return n_embd / n_head; }
};

// Switches for the optimizations, so the benchmark can turn each one on separately.
struct Options {
    int threads = 1;
    bool kv_cache = true;
    bool int8 = false;
};

struct Layer {
    std::vector<float> ln1_g, ln1_b, ln2_g, ln2_b;
    Matrix attn_qkv, attn_proj, mlp_fc, mlp_proj;  // stored as [out][in]
    std::vector<float> attn_qkv_b, attn_proj_b, mlp_fc_b, mlp_proj_b;
};

class Model {
public:
    // Loads model.safetensors from a HuggingFace GPT-2 checkpoint (any size: small to xl).
    Model(const std::string& safetensors_path, const Options& opts);

    const Config& config() const { return cfg_; }
    const Options& options() const { return opts_; }
    size_t weight_bytes() const;

    // Clears the KV cache and starts a new sequence.
    void reset();

    // Feeds the next token of the sequence and returns the logits for the token after it.
    // With kv_cache=false, every call reprocesses the whole sequence from scratch, which is
    // what an implementation without a cache has to do.
    const std::vector<float>& next(int token);

    // Feeds a prompt; returns logits for the token after its last token.
    const std::vector<float>& prefill(const std::vector<int>& tokens);

    int position() const { return static_cast<int>(history_.size()); }

private:
    Config cfg_;
    Options opts_;
    ThreadPool pool_;

    std::vector<float> wte_;   // [n_vocab][n_embd] token embeddings (also the output head)
    std::vector<float> wpe_;   // [n_ctx][n_embd] position embeddings
    Matrix lm_head_;           // same values as wte_, possibly quantized
    std::vector<Layer> layers_;
    std::vector<float> lnf_g_, lnf_b_;

    // KV cache: per layer, [n_ctx][n_embd] keys and values.
    std::vector<std::vector<float>> k_cache_, v_cache_;
    std::vector<int> history_;

    // Scratch buffers reused for every token.
    std::vector<float> x_, h_, qkv_, attn_, proj_, fc_, logits_, scores_;
    QuantBuffer qbuf_;

    void forward_token(int token, int pos, bool want_logits);
};

}  // namespace minigpt
