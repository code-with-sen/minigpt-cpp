#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "safetensors.hpp"

namespace minigpt {
namespace {

// HuggingFace's GPT-2 uses Conv1D layers whose weights are stored as [in][out]. We transpose
// them to [out][in] so each output is a dot product over one contiguous row.
Matrix load_linear(const SafeTensors& st, const std::string& name) {
    const TensorInfo& info = st.info(name);
    if (info.shape.size() != 2) throw std::runtime_error(name + " is not a matrix");
    const int in = static_cast<int>(info.shape[0]), out = static_cast<int>(info.shape[1]);
    std::vector<float> w = st.read_f32(name);
    Matrix m;
    m.rows = out;
    m.cols = in;
    m.f32.resize(w.size());
    for (int i = 0; i < in; ++i)
        for (int o = 0; o < out; ++o) m.f32[static_cast<size_t>(o) * in + i] = w[static_cast<size_t>(i) * out + o];
    return m;
}

}  // namespace

Model::Model(const std::string& path, const Options& opts) : opts_(opts), pool_(opts.threads) {
    SafeTensors st(path);
    // Some checkpoints prefix every name with "transformer."; accept both layouts.
    const std::string p = st.has("wte.weight") ? "" : "transformer.";

    const TensorInfo& wte = st.info(p + "wte.weight");
    const TensorInfo& wpe = st.info(p + "wpe.weight");
    cfg_.n_vocab = static_cast<int>(wte.shape[0]);
    cfg_.n_embd = static_cast<int>(wte.shape[1]);
    cfg_.n_ctx = static_cast<int>(wpe.shape[0]);
    while (st.has(p + "h." + std::to_string(cfg_.n_layer) + ".ln_1.weight")) ++cfg_.n_layer;
    // Every released GPT-2 size uses 64-dimensional attention heads.
    cfg_.n_head = cfg_.n_embd / 64;
    if (cfg_.n_layer == 0 || cfg_.n_head == 0 || cfg_.n_embd % 64 != 0)
        throw std::runtime_error("this does not look like a GPT-2 checkpoint");

    wte_ = st.read_f32(p + "wte.weight");
    wpe_ = st.read_f32(p + "wpe.weight");
    lnf_g_ = st.read_f32(p + "ln_f.weight");
    lnf_b_ = st.read_f32(p + "ln_f.bias");

    for (int l = 0; l < cfg_.n_layer; ++l) {
        const std::string b = p + "h." + std::to_string(l) + ".";
        Layer L;
        L.ln1_g = st.read_f32(b + "ln_1.weight");
        L.ln1_b = st.read_f32(b + "ln_1.bias");
        L.ln2_g = st.read_f32(b + "ln_2.weight");
        L.ln2_b = st.read_f32(b + "ln_2.bias");
        L.attn_qkv = load_linear(st, b + "attn.c_attn.weight");
        L.attn_qkv_b = st.read_f32(b + "attn.c_attn.bias");
        L.attn_proj = load_linear(st, b + "attn.c_proj.weight");
        L.attn_proj_b = st.read_f32(b + "attn.c_proj.bias");
        L.mlp_fc = load_linear(st, b + "mlp.c_fc.weight");
        L.mlp_fc_b = st.read_f32(b + "mlp.c_fc.bias");
        L.mlp_proj = load_linear(st, b + "mlp.c_proj.weight");
        L.mlp_proj_b = st.read_f32(b + "mlp.c_proj.bias");
        if (opts_.int8) {
            quantize_int8(L.attn_qkv);
            quantize_int8(L.attn_proj);
            quantize_int8(L.mlp_fc);
            quantize_int8(L.mlp_proj);
        }
        layers_.push_back(std::move(L));
    }

    // The output head reuses the token embeddings (weight tying), already laid out [vocab][embd].
    lm_head_.rows = cfg_.n_vocab;
    lm_head_.cols = cfg_.n_embd;
    if (opts_.int8) {
        // One int8 copy serves both the input embedding lookup and the output head, so the
        // fp32 table can be dropped entirely.
        lm_head_.f32 = std::move(wte_);
        quantize_int8(lm_head_);
        std::vector<float>().swap(wte_);
    }

    const int C = cfg_.n_embd;
    k_cache_.assign(cfg_.n_layer, std::vector<float>(static_cast<size_t>(cfg_.n_ctx) * C));
    v_cache_.assign(cfg_.n_layer, std::vector<float>(static_cast<size_t>(cfg_.n_ctx) * C));
    x_.resize(C); h_.resize(C); qkv_.resize(3 * C); attn_.resize(C); proj_.resize(C);
    fc_.resize(4 * C); logits_.resize(cfg_.n_vocab); scores_.resize(static_cast<size_t>(cfg_.n_head) * cfg_.n_ctx);
}

size_t Model::weight_bytes() const {
    size_t total = (wte_.size() + wpe_.size() + lnf_g_.size() + lnf_b_.size()) * sizeof(float) + lm_head_.bytes();
    for (const auto& L : layers_) {
        total += L.attn_qkv.bytes() + L.attn_proj.bytes() + L.mlp_fc.bytes() + L.mlp_proj.bytes();
        total += (L.ln1_g.size() + L.ln1_b.size() + L.ln2_g.size() + L.ln2_b.size() + L.attn_qkv_b.size() +
                  L.attn_proj_b.size() + L.mlp_fc_b.size() + L.mlp_proj_b.size()) *
                 sizeof(float);
    }
    return total;
}

void Model::reset() { history_.clear(); }

const std::vector<float>& Model::prefill(const std::vector<int>& tokens) {
    if (tokens.empty()) throw std::runtime_error("prefill needs at least one token");
    if (!opts_.kv_cache) {
        history_.assign(tokens.begin(), tokens.end() - 1);
        return next(tokens.back());
    }
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (position() >= cfg_.n_ctx) throw std::runtime_error("sequence longer than the context window");
        forward_token(tokens[i], position(), i + 1 == tokens.size());
        history_.push_back(tokens[i]);
    }
    return logits_;
}

const std::vector<float>& Model::next(int token) {
    if (token < 0 || token >= cfg_.n_vocab) throw std::runtime_error("token id out of range");
    if (position() >= cfg_.n_ctx) throw std::runtime_error("sequence longer than the context window");
    if (opts_.kv_cache) {
        forward_token(token, position(), true);
    } else {
        // No cache: rebuild every earlier position's keys and values before the new token.
        for (int pos = 0; pos < position(); ++pos) forward_token(history_[pos], pos, false);
        forward_token(token, position(), true);
    }
    history_.push_back(token);
    return logits_;
}

void Model::forward_token(int token, int pos, bool want_logits) {
    const int C = cfg_.n_embd, H = cfg_.n_head, D = cfg_.head_dim();
    const float* pe = &wpe_[static_cast<size_t>(pos) * C];
    if (lm_head_.quantized()) {
        lm_head_.dequantize_row(token, x_.data());
        for (int i = 0; i < C; ++i) x_[i] += pe[i];
    } else {
        const float* te = &wte_[static_cast<size_t>(token) * C];
        for (int i = 0; i < C; ++i) x_[i] = te[i] + pe[i];
    }

    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(D));
    for (int l = 0; l < cfg_.n_layer; ++l) {
        Layer& L = layers_[l];

        // --- Attention ---
        layernorm(x_.data(), L.ln1_g.data(), L.ln1_b.data(), h_.data(), C);
        matvec(pool_, L.attn_qkv, L.attn_qkv_b.data(), h_.data(), qkv_.data(), qbuf_);
        float* kc = k_cache_[l].data();
        float* vc = v_cache_[l].data();
        std::copy(qkv_.begin() + C, qkv_.begin() + 2 * C, kc + static_cast<size_t>(pos) * C);
        std::copy(qkv_.begin() + 2 * C, qkv_.begin() + 3 * C, vc + static_cast<size_t>(pos) * C);

        // Each head attends over positions 0..pos (the causal mask is implicit: later
        // positions simply are not in the cache yet). Heads are independent, so run them in parallel.
        const int T = pos + 1;
        pool_.parallel_for(H, [&](size_t hb, size_t he) {
            for (size_t h = hb; h < he; ++h) {
                const float* q = &qkv_[h * D];
                float* sc = &scores_[h * cfg_.n_ctx];
                for (int t = 0; t < T; ++t) sc[t] = dot_f32(q, kc + static_cast<size_t>(t) * C + h * D, D) * attn_scale;
                softmax_inplace(sc, T);
                float* o = &attn_[h * D];
                std::fill(o, o + D, 0.f);
                for (int t = 0; t < T; ++t) {
                    const float* v = vc + static_cast<size_t>(t) * C + h * D;
                    const float p = sc[t];
                    for (int d = 0; d < D; ++d) o[d] += p * v[d];
                }
            }
        });
        matvec(pool_, L.attn_proj, L.attn_proj_b.data(), attn_.data(), proj_.data(), qbuf_);
        for (int i = 0; i < C; ++i) x_[i] += proj_[i];

        // --- Feed-forward (MLP) ---
        layernorm(x_.data(), L.ln2_g.data(), L.ln2_b.data(), h_.data(), C);
        matvec(pool_, L.mlp_fc, L.mlp_fc_b.data(), h_.data(), fc_.data(), qbuf_);
        for (auto& v : fc_) v = gelu(v);
        matvec(pool_, L.mlp_proj, L.mlp_proj_b.data(), fc_.data(), proj_.data(), qbuf_);
        for (int i = 0; i < C; ++i) x_[i] += proj_[i];
    }

    if (!want_logits) return;
    layernorm(x_.data(), lnf_g_.data(), lnf_b_.data(), h_.data(), C);
    if (lm_head_.quantized()) {
        matvec(pool_, lm_head_, nullptr, h_.data(), logits_.data(), qbuf_);
    } else {
        // fp32 head: dot products directly against the embedding table (no second copy).
        pool_.parallel_for(cfg_.n_vocab, [&](size_t b, size_t e) {
            for (size_t v = b; v < e; ++v) logits_[v] = dot_f32(h_.data(), &wte_[v * C], C);
        });
    }
}

}  // namespace minigpt
