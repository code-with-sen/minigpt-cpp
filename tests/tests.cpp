// Tests for minigpt. Run: minigpt-tests <fixture dir>
//
// Unit tests always run. If the fixture dir also contains model.safetensors, logits.bin and
// tokens.json (made by tests/reference/make_reference.py), the model is checked against the
// official HuggingFace implementation too.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "json.hpp"
#include "model.hpp"
#include "ops.hpp"
#include "thread_pool.hpp"
#include "tokenizer.hpp"

using namespace minigpt;

static int g_failed = 0, g_passed = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (cond) {                                                                   \
            ++g_passed;                                                               \
        } else {                                                                      \
            ++g_failed;                                                               \
            std::fprintf(stderr, "  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                             \
    } while (0)

static void section(const char* name) { std::printf("- %s\n", name); }

static bool file_exists(const std::string& p) { return std::ifstream(p).good(); }

static std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static int argmax(const std::vector<float>& v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

static void test_json() {
    section("json parser");
    auto v = json::parse(R"({"a": [1, 2.5, -3e2], "s": "x\"\\\né😀", "t": true, "n": null})");
    CHECK(v.at("a").array.size() == 3);
    CHECK(v.at("a").array[2].number == -300.0);
    CHECK(v.at("s").string == "x\"\\\n\xC3\xA9\xF0\x9F\x98\x80");
    CHECK(v.at("t").boolean);
    bool threw = false;
    try { json::parse("{\"a\": }"); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
}

static void test_thread_pool() {
    section("thread pool covers every index exactly once");
    for (int threads : {1, 2, 3, 8}) {
        ThreadPool pool(threads);
        for (size_t n : {1u, 2u, 7u, 100u, 50257u}) {
            std::vector<std::atomic<int>> hits(n);
            for (auto& h : hits) h = 0;
            for (int rep = 0; rep < 3; ++rep)
                pool.parallel_for(n, [&](size_t b, size_t e) { for (size_t i = b; i < e; ++i) ++hits[i]; });
            bool ok = std::all_of(hits.begin(), hits.end(), [](const std::atomic<int>& h) { return h == 3; });
            CHECK(ok);
        }
    }
}

static void test_ops() {
    section("math kernels");
    std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<float> g(10, 1.f), b(10, 0.f), out(10);
    layernorm(x.data(), g.data(), b.data(), out.data(), 10);
    double mean = std::accumulate(out.begin(), out.end(), 0.0) / 10;
    CHECK(std::fabs(mean) < 1e-5);

    std::vector<float> s = {1, 2, 3, 1000};
    softmax_inplace(s.data(), 4);  // must not overflow
    CHECK(std::fabs(s[0] + s[1] + s[2] + s[3] - 1.f) < 1e-6);
    CHECK(s[3] > 0.999f);

    CHECK(std::fabs(gelu(0.f)) < 1e-7);
    CHECK(std::fabs(gelu(1.f) - 0.841192f) < 1e-5);  // tanh-approximated GELU

    // int8 matvec should stay close to fp32.
    Matrix m;
    m.rows = 64;
    m.cols = 37;  // not a multiple of 8, exercises the tail loop
    m.f32.resize(m.rows * m.cols);
    for (size_t i = 0; i < m.f32.size(); ++i) m.f32[i] = std::sin(0.37f * i) * 0.1f;
    std::vector<float> in(m.cols);
    for (int i = 0; i < m.cols; ++i) in[i] = std::cos(0.11f * i);
    ThreadPool pool(2);
    std::vector<float> ref(m.rows), q(m.rows);
    QuantBuffer qbuf;
    matvec(pool, m, nullptr, in.data(), ref.data(), qbuf);
    quantize_int8(m);
    CHECK(m.quantized() && m.f32.empty());
    matvec(pool, m, nullptr, in.data(), q.data(), qbuf);
    double max_err = 0, max_ref = 0;
    for (int r = 0; r < m.rows; ++r) {
        max_err = std::max(max_err, static_cast<double>(std::fabs(ref[r] - q[r])));
        max_ref = std::max(max_ref, static_cast<double>(std::fabs(ref[r])));
    }
    CHECK(max_err < 0.01 * max_ref);
}

static void test_tokenizer(const std::string& dir) {
    section("tokenizer");
    Tokenizer tok(dir + "/vocab.json", dir + "/merges.txt");
    CHECK(tok.vocab_size() == 50257);
    CHECK(tok.eot_token() == 50256);
    CHECK((tok.encode("Hello world") == std::vector<int>{15496, 995}));
    CHECK((Tokenizer::pretokenize("a  b\n\nc") == std::vector<std::string>{"a", " ", " b", "\n", "\n", "c"}));

    // Round trip: decoding the encoding gives back exactly the input, including odd bytes.
    for (const std::string s : {"", "plain ascii", "  spaces  ", "caf\xC3\xA9 \xF0\x9F\x98\x80", "bad utf8 \xFF\xFE end",
                                "line1\nline2\r\n\ttabbed"}) {
        CHECK(tok.decode(tok.encode(s)) == s);
    }

    const std::string fixture = dir + "/tokens.json";
    if (!file_exists(fixture)) {
        std::printf("  (skipping HuggingFace comparison: %s not found)\n", fixture.c_str());
        return;
    }
    auto ref = json::parse(read_file(fixture));
    int matched = 0, total = 0;
    for (const auto& item : ref.at("tokenizer").array) {
        std::vector<int> expected;
        for (const auto& id : item.at("ids").array) expected.push_back(static_cast<int>(id.number));
        std::vector<int> got = tok.encode(item.at("text").string);
        ++total;
        if (got == expected) {
            ++matched;
        } else {
            std::fprintf(stderr, "  tokenizer mismatch for: %s\n", item.at("text").string.c_str());
        }
    }
    std::printf("  matches HuggingFace on %d/%d strings\n", matched, total);
    CHECK(matched == total);
}

static void test_model(const std::string& dir) {
    const std::string weights = dir + "/model.safetensors", logits_path = dir + "/logits.bin";
    if (!file_exists(weights) || !file_exists(logits_path) || !file_exists(dir + "/tokens.json")) {
        std::printf("- model tests skipped (run tests/reference/make_reference.py to create fixtures)\n");
        return;
    }
    auto ref = json::parse(read_file(dir + "/tokens.json"));
    std::vector<int> prompt, greedy;
    for (const auto& v : ref.at("prompt_ids").array) prompt.push_back(static_cast<int>(v.number));
    for (const auto& v : ref.at("greedy_ids").array) greedy.push_back(static_cast<int>(v.number));
    const int V = static_cast<int>(ref.at("n_vocab").number);

    std::string raw = read_file(logits_path);
    std::vector<float> expected(raw.size() / sizeof(float));
    std::memcpy(expected.data(), raw.data(), raw.size());
    CHECK(expected.size() == prompt.size() * static_cast<size_t>(V));

    section("logits match HuggingFace at every position");
    Model model(weights, {1, true, false});
    double max_err = 0;
    int top1 = 0;
    std::vector<std::vector<float>> ours;
    for (size_t t = 0; t < prompt.size(); ++t) {
        const auto& lg = model.next(prompt[t]);
        ours.push_back(lg);
        const float* e = &expected[t * V];
        for (int v = 0; v < V; ++v) max_err = std::max(max_err, static_cast<double>(std::fabs(lg[v] - e[v])));
        if (argmax(lg) == static_cast<int>(std::max_element(e, e + V) - e)) ++top1;
    }
    std::printf("  max |difference| = %.2e over %zu positions, same top token at %d/%zu\n", max_err, prompt.size(),
                top1, prompt.size());
    CHECK(max_err < 1e-3);
    CHECK(top1 == static_cast<int>(prompt.size()));

    section("greedy generation matches HuggingFace generate()");
    {
        Model m(weights, {1, true, false});
        const std::vector<float>* lg = &m.prefill(prompt);
        std::vector<int> got;
        for (size_t i = 0; i < greedy.size(); ++i) {
            got.push_back(argmax(*lg));
            lg = &m.next(got.back());
        }
        CHECK(got == greedy);
    }

    section("KV cache, threads and no-cache give identical logits");
    {
        Model nocache(weights, {1, false, false});
        Model threaded(weights, {3, true, false});
        bool same_nocache = true, same_threads = true;
        for (size_t t = 0; t < prompt.size(); ++t) {
            same_nocache &= nocache.next(prompt[t]) == ours[t];
            same_threads &= threaded.next(prompt[t]) == ours[t];
        }
        CHECK(same_nocache);
        CHECK(same_threads);
    }

    section("int8 stays close to fp32");
    {
        Model q8(weights, {2, true, true});
        Model f32(weights, {2, true, false});
        CHECK(q8.weight_bytes() < f32.weight_bytes() / 2);
        double err = 0, scale = 0;
        for (size_t t = 0; t < prompt.size(); ++t) {
            const auto& a = q8.next(prompt[t]);
            for (int v = 0; v < V; ++v) {
                err = std::max(err, static_cast<double>(std::fabs(a[v] - ours[t][v])));
                scale = std::max(scale, static_cast<double>(std::fabs(ours[t][v])));
            }
        }
        std::printf("  max |difference| = %.3f (largest logit %.3f)\n", err, scale);
        CHECK(err < 0.05 * scale);
    }

    section("errors are reported, not crashes");
    {
        Model m(weights, {1, true, false});
        bool threw = false;
        try { m.next(V); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
        threw = false;
        try { Model bad(dir + "/vocab.json", {}); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "models/test";
    try {
        test_json();
        test_thread_pool();
        test_ops();
        test_tokenizer(dir);
        test_model(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "unexpected exception: %s\n", e.what());
        return 1;
    }
    std::printf("\n%d checks passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
