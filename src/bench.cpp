// minigpt-bench: measures how much each optimization speeds up generation.
//
// It runs the same greedy generation four times, adding one optimization each time, and
// prints a Markdown table. It also checks that int8 quantization barely changes the output.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>

#include "model.hpp"
#include "tokenizer.hpp"

using namespace minigpt;
using Clock = std::chrono::steady_clock;

namespace {

struct Result {
    double tokens_per_s;
    double weights_mb;
    std::vector<int> tokens;
};

int argmax(const std::vector<float>& v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

Result run(const std::string& dir, const Options& opts, const std::vector<int>& prompt, int n) {
    Model model(dir + "/model.safetensors", opts);
    Result r;
    r.weights_mb = model.weight_bytes() / 1e6;
    const std::vector<float>* logits = &model.prefill(prompt);
    auto t0 = Clock::now();
    for (int i = 0; i < n; ++i) {
        int next = argmax(*logits);
        r.tokens.push_back(next);
        logits = &model.next(next);
    }
    r.tokens_per_s = n / std::chrono::duration<double>(Clock::now() - t0).count();
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "models/gpt2";
    int n = argc > 2 ? std::stoi(argv[2]) : 32;
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

    try {
        Tokenizer tok(dir + "/vocab.json", dir + "/merges.txt");
        std::vector<int> prompt = tok.encode("In a shocking finding, scientists discovered a herd of unicorns living in a remote valley.");

        std::fprintf(stderr, "benchmarking %d generated tokens after a %zu-token prompt, %d threads available\n", n,
                     prompt.size(), threads);

        struct Step { const char* name; Options opts; };
        std::vector<Step> steps = {
            {"Baseline: 1 thread, no KV cache, fp32", {1, false, false}},
            {"+ KV cache", {1, true, false}},
            {"+ multithreading", {threads, true, false}},
            {"+ int8 weights", {threads, true, true}},
        };

        std::vector<Result> results;
        for (const auto& s : steps) {
            std::fprintf(stderr, "running: %s\n", s.name);
            results.push_back(run(dir, s.opts, prompt, n));
        }

        // How often int8 picks the same next token as fp32, fed the same (fp32) history.
        Model fp32(dir + "/model.safetensors", {threads, true, false});
        Model q8(dir + "/model.safetensors", {threads, true, true});
        std::vector<int> seq = prompt;
        seq.insert(seq.end(), results[2].tokens.begin(), results[2].tokens.end());
        int agree = 0;
        double max_err = 0;
        for (size_t i = 0; i < seq.size(); ++i) {
            const auto& a = fp32.next(seq[i]);
            const auto& b = q8.next(seq[i]);
            if (argmax(a) == argmax(b)) ++agree;
            for (size_t v = 0; v < a.size(); ++v) max_err = std::max(max_err, static_cast<double>(std::fabs(a[v] - b[v])));
        }

        const double base = results[0].tokens_per_s;
        std::printf("| Configuration | Tokens/sec | Speed-up | Weights |\n|---|---:|---:|---:|\n");
        for (size_t i = 0; i < steps.size(); ++i)
            std::printf("| %s | %.1f | %.1fx | %.0f MB |\n", steps[i].name, results[i].tokens_per_s,
                        results[i].tokens_per_s / base, results[i].weights_mb);
        std::printf("\nint8 vs fp32: same next token at %d of %zu positions (%.1f%%), largest logit difference %.3f\n",
                    agree, seq.size(), 100.0 * agree / seq.size(), max_err);
        std::printf("KV cache output identical to no-cache output: %s\n",
                    results[0].tokens == results[1].tokens ? "yes" : "NO");
        std::printf("Multithreaded output identical to single-threaded: %s\n",
                    results[1].tokens == results[2].tokens ? "yes" : "NO");
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
