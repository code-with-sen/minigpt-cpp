// minigpt: generate text with GPT-2 on the CPU.
//
//   minigpt -m models/gpt2 -p "The future of AI is" -n 64
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "model.hpp"
#include "sampler.hpp"
#include "tokenizer.hpp"

using namespace minigpt;
using Clock = std::chrono::steady_clock;

static void usage() {
    std::cerr << "usage: minigpt [options]\n"
                 "  -m, --model DIR       folder with model.safetensors, vocab.json, merges.txt (default models/gpt2)\n"
                 "  -p, --prompt TEXT     text to continue (default \"The meaning of life is\")\n"
                 "  -n, --tokens N        tokens to generate (default 64)\n"
                 "  -t, --temperature T   0 = always pick the most likely token (default 0.8)\n"
                 "  -k, --top-k K         sample only from the K most likely tokens (default 40)\n"
                 "  -s, --seed S          random seed (default 42)\n"
                 "  -j, --threads N       CPU threads (default: all cores)\n"
                 "      --int8            quantize weights to 8 bits (4x smaller)\n"
                 "      --no-kv-cache     recompute the whole sequence every step (slow; for comparison)\n";
}

int main(int argc, char** argv) {
    std::string dir = "models/gpt2", prompt = "The meaning of life is";
    int n_tokens = 64, top_k = 40;
    float temperature = 0.8f;
    unsigned seed = 42;
    Options opts;
    opts.threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") dir = val();
        else if (a == "-p" || a == "--prompt") prompt = val();
        else if (a == "-n" || a == "--tokens") n_tokens = std::stoi(val());
        else if (a == "-t" || a == "--temperature") temperature = std::stof(val());
        else if (a == "-k" || a == "--top-k") top_k = std::stoi(val());
        else if (a == "-s" || a == "--seed") seed = static_cast<unsigned>(std::stoul(val()));
        else if (a == "-j" || a == "--threads") opts.threads = std::stoi(val());
        else if (a == "--int8") opts.int8 = true;
        else if (a == "--no-kv-cache") opts.kv_cache = false;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::cerr << "unknown option: " << a << "\n"; usage(); return 2; }
    }

    try {
        auto t0 = Clock::now();
        Tokenizer tok(dir + "/vocab.json", dir + "/merges.txt");
        Model model(dir + "/model.safetensors", opts);
        auto t1 = Clock::now();
        const Config& c = model.config();
        std::fprintf(stderr, "loaded GPT-2: %d layers, %d heads, %d dims, %.0f MB of weights%s, %d threads (%.1fs)\n",
                     c.n_layer, c.n_head, c.n_embd, model.weight_bytes() / 1e6, opts.int8 ? " (int8)" : "",
                     opts.threads, std::chrono::duration<double>(t1 - t0).count());

        std::vector<int> ids = tok.encode(prompt);
        if (ids.empty()) ids.push_back(tok.eot_token());
        std::cout << prompt << std::flush;

        Sampler sampler(temperature, top_k, seed);
        auto t2 = Clock::now();
        const std::vector<float>* logits = &model.prefill(ids);
        auto t3 = Clock::now();
        int generated = 0;
        for (; generated < n_tokens && model.position() < c.n_ctx; ++generated) {
            int next = sampler.sample(*logits);
            if (next == tok.eot_token()) break;
            std::cout << tok.decode_token(next) << std::flush;
            // Every generated token costs one forward pass, so tokens/s below is honest.
            logits = &model.next(next);
        }
        auto t4 = Clock::now();
        std::cout << "\n";

        double prefill_s = std::chrono::duration<double>(t3 - t2).count();
        double gen_s = std::chrono::duration<double>(t4 - t3).count();
        std::fprintf(stderr, "prompt: %zu tokens in %.2fs | generated: %d tokens in %.2fs (%.1f tokens/s)\n",
                     ids.size(), prefill_s, generated, gen_s, generated > 0 ? generated / gen_s : 0.0);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
