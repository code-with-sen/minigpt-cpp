// Picks the next token from the model's logits.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace minigpt {

class Sampler {
public:
    // temperature 0 means greedy (always the most likely token). top_k 0 means no limit.
    Sampler(float temperature, int top_k, unsigned seed) : temperature_(temperature), top_k_(top_k), rng_(seed) {}

    int sample(const std::vector<float>& logits) {
        const int n = static_cast<int>(logits.size());
        if (temperature_ <= 0.f) {
            return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        }
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        int k = (top_k_ > 0 && top_k_ < n) ? top_k_ : n;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        std::vector<double> probs(k);
        const double mx = logits[idx[0]];
        double sum = 0;
        for (int i = 0; i < k; ++i) {
            probs[i] = std::exp((logits[idx[i]] - mx) / temperature_);
            sum += probs[i];
        }
        std::uniform_real_distribution<double> uni(0.0, sum);
        double r = uni(rng_);
        for (int i = 0; i < k; ++i) {
            r -= probs[i];
            if (r <= 0) return idx[i];
        }
        return idx[k - 1];
    }

private:
    float temperature_;
    int top_k_;
    std::mt19937 rng_;
};

}  // namespace minigpt
