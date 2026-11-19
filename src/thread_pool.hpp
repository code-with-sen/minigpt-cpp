// A small persistent thread pool with a single primitive: parallel_for.
//
// Workers are started once and reused for every call, because a token step runs ~50 matrix
// multiplies and spawning threads for each one would cost more than the work itself.
// Work is split into contiguous chunks so each thread streams through its own rows of weights.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace minigpt {

class ThreadPool {
public:
    explicit ThreadPool(int threads) : n_threads_(std::max(1, threads)) {
        for (int t = 1; t < n_threads_; ++t) workers_.emplace_back([this, t] { worker_loop(t); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
            ++generation_;
        }
        cv_start_.notify_all();
        for (auto& w : workers_) w.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return n_threads_; }

    // Calls fn(begin, end) over disjoint chunks covering [0, n). The calling thread does a share
    // of the work too, and the call returns only when every chunk is done.
    void parallel_for(size_t n, const std::function<void(size_t, size_t)>& fn) {
        if (n == 0) return;
        if (n_threads_ == 1 || n < 2) {
            fn(0, n);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            task_ = &fn;
            task_n_ = n;
            pending_ = n_threads_ - 1;
            ++generation_;
        }
        cv_start_.notify_all();
        run_chunk(0);
        std::unique_lock<std::mutex> lock(mu_);
        cv_done_.wait(lock, [this] { return pending_ == 0; });
        task_ = nullptr;
    }

private:
    int n_threads_;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_start_, cv_done_;
    const std::function<void(size_t, size_t)>* task_ = nullptr;
    size_t task_n_ = 0;
    int pending_ = 0;
    unsigned long generation_ = 0;
    bool stop_ = false;

    void run_chunk(int t) {
        size_t n = task_n_;
        size_t chunk = (n + n_threads_ - 1) / n_threads_;
        size_t begin = std::min(n, chunk * t);
        size_t end = std::min(n, begin + chunk);
        if (begin < end) (*task_)(begin, end);
    }

    void worker_loop(int t) {
        unsigned long seen = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_start_.wait(lock, [&] { return generation_ != seen; });
                seen = generation_;
                if (stop_) return;
            }
            run_chunk(t);
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (--pending_ == 0) cv_done_.notify_one();
            }
        }
    }
};

}  // namespace minigpt
