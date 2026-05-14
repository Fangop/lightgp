#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <vector>

namespace lightgp {
namespace bench {

/// Run a closure (warmup discarded) several times and return median wall time in milliseconds.
inline double median_ms(int runs, const std::function<void()>& work) {
    work();  // warmup
    std::vector<double> samples;
    samples.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        work();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace bench
}  // namespace lightgp
