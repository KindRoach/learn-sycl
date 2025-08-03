#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <sycl/sycl.hpp>

inline void print_human_readble_timeusage(double throughput, long long avgDurationMicroSec) {
    // Print human readable message
    if (avgDurationMicroSec < 1000) {
        std::cout << throughput << " iter/s @ avg: " << avgDurationMicroSec << " microseconds" << std::endl;
    } else if (avgDurationMicroSec < 1000000) {
        std::cout << throughput << " iter/s @ avg: " << avgDurationMicroSec / 1000.0 << " milliseconds" << std::endl;
    } else {
        std::cout << throughput << " iter/s @ avg: " << avgDurationMicroSec / 1000000.0 << " seconds" << std::endl;
    }
}

inline void benchmark_func(const std::function<void()> &func, int numIterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i) {
        func(); // Call the function
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    auto throughput = numIterations / (totalDuration.count() / 1000000.0);
    auto avgDuration = totalDuration.count() / numIterations;
    print_human_readble_timeusage(throughput, avgDuration);
}

inline void benchmark_func(const std::function<void()> &func, std::chrono::seconds duration) {
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start + duration;
    int numIterations = 0;

    while (std::chrono::high_resolution_clock::now() < end) {
        func(); // Call the function
        ++numIterations;
    }

    auto actualEnd = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(actualEnd - start);

    auto throughput = numIterations / (totalDuration.count() / 1000000.0);
    auto avgDuration = totalDuration.count() / numIterations;
    print_human_readble_timeusage(throughput, avgDuration);
}


inline void benchmark_func(const std::function<void()> &func) {
    benchmark_func(func, std::chrono::seconds(10));
}

using sycl_kernel = std::function<void(sycl::queue &)>;

inline std::chrono::microseconds benchmark_sycl_kernel(
    const sycl_kernel &submitKernel, sycl::queue &queue, int numIterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i) {
        submitKernel(queue); // Call the function
    }
    queue.wait();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

inline void benchmark_sycl_kernel(
    const sycl_kernel &submitKernel, sycl::queue &queue, std::chrono::seconds duration) {
    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(queue.get_context());

    auto iter1_sec = benchmark_sycl_kernel(submitKernel, queue, 1).count() / 1000000.0;
    auto iter100_sec = benchmark_sycl_kernel(submitKernel, queue, 100).count() / 1000000.0;
    auto sec_per_iter = (iter100_sec - iter1_sec) / 99.0;
    auto numIterations = static_cast<int>(duration.count() / sec_per_iter);

    auto totalDuration = benchmark_sycl_kernel(submitKernel, queue, numIterations);
    auto throughput = numIterations / (totalDuration.count() / 1000000.0);
    auto avgDuration = totalDuration.count() / numIterations;
    print_human_readble_timeusage(throughput, avgDuration);
}

inline void benchmark_sycl_kernel(const sycl_kernel &submitKernel, sycl::queue &queue) {
    benchmark_sycl_kernel(submitKernel, queue, std::chrono::seconds(10));
}

inline bool floatVectorEquals(const std::vector<float> &v1, const std::vector<float> &v2, float tolerance = 1e-5f) {
    if (v1.size() != v2.size()) {
        return false;
    }

    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::abs(v1[i] - v2[i]) / v1[i] > tolerance) {
            return false;
        }
    }

    return true;
}

inline int gpu_selector_by_cu(const sycl::device &dev) {
    int priorty = 0;

    if (dev.is_gpu()) {
        unsigned int cu = dev.get_info<sycl::info::device::max_compute_units>();
        priorty += static_cast<int>(cu);
    }

    if (dev.get_backend() == sycl::backend::ext_oneapi_level_zero) {
        priorty += 1;
    }

    return priorty;
}
