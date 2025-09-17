#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <sycl/sycl.hpp>

void print_human_readable_timeusage(int numIterations, std::chrono::microseconds totalDuration) {
    double throughput = numIterations / (totalDuration.count() / 1000000.0);
    double avgDurationMicroSec = totalDuration.count() / static_cast<double>(numIterations);
    if (avgDurationMicroSec < 1000) {
        std::cout << throughput << " iter/s @ avg: " << avgDurationMicroSec << " us" << "\n";
    } else if (avgDurationMicroSec < 1000000) {
        std::cout << throughput << " iter/s @ avg: " << avgDurationMicroSec / 1000.0 << " ms" << "\n";
    } else {
        std::cout << throughput << " iter/s @ avg: " << avgDurationMicroSec / 1000000.0 << " s" << "\n";
    }
}

void benchmark_func(
    int num_iter,
    const std::function<void()> &func,
    double warmup_ratio = 0.1
) {
    if (num_iter <= 1) {
        std::cerr << "Warning: num_iter less than 2, running func once.\n";
        func();
        return;
    }

    // Warm-up phase
    int warm_up_iter = std::max(1, static_cast<int>(num_iter * warmup_ratio));
    for (int i = 0; i < warm_up_iter; ++i) {
        func();
    }

    // benchmark phase
    num_iter -= warm_up_iter;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iter; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    print_human_readable_timeusage(num_iter, totalDuration);
}

using sycl_kernel = std::function<void(sycl::queue &)>;

void benchmark_sycl_kernel(
    int num_iter, sycl::queue &queue,
    const sycl_kernel &submitKernel,
    double warmup_ratio = 0.1
) {
    if (num_iter <= 1) {
        std::cerr << "Warning: num_iter less than 2, running kernel once.\n";
        submitKernel(queue);
        queue.wait();
        return;
    }

    // Warm-up phase
    int warm_up_iter = std::max(1, static_cast<int>(num_iter * warmup_ratio));
    for (int i = 0; i < warm_up_iter; ++i) { submitKernel(queue); }
    queue.wait();

    // benchmark phase
    num_iter -= warm_up_iter;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iter; ++i) {
        submitKernel(queue);
    }
    queue.wait();
    auto end = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    print_human_readable_timeusage(num_iter, totalDuration);
}
