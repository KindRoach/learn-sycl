#pragma once

#include <chrono>
#include <functional>
#include <iostream>

inline void print_human_readble_timeusage(long long avgDuration) {
    // Print human readable message
    if (avgDuration < 1000) {
        std::cout << avgDuration << " microseconds" << std::endl;
    } else if (avgDuration < 1000000) {
        std::cout << avgDuration / 1000.0 << " milliseconds" << std::endl;
    } else {
        std::cout << avgDuration / 1000000.0 << " seconds" << std::endl;
    }
}

inline void benchmark_func(const std::function<void()> &func, int numIterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i) {
        func(); // Call the function
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto avgDuration = totalDuration / numIterations;

    print_human_readble_timeusage(avgDuration);
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
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(actualEnd - start).count();
    auto avgDuration = totalDuration / numIterations;

    print_human_readble_timeusage(avgDuration);
}


inline void benchmark_func(const std::function<void()> &func) {
    benchmark_func(func, std::chrono::seconds(60));
}

inline std::chrono::microseconds benchmark_sycl_kernel(
    const std::function<void(sycl::queue &)> &submitKernel, sycl::queue &queue, int numIterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i) {
        submitKernel(queue); // Call the function
    }
    queue.wait();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::seconds>(end - start);
}

inline void benchmark_sycl_kernel(
    const std::function<void(sycl::queue &)> &submitKernel, sycl::queue &queue, std::chrono::seconds duration) {
    auto iter1_sec = benchmark_sycl_kernel(submitKernel, queue, 1).count() / 1000000.0;
    auto iter100_sec = benchmark_sycl_kernel(submitKernel, queue, 100).count() / 1000000.0;
    auto sec_per_iter = (iter100_sec - iter1_sec) / 99.0;

    auto numIterations = static_cast<int>(duration.count() / sec_per_iter);
    auto totalDuration = benchmark_sycl_kernel(submitKernel, queue, numIterations);
    auto avgDuration = totalDuration.count() / numIterations;

    print_human_readble_timeusage(avgDuration);
}

inline void benchmark_sycl_kernel(const std::function<void(sycl::queue &)> &submitKernel, sycl::queue &queue) {
    benchmark_sycl_kernel(submitKernel, queue, std::chrono::seconds(60));
}
