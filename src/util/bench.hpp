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

void benchmark_func(int num_iter, const std::function<void()> &func) {
    int warm_up_iter = 10;
    for (int i = 0; i < warm_up_iter; ++i) {
        func(); // Call the function
    }

    int benchmark_iter = std::max(num_iter - warm_up_iter, 1);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_iter; ++i) {
        func(); // Call the function
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    print_human_readable_timeusage(num_iter, totalDuration);
}

using sycl_kernel = std::function<void(sycl::queue &)>;

void benchmark_sycl_kernel(int numIterations, sycl::queue &queue, const sycl_kernel &submitKernel) {
    if (numIterations == 0) {
        submitKernel(queue);
        queue.wait();
        return;
    }

    int warm_up_iter = 10;
    for (int i = 0; i < warm_up_iter; ++i) {
        submitKernel(queue); // Call the function
    }
    queue.wait();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i) {
        submitKernel(queue); // Call the function
    }
    queue.wait();
    auto end = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    print_human_readable_timeusage(numIterations, totalDuration);
}
