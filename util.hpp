#pragma once

#include <chrono>
#include <functional>
#include <iostream>

inline void measureExecutionTime(const std::function<void()> &func, int numIterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i) {
        func(); // Call the function
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto avgDuration = totalDuration / numIterations;

    // Print human readable message
    if (avgDuration < 1000) {
        std::cout << avgDuration << " microseconds" << std::endl;
    } else if (avgDuration < 1000000) {
        std::cout << avgDuration / 1000.0 << " milliseconds" << std::endl;
    } else {
        std::cout << avgDuration / 1000000.0 << " seconds" << std::endl;
    }
}

inline void measureExecutionTime(const std::function<void()> &func, std::chrono::seconds duration) {
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

    // Print human readable message
    if (avgDuration < 1000) {
        std::cout << avgDuration << " microseconds" << std::endl;
    } else if (avgDuration < 1000000) {
        std::cout << avgDuration / 1000.0 << " milliseconds" << std::endl;
    } else {
        std::cout << avgDuration / 1000000.0 << " seconds" << std::endl;
    }
}


inline void measureExecutionTime(const std::function<void()> &func) {
    measureExecutionTime(func, std::chrono::seconds(60));
}
