#pragma once

#include <vector>
#include <random>
#include <stdexcept>

template<typename T>
void random_fill(std::vector<T> &vec, T min_val = T{}, T max_val = T{100}) {
    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr (std::is_integral<T>::value) {
        using dist_type = int64_t;
        std::uniform_int_distribution<dist_type> dist(static_cast<dist_type>(min_val), static_cast<dist_type>(max_val));
        for (auto &elem: vec) { elem = static_cast<T>(dist(gen)); }
    } else if constexpr (std::is_floating_point<T>::value || std::is_same_v<T, sycl::half>) {
        using dist_type = double;
        std::uniform_real_distribution<dist_type>
                dist(static_cast<dist_type>(min_val), static_cast<dist_type>(max_val));
        for (auto &elem: vec) { elem = static_cast<T>(dist(gen)); }
    } else { static_assert(0, "Unsupported type for random fill."); }
}

template<typename T>
void acc_check(const std::vector<T> &v1, const std::vector<T> &v2) {
    if (v1.size() != v2.size()) { throw std::runtime_error("Vectors must have the same size."); }
    if constexpr (std::is_integral<T>::value) {
        bool passed = std::equal(v1.begin(), v1.end(), v2.begin());
        std::cout << "Int Acc Check " << (passed ? "SUCCESS" : "FAILURE") << "\n";
    } else if constexpr (std::is_floating_point<T>::value || std::is_same_v<T, sycl::half>) {
        using acc_type = double;
        acc_type maxAbsDiff = 0;
        acc_type maxRelDiff = 0;
        acc_type sumAbsDiff = 0;
        acc_type sumRelDiff = 0;

        for (size_t i = 0; i < v1.size(); ++i) {
            acc_type x1 = static_cast<acc_type>(v1[i]);
            acc_type x2 = static_cast<acc_type>(v2[i]);

            acc_type absDiff = std::abs(x1 - x2);
            acc_type denominator = std::max(std::abs(x1), std::abs(x2));
            acc_type relDiff = denominator != 0 ? absDiff / denominator : 0;

            maxAbsDiff = std::max(maxAbsDiff, absDiff);
            maxRelDiff = std::max(maxRelDiff, relDiff);

            sumAbsDiff += absDiff;
            sumRelDiff += relDiff;
        }
        acc_type meanAbsError = sumAbsDiff / v1.size();
        acc_type meanRelError = sumRelDiff / v1.size();
        std::cout << "Float Acc Check: " << "maxAbsError = " << maxAbsDiff << ", meanAbsError = " << meanAbsError <<
                ", maxRelError = " << maxRelDiff << ", meanRelError = " << meanRelError << "\n";
    } else { static_assert(0, "Unsupported type for acc check."); }
}

template<typename T>
void acc_check(sycl::queue &q, std::vector<T> &gt, T *device_ptr) {
    size_t size = gt.size();
    std::vector<T> actual(size);
    q.memcpy(actual.data(), device_ptr, size * sizeof(T)).wait();
    acc_check(gt, actual);
}
