#pragma once

#include <vector>
#include <random>
#include <stdexcept>

template<typename T, typename PromotionType = double>
void random_fill(std::vector<T> &vec, T min_val = T{}, T max_val = T{100}) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<PromotionType> dist(
        static_cast<PromotionType>(min_val),
        static_cast<PromotionType>(max_val)
    );

    for (auto &elem: vec) {
        elem = static_cast<T>(dist(gen));
    }
}

template<typename T, typename AccType = double>
void acc_check(const std::vector<T> &v1, const std::vector<T> &v2) {
    if (v1.size() != v2.size()) {
        throw std::runtime_error("Vectors must have the same size.");
    }

    if constexpr (std::is_integral<T>::value) {
        bool passed = std::equal(v1.begin(), v1.end(), v2.begin());
        std::cout << "Int Acc Check " << (passed ? "SUCCESS" : "FAILURE") << "\n";
        return;
    }

    AccType maxAbsDiff = 0;
    AccType maxRelDiff = 0;
    AccType sumAbsDiff = 0;
    AccType sumRelDiff = 0;

    for (size_t i = 0; i < v1.size(); ++i) {
        auto x1 = static_cast<AccType>(v1[i]);
        auto x2 = static_cast<AccType>(v2[i]);

        AccType absDiff = std::abs(x1 - x2);
        AccType denominator = std::max(std::abs(x1), std::abs(x2));
        AccType relDiff = denominator != 0 ? absDiff / denominator : 0;

        maxAbsDiff = std::max(maxAbsDiff, absDiff);
        maxRelDiff = std::max(maxRelDiff, relDiff);

        sumAbsDiff += absDiff;
        sumRelDiff += relDiff;
    }

    AccType meanAbsError = sumAbsDiff / v1.size();
    AccType meanRelError = sumRelDiff / v1.size();
    std::cout << "Float Acc Check: "
            << "maxAbsError = " << maxAbsDiff
            << ", meanAbsError = " << meanAbsError
            << ", maxRelError = " << maxRelDiff
            << ", meanRelError = " << meanRelError << "\n";
}
