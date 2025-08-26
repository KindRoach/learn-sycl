#include <vector>
#include <random>
#include <stdexcept>

template<typename T>
void random_fill(std::vector<T> &vec, T min_val = T{}, T max_val = T{100}) {
    static_assert(std::is_arithmetic<T>::value, "random_fill only supports arithmetic types.");

    std::random_device rd;
    std::mt19937 gen(rd());

    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(min_val, max_val);
        for (auto &elem: vec) {
            elem = dist(gen);
        }
    } else if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        for (auto &elem: vec) {
            elem = dist(gen);
        }
    } else {
        throw std::runtime_error("Unsupported type for random fill.");
    }
}

template<typename T>
void acc_check(const std::vector<T> &v1, const std::vector<T> &v2) {
    if constexpr (std::is_integral<T>::value) {
        bool passed = true;
        for (size_t i = 0; i < v1.size(); ++i) {
            if (v1[i] != v2[i]) {
                passed = false;
                break;
            }
        }

        std::cout << "Int Acc Check " << (passed ? "SUCCESS" : "FAILURE") << "\n";
    } else if constexpr (std::is_floating_point<T>::value) {
        T maxAbsDiff = 0;
        T maxRelDiff = 0;
        for (size_t i = 0; i < v1.size(); ++i) {
            float absDiff = std::abs(v1[i] - v2[i]);
            float denominator = std::max(std::abs(v1[i]), std::abs(v2[i]));
            float relDiff = absDiff / denominator;

            maxAbsDiff = std::max(maxAbsDiff, absDiff);
            maxRelDiff = std::max(maxRelDiff, relDiff);
        }

        std::cout << "Float Acc Check: maxAbsError = " << maxAbsDiff << ", maxRelError = " << maxRelDiff << "\n";
    } else {
        throw std::runtime_error("Unsupported type for acc_check.");
    }
}
