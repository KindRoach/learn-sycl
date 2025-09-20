#pragma once

#include <stdexcept>

template<typename T>
void check_divisible(T dividend, T divisor, const std::string &errorMsg) {
    if (divisor == 0) {
        throw std::invalid_argument("Divisor cannot be 0");
    }
    if (dividend % divisor != 0) {
        throw std::invalid_argument(errorMsg);
    }
}
