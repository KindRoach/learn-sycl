#pragma once

template<typename T>
T &mat(T *data, size_t ld, size_t i, size_t j) noexcept {
    return data[i * ld + j];
}
