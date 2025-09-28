#pragma once

enum layout {
    row_major,
    col_major
};

template<typename T>
T &mat(T *data, size_t ld, size_t i, size_t j) noexcept {
    return data[i * ld + j];
}

template<typename T>
T *mat_ptr(T *data, size_t ld, size_t i, size_t j) noexcept {
    return &data[i * ld + j];
}


template<typename T>
void print_matrix(T *data, size_t m, size_t n) {
    std::cout << "Matrix " << m << " x " << n << ":\n";
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            std::cout << std::setw(5) << mat<T>(data, m, i, j) << " ";
        }
    }
}
