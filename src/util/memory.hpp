#pragma once

// Proxy for a row
template<typename T>
struct RowProxy {
    T *row_ptr;

    inline T &operator[](size_t j) const {
        return row_ptr[j];
    }
};

// 2D matrix view
template<typename T>
struct Matrix2D {
    T *data;

    size_t rows;
    size_t cols;

    inline RowProxy<T> operator[](size_t i) const {
        return RowProxy<T>{data + i * cols};
    }
};
