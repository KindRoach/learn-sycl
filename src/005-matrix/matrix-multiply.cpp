#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include "util/bench.hpp"
#include "util/device.hpp"
#include "util/vector.hpp"

// A : [m,k] in row-major
// B : [k,n] in row-major or col-major
// C = A x B : [m,n] in row-major

enum layout {
    row_major,
    col_major
};

template<typename T, layout b_layout>
void matrix_multiply_ref(
    const std::vector<T> &a,
    const std::vector<T> &b,
    std::vector<T> &c,
    size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            T sum = 0;
            for (size_t p = 0; p < k; p++) {
                if constexpr (b_layout == row_major) {
                    sum += a[i * k + p] * b[p * n + j];
                } else {
                    sum += a[i * k + p] * b[j * k + p];
                }
            }
            c[i * n + j] = sum;
        }
    }
}

template<typename T, layout b_layout>
void matrix_multiply_mkl(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    try {
        // oneMKL gemm: submits to the provided SYCL queue and returns an event.
        oneapi::mkl::blas::gemm(
            q,
            oneapi::mkl::transpose::nontrans,
            b_layout == row_major ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans,
            m, n, k,
            1.0f,
            a, k,
            b, b_layout == row_major ? n : k,
            0.0f,
            c, n);
    } catch (const std::exception &e) {
        // rethrow or handle as desired; here we convert to runtime_error with message.
        throw std::runtime_error(std::string("oneMKL gemm failed: ") + e.what());
    }
}

template<typename T, layout b_layout>
void matrix_multiply_naive(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    q.parallel_for({m, n}, [=](sycl::id<2> idx) {
        size_t i = idx.get(0);
        size_t j = idx.get(1);

        T sum = 0;
        for (size_t p = 0; p < k; p++) {
            if constexpr (b_layout == row_major) {
                sum += a[i * k + p] * b[p * n + j];
            } else {
                sum += a[i * k + p] * b[j * k + p];
            }
        }

        c[i * n + j] = sum;
    });
}

template<
    typename T,
    uint16_t WG_SIZE,
    uint8_t SG_SIZE,
    layout b_layout
>
void matrix_multiply_nd_range(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    q.parallel_for(
        sycl::nd_range<2>{{m, n}, {WG_SIZE, WG_SIZE}},
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t i = item.get_global_id(0);
            size_t j = item.get_global_id(1);

            T sum = 0;
            for (size_t p = 0; p < k; p++) {
                if constexpr (b_layout == row_major) {
                    sum += a[i * k + p] * b[p * n + j];
                } else {
                    sum += a[i * k + p] * b[j * k + p];
                }
            }

            c[i * n + j] = sum;
        });
}

template<
    typename T,
    uint16_t WG_SIZE,
    uint8_t SG_SIZE,
    uint8_t WI_SIZE,
    layout b_layout
>
void matrix_multiply_nd_range_vec(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    q.parallel_for(
        sycl::nd_range<2>{{m, n}, {WG_SIZE, WG_SIZE}},
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t i = item.get_global_id(0);
            size_t j = item.get_global_id(1);
            sycl::vec<T, WI_SIZE> vec_a, vec_b, vec_c{0};
            for (size_t p = 0; p < k; p += WI_SIZE) {
                vec_a.load(0, a + i * k + p);
                if constexpr (b_layout == row_major) {
                    for (int vec_i = 0; vec_i < WI_SIZE; ++vec_i) {
                        vec_b[vec_i] = b[(p + vec_i) * n + j];
                    }
                } else {
                    vec_b.load(0, b + j * k + p);
                }
                vec_c += vec_a * vec_b;
            }

            T sum = 0;
            for (int vec_i = 0; vec_i < WI_SIZE; ++vec_i) {
                sum += vec_c[vec_i];
            }
            c[i * n + j] = sum;
        });
}

template<
    typename T,
    uint16_t WG_SIZE,
    uint8_t SG_SIZE,
    layout b_layout
>
void matrix_multiply_nd_range_slm(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<T, 2> slm_a{{WG_SIZE, WG_SIZE}, cgh};
        sycl::local_accessor<T, 2> slm_b{{WG_SIZE, WG_SIZE}, cgh};
        cgh.parallel_for(
            sycl::nd_range<2>{{m, n}, {WG_SIZE, WG_SIZE}},
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                size_t i = item.get_global_id(0);
                size_t j = item.get_global_id(1);

                size_t x = item.get_local_id(0);
                size_t y = item.get_local_id(1);

                T sum = 0;
                for (size_t p = 0; p < k; p += WG_SIZE) {
                    slm_a[x][y] = a[i * k + p + y];
                    if constexpr (b_layout == row_major) {
                        slm_b[x][y] = b[(p + x) * n + j];
                    } else {
                        slm_b[x][y] = b[j * k + p + x];
                    }
                    item.barrier();

                    for (size_t tile_k = 0; tile_k < WG_SIZE; tile_k++) {
                        sum += slm_a[x][tile_k] * slm_b[tile_k][y];
                    }
                    item.barrier();
                }

                c[i * n + j] = sum;
            });
    });
}

template<typename T, layout b_layout>
void test_matrix_multiply(
    size_t loop, size_t m, size_t n, size_t k,
    const std::vector<T> &a,
    const std::vector<T> &b,
    std::vector<T> &c,
    sycl::queue &q,
    T *d_a, T *d_b, T *d_c
) {
    std::cout << "-------------- b " << (b_layout == row_major ? "row major" : "col major") << " --------------\n";

    std::cout << "matrix_multiply_ref:\n";
    benchmark_func(10, [&]() {
        matrix_multiply_ref<T, b_layout>(a, b, c, m, n, k);
    });

    using func_t = std::function<void(sycl::queue &, T *, T *, T *, size_t, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {"matrix_multiply_mkl", matrix_multiply_mkl<T, b_layout>},
        {"matrix_multiply_naive", matrix_multiply_naive<T, b_layout>},
        {"matrix_multiply_nd_range", matrix_multiply_nd_range<T, 32, 32, b_layout>},
        {"matrix_multiply_nd_range_vec", matrix_multiply_nd_range_vec<T, 32, 32, 4, b_layout>},
        {"matrix_multiply_nd_range_slm", matrix_multiply_nd_range_slm<T, 32, 32, b_layout>}
    };

    for (auto [func_name,func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        q.fill(d_c, T{0}, c.size()).wait();
        benchmark_sycl_kernel(loop, q, [&](sycl::queue &q) {
            func(q, d_a, d_b, d_c, m, n, k);
        });
        acc_check(q, c, d_c);
    }
}

int main() {
    using dtype = float;
    size_t loop = 1000;
    size_t m = 1024, n = 1024, k = 1024; // 2G FLOPs

    std::vector<dtype> a(m * k), b(k * n), c(m * n);
    random_fill(a);
    random_fill(b);

    sycl::queue q{gpu_selector_by_cu, sycl::property::queue::in_order()};
    auto *d_a = sycl::malloc_device<dtype>(a.size(), q);
    auto *d_b = sycl::malloc_device<dtype>(b.size(), q);
    auto *d_c = sycl::malloc_device<dtype>(c.size(), q);
    q.memcpy(d_a, a.data(), a.size() * sizeof(dtype)).wait();
    q.memcpy(d_b, b.data(), b.size() * sizeof(dtype)).wait();

    test_matrix_multiply<dtype, row_major>(loop, m, n, k, a, b, c, q, d_a, d_b, d_c);
    test_matrix_multiply<dtype, col_major>(loop, m, n, k, a, b, c, q, d_a, d_b, d_c);
}
