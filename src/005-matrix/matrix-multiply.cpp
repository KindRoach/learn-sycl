#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include "util/bench.hpp"
#include "util/device.hpp"
#include "util/memory.hpp"
#include "util/vector.hpp"

// A : [m,k] in row-major
// B : [k,n] in row-major
// C = A x B : [m,n] in row-major

template<typename T>
void matrix_multiply_ref(
    std::vector<T> &a,
    std::vector<T> &b,
    std::vector<T> &c,
    size_t m, size_t n, size_t k) {

    size_t lda = k, ldb = n, ldc = n;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            T sum = 0;
            for (size_t p = 0; p < k; p++) {
                sum += mat(a.data(), lda, i, p) * mat(b.data(), ldb, p, j);
            }
            mat(c.data(), ldc, i, j) = sum;
        }
    }
}

template<typename T>
void matrix_multiply_mkl(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    try {
        // oneMKL gemm: submits to the provided SYCL queue and returns an event.
        oneapi::mkl::blas::gemm(
            q,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            m, n, k,
            1.0f,
            a, k,
            b, n,
            0.0f,
            c, n);
    } catch (const std::exception &e) {
        // rethrow or handle as desired; here we convert to runtime_error with message.
        throw std::runtime_error(std::string("oneMKL gemm failed: ") + e.what());
    }
}

template<typename T>
void matrix_multiply_naive(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    size_t lda = k, ldb = n, ldc = n;

    q.parallel_for({m, n}, [=](sycl::id<2> idx) {
        size_t i = idx[0];
        size_t j = idx[1];
        T sum = 0;
        for (size_t p = 0; p < k; p++) {
            sum += mat(a, lda, i, p) * mat(b, ldb, p, j);
        }
        mat(c, ldc, i, j) = sum;
    });
}

template<typename T, uint16_t WG_SIZE, uint8_t SG_SIZE>
void matrix_multiply_nd_range(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    size_t lda = k, ldb = n, ldc = n;

    q.parallel_for(
        sycl::nd_range<2>{{m, n}, {WG_SIZE, WG_SIZE}},
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t i = item.get_global_id(0);
            size_t j = item.get_global_id(1);

            T sum = 0;
            for (size_t p = 0; p < k; p++) {
                sum += mat(a, lda, i, p) * mat(b, ldb, p, j);
            }
            mat(c, ldc, i, j) = sum;
        });
}

template<typename T, uint16_t WG_SIZE, uint8_t SG_SIZE, uint8_t WI_SIZE>
void matrix_multiply_nd_range_vec(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    size_t lda = k, ldb = n, ldc = n;

    q.parallel_for(
        sycl::nd_range<2>{{m, n}, {WG_SIZE, WG_SIZE}},
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t i = item.get_global_id(0);
            size_t j = item.get_global_id(1);

            sycl::vec<T, WI_SIZE> vec_a, vec_b, vec_c{0};

            for (size_t p = 0; p < k; p += WI_SIZE) {
                vec_a.load(0, &mat(a, lda, i, p));
                for (int v = 0; v < WI_SIZE; ++v) {
                    vec_b[v] = mat(b, ldb, p + v, j);
                }
                vec_c += vec_a * vec_b;
            }

            T sum = 0;
            for (int v = 0; v < WI_SIZE; ++v) {
                sum += vec_c[v];
            }
            mat(c, ldc, i, j) = sum;
        });
}

template<typename T, uint16_t WG_SIZE, uint8_t SG_SIZE>
void matrix_multiply_nd_range_slm(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    size_t lda = k, ldb = n, ldc = n;

    q.submit([&](sycl::handler &cgh) {
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
                    slm_a[x][y] = mat(a, lda, i, p + y);
                    slm_b[x][y] = mat(b, ldb, p + x, j);

                    item.barrier();

                    for (size_t tile_k = 0; tile_k < WG_SIZE; tile_k++) {
                        sum += slm_a[x][tile_k] * slm_b[tile_k][y];
                    }
                    item.barrier();
                }
                mat(c, ldc, i, j) = sum;
            });
    });
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

    std::cout << "matrix_multiply_ref:\n";
    benchmark_func(1, [&]() {
        matrix_multiply_ref<dtype>(a, b, c, m, n, k);
    });

    using func_t = std::function<void(sycl::queue &, dtype *, dtype *, dtype *, size_t, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {"matrix_multiply_mkl", matrix_multiply_mkl<dtype>},
        {"matrix_multiply_naive", matrix_multiply_naive<dtype>},
        {"matrix_multiply_nd_range", matrix_multiply_nd_range<dtype, 32, 32>},
        {"matrix_multiply_nd_range_vec", matrix_multiply_nd_range_vec<dtype, 32, 32, 4>},
        {"matrix_multiply_nd_range_slm", matrix_multiply_nd_range_slm<dtype, 32, 32>}
    };

    for (auto [func_name,func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        q.fill(d_c, dtype{0}, c.size()).wait();
        benchmark_sycl_kernel(loop, q, [&](sycl::queue &q) {
            func(q, d_a, d_b, d_c, m, n, k);
        });
        acc_check(q, c, d_c);
    }
}
