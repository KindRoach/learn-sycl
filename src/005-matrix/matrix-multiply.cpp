#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include "util/bench.hpp"
#include "util/device.hpp"
#include "util/memory.hpp"
#include "util/validate.hpp"
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

template<typename T, size_t WG_SIZE, size_t SG_SIZE>
void matrix_multiply_nd_range(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    check_divisible(m, WG_SIZE, "M must be divisible by WG_SIZE");
    check_divisible(n, WG_SIZE, "N must be divisible by WG_SIZE");

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

template<typename T, size_t WG_SIZE, size_t SG_SIZE, size_t WI_SIZE>
void matrix_multiply_nd_range_vec(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    check_divisible(m, WG_SIZE, "M must be divisible by WG_SIZE");
    check_divisible(n, WG_SIZE, "N must be divisible by WG_SIZE");
    check_divisible(k, WI_SIZE, "K must be divisible by WI_SIZE");

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

template<typename T, size_t WG_SIZE, size_t SG_SIZE>
void matrix_multiply_nd_range_slm(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    check_divisible(m, WG_SIZE, "M must be divisible by WG_SIZE");
    check_divisible(n, WG_SIZE, "N must be divisible by WG_SIZE");
    check_divisible(k, WG_SIZE, "K must be divisible by WG_SIZE");

    size_t lda = k, ldb = n, ldc = n;
    q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<T, 2> slm_a{{WG_SIZE, WG_SIZE}, cgh};
        sycl::local_accessor<T, 2> slm_b{{WG_SIZE, WG_SIZE}, cgh};

        cgh.parallel_for(
            sycl::nd_range<2>{{m, n}, {WG_SIZE, WG_SIZE}},
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                size_t i = item.get_global_id(0);
                size_t j = item.get_global_id(1);

                size_t l_i = item.get_local_id(0);
                size_t l_j = item.get_local_id(1);

                T sum = 0;
                for (size_t p = 0; p < k; p += WG_SIZE) {
                    slm_a[l_i][l_j] = mat(a, lda, i, p + l_j);
                    slm_b[l_i][l_j] = mat(b, ldb, p + l_i, j);

                    item.barrier();

                    for (size_t tile_k = 0; tile_k < WG_SIZE; tile_k++) {
                        sum += slm_a[l_i][tile_k] * slm_b[tile_k][l_j];
                    }
                    item.barrier();
                }
                mat(c, ldc, i, j) = sum;
            });
    });
}

template<typename T, size_t WG_SIZE>
void matrix_multiply_subgroup_broadcast(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    check_divisible(m, WG_SIZE, "M must be divisible by WG_SIZE");
    check_divisible(n, WG_SIZE, "N must be divisible by WG_SIZE");
    check_divisible(k, WG_SIZE, "K must be divisible by WG_SIZE");

    size_t lda = k, ldb = n, ldc = n;
    q.submit([&](sycl::handler &h) {
        h.parallel_for(
            sycl::nd_range<2>{{m, n}, {WG_SIZE, WG_SIZE}},
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WG_SIZE)]] {
                size_t i = it.get_global_id(0);
                size_t j = it.get_global_id(1);
                size_t local_j = it.get_local_id(1);

                T sum = 0;
                for (size_t t = 0; t < k; t += WG_SIZE) {
                    T a_i_tile_j = mat(a, lda, i, t + local_j);
                    for (size_t tile_k = 0; tile_k < WG_SIZE; tile_k++) {
                        T a_i_tile_k = group_broadcast(it.get_sub_group(), a_i_tile_j, tile_k);
                        sum += a_i_tile_k * mat(b, ldb, t + tile_k, j);
                    }
                }

                mat(c, ldc, i, j) = sum;
            });
    });
}


int main() {
    using dtype = float;
    constexpr uint16_t wg_size = 32;
    constexpr uint8_t sg_size = 32;
    constexpr uint8_t wi_size = 4;

    size_t secs = 10;
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
    benchmark_func_by_time(secs, [&]() {
        matrix_multiply_ref<dtype>(a, b, c, m, n, k);
    });

    using func_t = std::function<void(sycl::queue &, dtype *, dtype *, dtype *, size_t, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {"matrix_multiply_mkl", matrix_multiply_mkl<dtype>},
        {"matrix_multiply_naive", matrix_multiply_naive<dtype>},
        {"matrix_multiply_nd_range", matrix_multiply_nd_range<dtype, wg_size, sg_size>},
        {"matrix_multiply_nd_range_vec", matrix_multiply_nd_range_vec<dtype, wg_size, sg_size, wi_size>},
        {"matrix_multiply_nd_range_slm", matrix_multiply_nd_range_slm<dtype, wg_size, sg_size>},
        {"matrix_multiply_subgroup_broadcast", matrix_multiply_subgroup_broadcast<dtype, wg_size>},
    };

    for (auto [func_name,func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        q.fill(d_c, dtype{0}, c.size()).wait();
        benchmark_func_by_time(secs, [&]() {
            func(q, d_a, d_b, d_c, m, n, k);
            q.wait();
        });
        acc_check(q, c, d_c);
    }
}
