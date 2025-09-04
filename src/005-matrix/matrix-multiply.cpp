#include <sycl/sycl.hpp>

#include "util/bench.hpp"
#include "util/device.hpp"
#include "util/vector.hpp"

// A : [m,k] in row-major
// B : [k,n] in col-major
// C = A x B : [m,n] in row-major

template<typename T>
void matrix_multiply_ref(
    const std::vector<T> &a,
    const std::vector<T> &b,
    std::vector<T> &c,
    size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            T sum = 0;
            for (size_t p = 0; p < k; p++) {
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
}

template<typename T>
void matrix_multiply_naive(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    q.parallel_for({m, n}, [=](sycl::id<2> idx) {
        T sum = 0;
        size_t i = idx[0];
        size_t j = idx[1];
        for (size_t p = 0; p < k; p++) {
            sum += a[i * k + p] * b[j * k + p];
        }
        c[i * n + j] = sum;
    });
}

template<
    typename T,
    uint16_t WG_SIZE
>
void matrix_multiply_nd_range(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    q.parallel_for(
        sycl::nd_range<2>{{m, n}, {WG_SIZE, WG_SIZE}},
        [=](sycl::nd_item<2> item) {
            T sum = 0;
            size_t i = item.get_global_id(0);
            size_t j = item.get_global_id(1);
            for (size_t p = 0; p < k; p++) {
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        });
}

template<
    typename T,
    uint16_t WG_SIZE,
    uint8_t WI_SIZE
>
void matrix_multiply_nd_range_vec(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n, size_t k) {
    q.parallel_for(
        sycl::nd_range<2>{{m, n}, {WG_SIZE, WG_SIZE}},
        [=](sycl::nd_item<2> item) {
            size_t i = item.get_global_id(0);
            size_t j = item.get_global_id(1);
            sycl::vec<T, WI_SIZE> vec_a, vec_b, vec_c{0};
            for (size_t p = 0; p < k; p += WI_SIZE) {
                vec_a.load(0, a + i * k + p);
                vec_b.load(0, b + j * k + p);
                vec_c += vec_a * vec_b;
            }

            T sum = 0;
            for (int p = 0; p < WI_SIZE; ++p) {
                sum += vec_c[p];
            }
            c[i * n + j] = sum;
        });
}


int main() {
    using dtype = float;
    size_t loop = 1000;
    size_t m = 1024, n = 1024, k = 1024; // 2G Flops

    std::vector<dtype> a(m * k), b(k * n), c(m * n);
    random_fill(a);
    random_fill(b);

    std::cout << "matrix_multiply_ref:\n";
    benchmark_func(10, [&]() {
        matrix_multiply_ref(a, b, c, m, n, k);
    });

    sycl::queue q{gpu_selector_by_cu, sycl::property::queue::in_order()};
    auto *d_a = sycl::malloc_device<dtype>(a.size(), q);
    auto *d_b = sycl::malloc_device<dtype>(b.size(), q);
    auto *d_c = sycl::malloc_device<dtype>(c.size(), q);
    q.memcpy(d_a, a.data(), a.size() * sizeof(dtype)).wait();
    q.memcpy(d_b, b.data(), b.size() * sizeof(dtype)).wait();

    using func_t = std::function<void(sycl::queue &, dtype *, dtype *, dtype *, size_t, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {
            "matrix_multiply_naive",
            matrix_multiply_naive<dtype>
        },
        {
            "matrix_multiply_nd_range",
            matrix_multiply_nd_range<dtype, 32>
        },
        {
            "matrix_multiply_nd_range_vec",
            matrix_multiply_nd_range_vec<dtype, 32, 4>
        }
    };

    for (auto [func_name,func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        q.fill(d_c, 0, c.size()).wait();
        benchmark_sycl_kernel(loop, q, [&](sycl::queue &q) {
            func(q, d_a, d_b, d_c, m, n, k);
        });
        acc_check(q, c, d_c);
    }
}
