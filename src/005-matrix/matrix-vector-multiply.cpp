#include <sycl/sycl.hpp>

#include "util/util.hpp"

// A matrix: [m, n] in row-major or col-major
// b vector: [n]
// o = A x b^T vector : [m]

template<typename T, layout a_layout>
void matrix_vector_multiply_ref(std::vector<T> &a, std::vector<T> &b, std::vector<T> &c, size_t m, size_t n) {
    size_t ld = a_layout == layout::row_major ? n : m;
    for (size_t i = 0; i < m; i++) {
        T sum = 0;
        for (size_t k = 0; k < n; k++) {
            if constexpr (a_layout == layout::row_major) {
                sum += mat(a.data(), ld, i, k) * b[k];
            } else {
                sum += mat(a.data(), ld, k, i) * b[k];
            }
        }
        c[i] = sum;
    }
}

template<typename T, layout a_layout>
void matrix_vector_multiply_naive(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n) {
    size_t ld = a_layout == layout::row_major ? n : m;
    q.parallel_for(
        sycl::range<1>(m),
        [=](sycl::id<1> i) {
            T sum = 0;
            for (size_t k = 0; k < n; k++) {
                if constexpr (a_layout == layout::row_major) {
                    sum += mat(a, ld, i, k) * b[k];
                } else {
                    sum += mat(a, ld, k, i) * b[k];
                }
            }
            c[i] = sum;
        });
}

template<typename T, layout a_layout, size_t WG_SIZE, size_t SG_SIZE>
void matrix_vector_multiply_nd_range(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n) {
    check_divisible(m, WG_SIZE, "M must be divisible by WG_SIZE");

    size_t ld = a_layout == layout::row_major ? n : m;
    q.parallel_for(
        sycl::nd_range<1>{m, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            T sum = 0;
            size_t i = item.get_global_id();
            for (size_t k = 0; k < n; k++) {
                if constexpr (a_layout == layout::row_major) {
                    sum += mat(a, ld, i, k) * b[k];
                } else {
                    sum += mat(a, ld, k, i) * b[k];
                }
            }
            c[i] = sum;
        });
}

template<typename T, layout a_layout, size_t WG_SIZE, size_t SG_SIZE>
void matrix_vector_multiply_n_split_sg(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n) {
    check_divisible(m, WG_SIZE, "M must be divisible by WG_SIZE");
    check_divisible(n, SG_SIZE, "N must be divisible by SG_SIZE");

    size_t ld = a_layout == layout::row_major ? n : m;
    q.parallel_for(
        sycl::nd_range<2>{{m, SG_SIZE}, {WG_SIZE, SG_SIZE}},
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t i = item.get_global_id(0);

            auto sg = item.get_sub_group();
            size_t sg_i = sg.get_local_linear_id();

            T sum = 0;
            for (size_t k = 0; k < n; k += SG_SIZE) {
                if constexpr (a_layout == layout::row_major) {
                    sum += mat(a, ld, i, k + sg_i) * b[k + sg_i];
                } else {
                    sum += mat(a, ld, k + sg_i, i) * b[k + sg_i];
                }
            }

            T sg_sum = sycl::reduce_over_group(sg, sum, sycl::plus<>());

            if (sg_i == 0) {
                c[i] = sg_sum;
            }
        });
}

template<typename T, layout a_layout, size_t WG_SIZE, size_t SG_SIZE>
void matrix_vector_multiply_n_split_wg(sycl::queue &q, T *a, T *b, T *c, size_t m, size_t n) {
    check_divisible(n, WG_SIZE, "N must be divisible by WG_SIZE");

    size_t ld = a_layout == layout::row_major ? n : m;
    size_t ele_per_sg = n / (WG_SIZE / SG_SIZE);
    q.parallel_for(
        sycl::nd_range<2>{{m, WG_SIZE}, {1, WG_SIZE}},
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t i = item.get_global_id(0);

            auto sg = item.get_sub_group();
            size_t sg_i = sg.get_local_linear_id();
            size_t sg_group_id = sg.get_group_id();
            size_t start_id = sg_group_id * ele_per_sg;
            size_t end_id = start_id + ele_per_sg;

            T sum = 0;
            for (size_t k = start_id; k < end_id; k += SG_SIZE) {
                if constexpr (a_layout == layout::row_major) {
                    sum += mat(a, ld, i, k + sg_i) * b[k + sg_i];
                } else {
                    sum += mat(a, ld, k + sg_i, i) * b[k + sg_i];
                }
            }

            T sg_sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<>());

            if (item.get_group().leader()) {
                c[i] = sg_sum;
            }
        });
}


template<layout a_layout>
void test_matrix_multiply() {
    std::string a_major = a_layout == layout::row_major ? "row major" : "col major";
    std::cout << "-------------- matrix a in " << a_major << " --------------\n";

    using dtype = float;
    constexpr uint8_t sg_size = 32;

    size_t secs = 10;
    size_t m = 512 * 1024, n = 1024; // 1G FLOPs

    std::vector<dtype> a(m * n), b(n), c(m);
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
        matrix_vector_multiply_ref<dtype, a_layout>(a, b, c, m, n);
    });

    using func_t = std::function<void(sycl::queue &, dtype *, dtype *, dtype *, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {"matrix_vector_multiply_naive", matrix_vector_multiply_naive<dtype, a_layout>},
        {"matrix_vector_multiply_nd_range", matrix_vector_multiply_nd_range<dtype, a_layout, 256, sg_size>},
        {"matrix_vector_multiply_n_split_sg", matrix_vector_multiply_n_split_sg<dtype, a_layout, 32, sg_size>},
        {"matrix_vector_multiply_n_split_wg", matrix_vector_multiply_n_split_wg<dtype, a_layout, 256, sg_size>},
    };

    for (auto [func_name,func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        q.fill(d_c, dtype{0}, c.size()).wait();
        benchmark_func_by_time(secs, [&]() {
            func(q, d_a, d_b, d_c, m, n);
            q.wait();
        });
        sycl_acc_check(q, c, d_c);
    }
}


int main() {
    test_matrix_multiply<layout::row_major>();
    test_matrix_multiply<layout::col_major>();
}
