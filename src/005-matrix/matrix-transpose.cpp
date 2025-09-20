#include <sycl/sycl.hpp>

#include "util/bench.hpp"
#include "util/device.hpp"
#include "util/memory.hpp"
#include "util/validate.hpp"
#include "util/vector.hpp"

// In  : [m,n] in row-major
// Out : [n,m] in row-major


template<typename T>
void matrix_transpose_ref(std::vector<T> &in, std::vector<T> &out, size_t m, size_t n) {
    size_t ld_in = n, ld_out = m;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            mat(out.data(), ld_out, j, i) = mat(in.data(), ld_in, i, j);
        }
    }
}

template<typename T>
void matrix_transpose_naive_read_continue(sycl::queue &q, T *in, T *out, size_t m, size_t n) {
    size_t ld_in = n, ld_out = m;
    q.parallel_for({m, n}, [=](sycl::id<2> idx) {
        size_t i = idx[0];
        size_t j = idx[1];
        mat(out, ld_out, j, i) = mat(in, ld_in, i, j);
    });
}

template<typename T>
void matrix_transpose_naive_write_continue(sycl::queue &q, T *in, T *out, size_t m, size_t n) {
    size_t ld_in = n, ld_out = m;
    q.parallel_for({n, m}, [=](sycl::id<2> idx) {
        size_t i = idx[0];
        size_t j = idx[1];
        mat(out, ld_out, i, j) = mat(in, ld_in, j, i);
    });
}

template<typename T, size_t WG_SIZE, size_t SG_SIZE>
void matrix_transpose_nd_range_read_continue(sycl::queue &q, T *in, T *out, size_t m, size_t n) {
    check_divisible(m, WG_SIZE, "M must be divisible by WG_SIZE");
    check_divisible(n, WG_SIZE, "N must be divisible by WG_SIZE");

    size_t ld_in = n, ld_out = m;
    q.parallel_for(
        sycl::nd_range<2>{{m, n}, {WG_SIZE, WG_SIZE}},
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t i = item.get_global_id(0);
            size_t j = item.get_global_id(1);
            mat(out, ld_out, j, i) = mat(in, ld_in, i, j);
        });
}

template<typename T, size_t WG_SIZE, size_t SG_SIZE>
void matrix_transpose_nd_range_write_continue(sycl::queue &q, T *in, T *out, size_t m, size_t n) {
    check_divisible(m, WG_SIZE, "M must be divisible by WG_SIZE");
    check_divisible(n, WG_SIZE, "N must be divisible by WG_SIZE");

    size_t ld_in = n, ld_out = m;
    q.parallel_for(
        sycl::nd_range<2>{{n, m}, {WG_SIZE, WG_SIZE}},
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t i = item.get_global_id(0);
            size_t j = item.get_global_id(1);
            mat(out, ld_out, i, j) = mat(in, ld_in, j, i);
        });
}

template<typename T, size_t WG_SIZE, size_t SG_SIZE, size_t WI_SIZE>
void matrix_transpose_nd_range_read_continue_vec(sycl::queue &q, T *in, T *out, size_t m, size_t n) {
    check_divisible(m, WG_SIZE, "M must be divisible by WG_SIZE");
    check_divisible(n, WG_SIZE * WI_SIZE, "N must be divisible by WG_SIZE * WI_SIZE");

    size_t ld_in = n, ld_out = m;
    q.parallel_for(
        sycl::nd_range<2>{{m, n / WI_SIZE}, {WG_SIZE, WG_SIZE}},
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t i = item.get_global_id(0);
            size_t j = item.get_global_id(1) * WI_SIZE;
            sycl::vec<T, WI_SIZE> vec;
            vec.load(0, &mat(in, ld_in, i, j));
            for (size_t k = 0; k < WI_SIZE; ++k) {
                mat(out, ld_out, j + k, i) = vec[k];
            }
        });
}

template<typename T, size_t WG_SIZE, size_t SG_SIZE, size_t WI_SIZE>
void matrix_transpose_nd_range_write_continue_vec(sycl::queue &q, T *in, T *out, size_t m, size_t n) {
    check_divisible(m, WG_SIZE, "M must be divisible by WG_SIZE");
    check_divisible(n, WG_SIZE * WI_SIZE, "N must be divisible by WG_SIZE * WI_SIZE");

    size_t ld_in = n, ld_out = m;
    q.parallel_for(
        sycl::nd_range<2>{{n, m / WI_SIZE}, {WG_SIZE, WG_SIZE}},
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t i = item.get_global_id(0);
            size_t j = item.get_global_id(1) * WI_SIZE;
            sycl::vec<T, WI_SIZE> vec;
            for (size_t k = 0; k < WI_SIZE; ++k) {
                vec[k] = mat(in, ld_in, j + k, i);
            }
            vec.store(0, &mat(out, ld_out, i, j));
        });
}

template<typename T, size_t WG_SIZE, size_t SG_SIZE, size_t WI_SIZE>
void matrix_transpose_nd_range_tile_vec(sycl::queue &q, T *in, T *out, size_t m, size_t n) {
    check_divisible(m, WG_SIZE * WI_SIZE, "M must be divisible by WG_SIZE * WI_SIZE");
    check_divisible(n, WG_SIZE * WI_SIZE, "N must be divisible by WG_SIZE * WI_SIZE");

    size_t ld_in = n, ld_out = m;
    q.parallel_for(
        sycl::nd_range<2>{{m / WI_SIZE, n / WI_SIZE}, {WG_SIZE, WG_SIZE}},
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t i = item.get_global_id(0) * WI_SIZE;
            size_t j = item.get_global_id(1) * WI_SIZE;

            sycl::vec<T, WI_SIZE> vec[WI_SIZE];
            for (size_t k = 0; k < WI_SIZE; ++k) {
                vec[k].load(0, &mat(in, ld_in, i + k, j));
            }

            // in-place transpose of WI_SIZE x WI_SIZE block
            for (size_t k_i = 0; k_i < WI_SIZE; ++k_i) {
                for (size_t k_j = k_i + 1; k_j < WI_SIZE; ++k_j) {
                    std::swap(vec[k_i][k_j], vec[k_j][k_i]);
                }
            }

            for (size_t k = 0; k < WI_SIZE; ++k) {
                vec[k].store(0, &mat(out, ld_out, j + k, i));
            }
        });
}

template<typename T, size_t WG_SIZE, size_t SG_SIZE>
void matrix_transpose_nd_range_tile_slm(sycl::queue &q, T *in, T *out, size_t m, size_t n) {
    check_divisible(m, WG_SIZE, "M must be divisible by WG_SIZE");
    check_divisible(n, WG_SIZE, "N must be divisible by WG_SIZE");

    size_t ld_in = n, ld_out = m;
    q.submit([&](sycl::handler &h) {
        sycl::local_accessor<T, 2> slm{{WG_SIZE, WG_SIZE}, h};
        h.parallel_for(
            sycl::nd_range<2>{{m, n}, {WG_SIZE, WG_SIZE}},
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                size_t i = item.get_global_id(0);
                size_t j = item.get_global_id(1);

                size_t l_i = item.get_local_id(0);
                size_t l_j = item.get_local_id(1);

                slm[l_i][l_j] = mat(in, ld_in, i, j);
                item.barrier();
                mat(out, ld_out, j, i) = slm[l_i][l_j];
            });
    });
}


int main() {
    using dtype = float;
    constexpr uint16_t wg_size = 32;
    constexpr uint8_t sg_size = 32;
    constexpr uint8_t wi_size = 4;

    size_t secs = 10;
    size_t loop = 100;
    size_t m = 20 * 1024, n = 5 * 1024; // 100M elements

    size_t size = m * n;
    std::vector<dtype> matrix(size), out(size);
    random_fill(matrix);

    std::cout << "matrix_transpose_ref:\n";
    benchmark_func_by_time(secs, [&] { matrix_transpose_ref(matrix, out, m, n); });

    sycl::queue q{gpu_selector_by_cu, sycl::property::queue::in_order()};
    auto *p_matrix = sycl::malloc_device<dtype>(size, q);
    auto *p_out = sycl::malloc_device<dtype>(size, q);
    q.memcpy(p_matrix, matrix.data(), size * sizeof(dtype)).wait();

    using func_t = std::function<void(sycl::queue &, dtype *, dtype *, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {
            "matrix_transpose_naive_read_continue",
            matrix_transpose_naive_read_continue<dtype>
        },
        {
            "matrix_transpose_naive_write_continue",
            matrix_transpose_naive_write_continue<dtype>
        },
        {
            "matrix_transpose_nd_range_read_continue",
            matrix_transpose_nd_range_read_continue<dtype, wg_size, sg_size>
        },
        {
            "matrix_transpose_nd_range_write_continue",
            matrix_transpose_nd_range_write_continue<dtype, wg_size, sg_size>
        },
        {
            "matrix_transpose_nd_range_read_continue_vec",
            matrix_transpose_nd_range_read_continue_vec<dtype, wg_size, sg_size, wi_size>
        },
        {
            "matrix_transpose_nd_range_write_continue_vec",
            matrix_transpose_nd_range_write_continue_vec<dtype, wg_size, sg_size, wi_size>
        },
        {
            "matrix_transpose_nd_range_tile_vec",
            matrix_transpose_nd_range_tile_vec<dtype, wg_size, sg_size, wi_size>
        },
        {
            "matrix_transpose_nd_range_tile_slm",
            matrix_transpose_nd_range_tile_slm<dtype, wg_size, sg_size>
        },
    };

    for (auto [func_name,func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        q.fill(p_out, dtype{0}, size).wait();
        benchmark_func_by_time(secs, [&]() {
            func(q, p_matrix, p_out, m, n);
            q.wait();
        });
        acc_check(q, out, p_out);
    }
}
