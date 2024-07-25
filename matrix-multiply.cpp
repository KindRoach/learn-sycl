#include <algorithm>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

#include "util.hpp"

using namespace sycl;

constexpr size_t N = 512;
constexpr size_t B = 16;

void matrix_multiply_naive(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

void matrix_multiply(
    queue &q,
    buffer<float, 2> &a_buf, buffer<float, 2> &b_buf, buffer<float, 2> &c_buf) {
    // Submit the kernel to the queue
    q.submit([&](handler &h) {
        accessor a{a_buf, h, read_only};
        accessor b{b_buf, h, read_only};
        accessor c{c_buf, h, write_only, no_init};

        // BEGIN CODE SNIP
        h.parallel_for(range{N, N}, [=](id<2> idx) {
            int i = idx[0];
            int j = idx[1];
            float sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        });
        // END CODE SNIP
    });
}

void matrix_multiply_nd_range(
    queue &q,
    buffer<float, 2> &a_buf, buffer<float, 2> &b_buf, buffer<float, 2> &c_buf) {
    // Submit the kernel to the queue
    q.submit([&](handler &h) {
        accessor a{a_buf, h, read_only};
        accessor b{b_buf, h, read_only};
        accessor c{c_buf, h, write_only, no_init};

        // BEGIN CODE SNIP
        range global{N, N};
        range local{B, B};
        h.parallel_for(nd_range{global, local}, [=](nd_item<2> it) {
            int i = it.get_global_id(0);
            int j = it.get_global_id(1);

            float sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        });
        // END CODE SNIP
    });
}

void matrix_multiply_nd_range_group_local_mem(
    queue &q,
    buffer<float, 2> &a_buf, buffer<float, 2> &b_buf, buffer<float, 2> &c_buf) {
    // Submit the kernel to the queue
    q.submit([&](handler &h) {
        accessor a{a_buf, h, read_only};
        accessor b{b_buf, h, read_only};
        accessor c{c_buf, h, write_only, no_init};

        auto tile = local_accessor<float, 2>({B, B}, h);

        // BEGIN CODE SNIP
        range global{N, N};
        range local{B, B};
        h.parallel_for(nd_range{global, local}, [=](nd_item<2> it) {
            int m = it.get_global_id(0);
            int n = it.get_global_id(1);

            int i = it.get_local_id(0);
            int j = it.get_local_id(1);

            float sum = 0;
            for (int t = 0; t < N; t += B) {
                // load the matrix tile from matrix A
                // each work-item read one element
                // synchronize to wait for other work-item
                tile[i][j] = a[m][t + j];
                group_barrier(it.get_group());

                // Perform computation using the local memory
                // tile, and matrix B in global memory.
                for (int k = 0; k < B; k++) {
                    sum += tile[i][k] * b[t + k][n];
                }

                // synchronize to wait for other work-item
                group_barrier(it.get_group());
            }

            c[m][n] = sum;
        });
        // END CODE SNIP
    });
}

void matrix_multiply_nd_range_group_broadcast(
    queue &q,
    buffer<float, 2> &a_buf, buffer<float, 2> &b_buf, buffer<float, 2> &c_buf) {
    // Submit the kernel to the queue
    q.submit([&](handler &h) {
        accessor a{a_buf, h, read_only};
        accessor b{b_buf, h, read_only};
        accessor c{c_buf, h, write_only, no_init};

        size_t tile_size = B;

        // BEGIN CODE SNIP
        range global{N, N};
        range local{1, tile_size};
        h.parallel_for(nd_range{global, local}, [=](nd_item<2> it) {
            // Indices in the global index space:
            int m = it.get_global_id()[0];
            int n = it.get_global_id()[1];

            // Index in the local index space:
            int i = it.get_local_id()[1];

            float sum = 0;
            for (int t = 0; t < N; t += tile_size) {
                // Load the matrix tile from matrix A.
                float tileI = a[m][t + i];

                // Perform computation by broadcasting from
                // the matrix tile and loading from matrix B
                // in global memory.  The loop variable k
                // describes which work-item in the work-group
                // to broadcast data from.
                for (int k = 0; k < tile_size; k++) {
                    sum += group_broadcast(it.get_group(), tileI, k) * b[t + k][n];
                }
            }

            // Write the final result to global memory.
            c[m][n] = sum;
        });
        // END CODE SNIP
    });
}

void matrix_multiply_nd_range_sub_group_local_mem(
    queue &q,
    buffer<float, 2> &a_buf, buffer<float, 2> &b_buf, buffer<float, 2> &c_buf) {
    // Submit the kernel to the queue
    q.submit([&](handler &h) {
        accessor a{a_buf, h, read_only};
        accessor b{b_buf, h, read_only};
        accessor c{c_buf, h, write_only, no_init};

        size_t tile_size = 4;
        auto tile = local_accessor<float, 1>(tile_size, h);

        // BEGIN CODE SNIP
        range global{N, N};
        range local{1, tile_size};
        h.parallel_for(nd_range{global, local}, [=](nd_item<2> it) {
            int m = it.get_global_id(0);
            int n = it.get_global_id(1);

            int i = it.get_local_id(1);

            float sum = 0;
            for (int t = 0; t < N; t += tile_size) {
                // load the matrix tile from matrix A
                // each work-item read one element
                // synchronize to wait for other work-item
                tile[i] = a[m][t + i];
                group_barrier(it.get_sub_group());

                // Perform computation using the local memory
                // tile, and matrix B in global memory.
                for (int k = 0; k < tile_size; k++) {
                    sum += tile[k] * b[t + k][n];
                }

                // synchronize to wait for other work-item
                group_barrier(it.get_sub_group());
            }

            c[m][n] = sum;
        });
        // END CODE SNIP
    });
}

void matrix_multiply_nd_range_sub_group_broadcast(
    queue &q,
    buffer<float, 2> &a_buf, buffer<float, 2> &b_buf, buffer<float, 2> &c_buf) {
    // Submit the kernel to the queue
    q.submit([&](handler &h) {
        accessor a{a_buf, h, read_only};
        accessor b{b_buf, h, read_only};
        accessor c{c_buf, h, write_only, no_init};

        size_t tile_size = 4;

        // BEGIN CODE SNIP
        range global{N, N};
        range local{1, tile_size};
        h.parallel_for(nd_range{global, local}, [=](nd_item<2> it) {
            // Indices in the global index space:
            int m = it.get_global_id()[0];
            int n = it.get_global_id()[1];

            // Index in the local index space:
            int i = it.get_local_id()[1];

            float sum = 0;
            for (int t = 0; t < N; t += tile_size) {
                // Load the matrix tile from matrix A.
                float tileI = a[m][t + i];

                // Perform computation by broadcasting from
                // the matrix tile and loading from matrix B
                // in global memory.  The loop variable k
                // describes which work-item in the sub-group
                // to broadcast data from.
                for (int k = 0; k < tile_size; k++) {
                    sum += group_broadcast(it.get_sub_group(), tileI, k) * b[t + k][n];
                }
            }

            // Write the final result to global memory.
            c[m][n] = sum;
        });
        // END CODE SNIP
    });
}

void test_perfomance() {
    std::vector<float> a(N * N), b(N * N), c(N * N);

    std::cout << "CPU single core: ";
    benchmark_func([&] { matrix_multiply_naive(a, b, c); });

    queue cpu_q{cpu_selector_v};
    queue gpu_q{gpu_selector_by_cu};
    std::vector<std::tuple<
        std::string,
        std::function<void(queue &, buffer<float, 2> &, buffer<float, 2> &, buffer<float, 2> &)>,
        queue> > tests = {
        {"CPU SYCL", matrix_multiply, cpu_q},
        {"CPU SYCL ND-range", matrix_multiply_nd_range, cpu_q},
        {"CPU SYCL ND-range Group Local Memory", matrix_multiply_nd_range_group_local_mem, cpu_q},
        {"CPU SYCL ND-range Group Broadcast", matrix_multiply_nd_range_group_broadcast, cpu_q},
        {"CPU SYCL ND-range Sub-group Local Memory", matrix_multiply_nd_range_sub_group_local_mem, cpu_q},
        {"CPU SYCL ND-range Sub-group Broadcast", matrix_multiply_nd_range_sub_group_broadcast, cpu_q},
        {"GPU SYCL", matrix_multiply, gpu_q},
        {"GPU SYCL ND-range", matrix_multiply_nd_range, gpu_q},
        {"GPU SYCL ND-range Group Local Memory", matrix_multiply_nd_range_group_local_mem, gpu_q},
        {"GPU SYCL ND-range Group Broadcast", matrix_multiply_nd_range_group_broadcast, gpu_q},
        {"GPU SYCL ND-range Sub-group Local Memory", matrix_multiply_nd_range_sub_group_local_mem, gpu_q},
        {"GPU SYCL ND-range Sub-group Broadcast", matrix_multiply_nd_range_sub_group_broadcast, gpu_q},
    };

    for (auto &[name,kernel, q]: tests) {
        {
            buffer<float, 2> a_buf(a.data(), range<2>(N, N)),
                    b_buf(b.data(), range<2>(N, N)),
                    c_buf(c.data(), range<2>(N, N));

            std::cout << name << ": ";
            benchmark_sycl_kernel([&](queue &q) { kernel(q, a_buf, b_buf, c_buf); }, q);
        }
    }
}

void test_acc() {
    // Initialize input and output memory on the host
    std::vector<float> a(N * N), b(N * N), c(N * N), gt(N * N);
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    auto rng = [&]() { return dist(gen); };
    std::generate(a.begin(), a.end(), rng);
    std::generate(b.begin(), b.end(), rng);

    // calculate groud truth
    matrix_multiply_naive(a, b, gt);

    queue cpu_q{cpu_selector_v};
    queue gpu_q{gpu_selector_by_cu};
    std::vector<std::tuple<
        std::string,
        std::function<void(queue &, buffer<float, 2> &, buffer<float, 2> &, buffer<float, 2> &)>,
        queue> > tests = {
        {"CPU SYCL", matrix_multiply, cpu_q},
        {"CPU SYCL ND-range", matrix_multiply_nd_range, cpu_q},
        {"CPU SYCL ND-range Group Local Memory", matrix_multiply_nd_range_group_local_mem, cpu_q},
        {"CPU SYCL ND-range Group Broadcast", matrix_multiply_nd_range_group_broadcast, cpu_q},
        {"CPU SYCL ND-range Sub-group Local Memory", matrix_multiply_nd_range_sub_group_local_mem, cpu_q},
        {"CPU SYCL ND-range Sub-group Broadcast", matrix_multiply_nd_range_sub_group_broadcast, cpu_q},
        {"GPU SYCL", matrix_multiply, gpu_q},
        {"GPU SYCL ND-range", matrix_multiply_nd_range, gpu_q},
        {"GPU SYCL ND-range Group Local Memory", matrix_multiply_nd_range_group_local_mem, gpu_q},
        {"GPU SYCL ND-range Group Broadcast", matrix_multiply_nd_range_group_broadcast, gpu_q},
        {"GPU SYCL ND-range Sub-group Local Memory", matrix_multiply_nd_range_sub_group_local_mem, gpu_q},
        {"GPU SYCL ND-range Sub-group Broadcast", matrix_multiply_nd_range_sub_group_broadcast, gpu_q},
    };

    for (auto &[name,kernel, q]: tests) {
        {
            std::fill(c.begin(), c.end(), 0);
            buffer<float, 2> a_buf(a.data(), range<2>(N, N)),
                    b_buf(b.data(), range<2>(N, N)),
                    c_buf(c.data(), range<2>(N, N));

            kernel(q, a_buf, b_buf, c_buf);
        }

        auto success = floatVectorEquals(c, gt);
        std::cout << name << ": " << (success ? "SUCCESS" : "FAILURE") << std::endl;
    }
}


int main() {
    test_acc();
    test_perfomance();
}
