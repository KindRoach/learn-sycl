#include <algorithm>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

#include "util.hpp"

using namespace sycl;

constexpr size_t N = 512;
constexpr size_t B = 16;

void matrix_multiply(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c) {
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

void test_perfomance() {
    std::vector<float> a(N * N), b(N * N), c(N * N);

    std::cout << "CPU single core: ";
    benchmark_func([&] { matrix_multiply(a, b, c); });

    queue cpu_q{cpu_selector_v};
    queue gpu_q{gpu_selector_by_cu};

    // wrap buffer lifecycle
    {
        buffer<float, 2> a_buf(a.data(), range<2>(N, N)),
                b_buf(b.data(), range<2>(N, N)),
                c_buf(c.data(), range<2>(N, N));

        std::cout << "CPU SYCL bais: ";
        benchmark_sycl_kernel([&](queue &q) { matrix_multiply(q, a_buf, b_buf, c_buf); }, cpu_q);
    }

    // wrap buffer lifecycle
    {
        buffer<float, 2> a_buf(a.data(), range<2>(N, N)),
                b_buf(b.data(), range<2>(N, N)),
                c_buf(c.data(), range<2>(N, N));

        std::cout << "CPU SYCL ND-range: ";
        benchmark_sycl_kernel([&](queue &q) { matrix_multiply_nd_range(q, a_buf, b_buf, c_buf); }, cpu_q);
    }

    // wrap buffer lifecycle
    {
        buffer<float, 2> a_buf(a.data(), range<2>(N, N)),
                b_buf(b.data(), range<2>(N, N)),
                c_buf(c.data(), range<2>(N, N));

        std::cout << "GPU basic: ";
        benchmark_sycl_kernel([&](queue &q) { matrix_multiply(q, a_buf, b_buf, c_buf); }, gpu_q);
    }

    // wrap buffer lifecycle
    {
        buffer<float, 2> a_buf(a.data(), range<2>(N, N)),
                b_buf(b.data(), range<2>(N, N)),
                c_buf(c.data(), range<2>(N, N));

        std::cout << "GPU ND-range: ";
        benchmark_sycl_kernel([&](queue &q) { matrix_multiply_nd_range(q, a_buf, b_buf, c_buf); }, gpu_q);
    }
}

void test_acc() {
    // Initialize input and output memory on the host
    std::vector<float> a(N * N), b(N * N), c(N * N);
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    auto rng = [&]() { return dist(gen); };
    std::generate(a.begin(), a.end(), rng);
    std::generate(b.begin(), b.end(), rng);
    std::fill(c.begin(), c.end(), 0);

    queue gpu_q{gpu_selector_by_cu};

    // wrap buffer lifecycle
    {
        // Create buffers associated with inputs and output
        buffer<float, 2> a_buf(a.data(), range<2>(N, N)),
                b_buf(b.data(), range<2>(N, N)),
                c_buf(c.data(), range<2>(N, N));

        matrix_multiply_nd_range(gpu_q, a_buf, b_buf, c_buf);
    }

    // Check that all outputs match serial execution
    bool passed = true;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float gold = 0;
            for (int k = 0; k < N; ++k) {
                gold += a[i * N + k] * b[k * N + j];
            }
            if (std::abs(gold - c[i * N + j]) / gold > 1.0E-05) {
                passed = false;
            }
        }
    }
    std::cout << (passed ? "SUCCESS" : "FAILURE") << std::endl;
}


int main() {
    test_acc();
    test_perfomance();
}
