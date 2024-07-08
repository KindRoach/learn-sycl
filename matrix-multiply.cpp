#include <algorithm>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

#include "util.hpp"

using namespace sycl;

constexpr size_t N = 2048;
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
    const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c) {
    // Create buffers associated with inputs and output
    buffer<float, 2> a_buf(a.data(), range<2>(N, N)),
            b_buf(b.data(), range<2>(N, N)),
            c_buf(c.data(), range<2>(N, N));

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
    queue q,
    const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c) {
    // Create buffers associated with inputs and output
    buffer<float, 2> a_buf(a.data(), range<2>(N, N)),
            b_buf(b.data(), range<2>(N, N)),
            c_buf(c.data(), range<2>(N, N));

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
    measureExecutionTime([&] { matrix_multiply(a, b, c); });

    std::cout << "CPU SYCL bais: ";
    queue cpu_q{cpu_selector_v};
    measureExecutionTime([&] { matrix_multiply(cpu_q, a, b, c); });

    std::cout << "CPU SYCL ND-range: ";
    measureExecutionTime([&] { matrix_multiply_nd_range(cpu_q, a, b, c); });

    std::cout << "GPU basic: ";
    queue gpu_q{gpu_selector_v};
    measureExecutionTime([&] { matrix_multiply(gpu_q, a, b, c); });

    std::cout << "GPU ND-range: ";
    measureExecutionTime([&] { matrix_multiply_nd_range(gpu_q, a, b, c); });
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

    queue gpu_q{gpu_selector_v};
    matrix_multiply_nd_range(gpu_q, a, b, c);

    // Check that all outputs match serial execution
    bool passed = true;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            float gold = 0;
            for (int k = 0; k < N; ++k) {
                gold += a[j * N + k] * b[k * N + i];
            }
            if (std::abs(gold - c[j * N + i]) / gold > 1.0E-05) {
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
