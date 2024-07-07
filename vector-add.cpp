#include <algorithm>
#include <iostream>
#include <sycl/sycl.hpp>

#include "util.hpp"

using namespace sycl;

constexpr size_t N = 1024 * 1024;

void vector_add(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c) {
    for (int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }
}

void vector_add(
    queue &q,
    const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c) {
    // Create buffers associated with inputs and output
    buffer<float, 1> a_buf{a}, b_buf{b}, c_buf{c};

    // Submit the kernel to the queue
    q.submit([&](handler &h) {
        accessor A{a_buf, h, read_only};
        accessor B{b_buf, h, read_only};
        accessor C{c_buf, h, write_only};

        // BEGIN CODE SNIP
        h.parallel_for(range{N}, [=](id<1> idx) {
            C[idx] = A[idx] + B[idx];
        });
        // END CODE SNIP
    });
}

void test_performance() {
    std::vector<float> a(N), b(N), c(N);

    std::cout << "CPU single core: ";
    measureExecutionTime([&] { vector_add(a, b, c); });

    std::cout << "CPU SYCL: ";
    queue cpu_q{cpu_selector_v};
    measureExecutionTime([&] { vector_add(cpu_q, a, b, c); });

    std::cout << "GPU: ";
    queue gpu_q{gpu_selector_v};
    measureExecutionTime([&] { vector_add(gpu_q, a, b, c); });
}

void test_acc() {
    // Initialize input and output memory on the host
    std::vector<float> a(N), b(N), c(N);
    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 2);
    std::fill(c.begin(), c.end(), 0);

    queue gpu_q{gpu_selector_v};
    vector_add(gpu_q, a, b, c);

    // Check that all outputs match expected value
    bool passed = std::all_of(c.begin(), c.end(), [](float i) { return i == 3; });
    std::cout << (passed ? "SUCCESS" : "FAILURE") << std::endl;
}

int main() {
    test_acc();
    test_performance();
}
