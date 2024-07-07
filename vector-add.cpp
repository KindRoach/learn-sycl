#include <algorithm>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr size_t N = 1'000'000;

void vector_add(std::vector<float> &a, std::vector<float> &b, std::vector<float> &c) {
    queue q;

    // Create buffers associated with inputs and output
    buffer<float, 1> a_buf{a}, b_buf{b}, c_buf{c};

    // Submit the kernel to the queue
    q.submit([&](handler &h) {
        accessor A{a_buf, h};
        accessor B{b_buf, h};
        accessor C{c_buf, h};

        // BEGIN CODE SNIP
        h.parallel_for(range{N}, [=](id<1> idx) {
            C[idx] = A[idx] + B[idx];
        });
        // END CODE SNIP
    });
}

void test_performance() {
    std::vector<float> a(N), b(N), c(N);
    vector_add(a, b, c);
}

void test_acc() {
    // Initialize input and output memory on the host
    std::vector<float> a(N), b(N), c(N);
    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 2);
    std::fill(c.begin(), c.end(), 0);

    vector_add(a, b, c);

    // Check that all outputs match expected value
    bool passed = std::all_of(c.begin(), c.end(), [](float i) { return i == 3; });
    std::cout << (passed ? "SUCCESS" : "FAILURE") << std::endl;
}

int main() {
    test_performance();
}
