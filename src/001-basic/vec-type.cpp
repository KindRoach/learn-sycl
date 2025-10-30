#include <sycl/sycl.hpp>

#include "cpp-bench-utils/utils.hpp"

template<typename T, uint8_t SIZE>
void test_vec(sycl::queue &q, T *p, size_t m, size_t n) {
    q.submit([&](sycl::handler &h) {
        sycl::stream out(65536, 4096, h);
        h.single_task([=]() {
            sycl::vec<T, SIZE> vec;
            for (int i = 0; i < m; i++) {
                vec.load(0, p + i);
                out << "vec.load(0, p + " << i << ") = " << vec << "\n";
            }
            for (int i = 0; i < m; i++) {
                vec.load(i, p);
                out << "vec.load(" << i << ", p) = " << vec << "\n";
            }
        });
    }).wait();
}

int main() {
    using dtype = float;
    size_t m = 8, n = 8;
    sycl::queue q{cbu::gpu_selector_by_cu};
    auto *p = sycl::malloc_device<dtype>(m * n, q);

    // init matrix
    q.single_task([=]() {
        for (size_t i = 0; i < m * n; i++) {
            p[i] = static_cast<dtype>(i);
        }
    }).wait();

    // print matrix
    q.submit([&](sycl::handler &h) {
        sycl::stream out(65536, 4096, h);
        h.single_task([=]() {
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) { out << sycl::setw(3) << p[i * n + j] << " "; }
                out << "\n";
            }
        });
    }).wait();

    // test vec
    test_vec<dtype, 4>(q, p, m, n);
}
