#include <sycl/sycl.hpp>

#include "util/device.hpp"

template<typename T>
void test_mem(sycl::queue &q, T *device_prt, size_t size) {
    std::cout << "test_mem:" << device_prt << std::endl;

    // write
    q.parallel_for({size}, [=](sycl::id<1> id) {
        size_t i = id.get(0);
        device_prt[i] = T{i};
    });

    // read
    q.submit([&](sycl::handler &h) {
        sycl::stream stream(65536, 256, h);
        h.parallel_for({size}, [=](sycl::id<1> id) {
            size_t i = id.get(0);
            stream << "Index: " << i << ", Value: " << device_prt[i] << "\n";
        });
    }).wait();
}

int main() {
    sycl::queue q{gpu_selector_by_cu};

    using dtype = float;
    size_t size = 8;

    // C-style
    dtype *p1 = static_cast<dtype *>(sycl::malloc_device(size * sizeof(dtype), q));

    // Cpp-style
    dtype *p2 = sycl::malloc_device<dtype>(size, q);

    // Cpp-allocator-style (only host and shared surpported)
    sycl::usm_allocator<dtype, sycl::usm::alloc::shared> alloc(q);
    dtype *p3 = alloc.allocate(size);

    test_mem<dtype>(q, p1, size);
    test_mem<dtype>(q, p2, size);
    test_mem<dtype>(q, p3, size);

    sycl::free(p1, q);
    sycl::free(p2, q);
    alloc.deallocate(p3, size);
}
