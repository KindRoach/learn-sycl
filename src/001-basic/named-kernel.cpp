#include <sycl/sycl.hpp>

#include "cpp-bench-utils/utils.hpp"

class MyKernel; // only for naming kernel.

int main() {
    sycl::queue q{cbu::gpu_selector_by_cu};
    q.submit([&](sycl::handler &h) {
        sycl::stream stream(65536, 256, h);
        h.parallel_for<MyKernel>(8, [stream](sycl::id<1> i) {
            stream << "Hello World from kernel " << i.get(0) << " !\n";
        });
    }).wait();
}
