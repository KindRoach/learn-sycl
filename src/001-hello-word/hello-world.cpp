#include <sycl/sycl.hpp>

int main() {
    sycl::queue queue;

    constexpr int n_items = 16;
    queue.parallel_for(sycl::range<1>(n_items), [](sycl::id<1> i) {
        sycl::ext::oneapi::experimental::printf(
            "Hello World from kernel %d!\n", static_cast<int>(i)
        );
    }).wait();
}
