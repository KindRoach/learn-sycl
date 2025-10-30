#include <sycl/sycl.hpp>

#include "cpp-bench-utils/utils.hpp"

template<typename T, uint16_t WG_SIZE>
void test_local_memory(sycl::queue &q) {
    q.submit([&](sycl::handler &h) {
        sycl::local_accessor<T> slm{WG_SIZE, h};
        sycl::stream stream(65536, 256, h);
        h.parallel_for(sycl::nd_range<1>{2 * WG_SIZE, WG_SIZE}, [=](sycl::nd_item<1> item_id) {
            size_t global_id = item_id.get_global_linear_id();
            size_t local_id = item_id.get_local_linear_id();

            size_t slm_read_id = local_id;
            size_t slm_write_id = WG_SIZE - local_id - 1;

            slm[slm_read_id] = static_cast<T>(global_id);
            item_id.barrier(sycl::access::fence_space::local_space);
            stream << "global id = " << global_id
                    << ", local id = " << local_id
                    << ", slm[" << slm_write_id << "]=" << slm[slm_write_id]
                    << "\n";
        });
    }).wait();
}

int main() {
    using dtype = float;
    constexpr uint16_t wg_size = 8;
    sycl::queue q{cbu::gpu_selector_by_cu};
    test_local_memory<dtype, wg_size>(q);
}
