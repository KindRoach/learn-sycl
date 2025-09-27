#include <sycl/sycl.hpp>

#include "util/util.hpp"


template<int WG_SIZE, int SG_SIZE>
void print_sub_group_mapping_1d(sycl::queue &q) {
    std::cout << "=========================" << std::endl;

    using namespace sycl;
    q.submit([](auto &h) {
        stream out(65536, 4096, h);
        h.parallel_for(
            nd_range(range{WG_SIZE * 2}, range{WG_SIZE}),
            [out](nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                sycl_print_item_info(it);
            });
    }).wait();
}

template<int WG_SIZE, int SG_SIZE>
void print_sub_group_mapping_2d(sycl::queue &q) {
    std::cout << "=========================" << std::endl;

    using namespace sycl;
    q.submit([&](auto &h) {
        stream out(65536, 4096, h);
        h.parallel_for(
            nd_range(range{WG_SIZE * 2, WG_SIZE * 2}, range{WG_SIZE, WG_SIZE}),
            [=](nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                sycl_print_item_info(it);
            });
    }).wait();
}


int main() {
    using namespace sycl;
    queue q{gpu_selector_by_cu};
    print_sub_group_mapping_1d<64, 16>(q);
    print_sub_group_mapping_1d<64, 32>(q);
    print_sub_group_mapping_2d<8, 16>(q);
    print_sub_group_mapping_2d<8, 32>(q);
}
