#include <sycl/sycl.hpp>

#include "util/device.hpp"


template<int WG_SIZE, int SG_SIZE>
void print_sub_group_mapping_1d(sycl::queue &q) {
    std::cout << "=========================" << std::endl;

    using namespace sycl;
    q.submit([](auto &h) {
        stream out(65536, 4096, h);
        h.parallel_for(
            nd_range(range{WG_SIZE * 2}, range{WG_SIZE}),
            [out](nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                size_t group_id_x = it.get_group(0);
                size_t local_id_x = it.get_local_id(0);
                size_t global_id = it.get_global_linear_id();

                auto sg = it.get_sub_group();
                size_t sg_size = sg.get_local_range()[0];
                size_t sg_group_id = sg.get_group_id()[0];
                size_t sg_id = sg.get_local_id()[0];

                out << " group_id = " << setw(3) << group_id_x
                        << " local_id = " << setw(3) << local_id_x
                        << " global_id = " << setw(3) << global_id
                        << " sg_group_id = " << setw(3) << sg_group_id
                        << " sg_id = " << setw(3) << sg_id
                        << " sg_size = " << setw(3) << sg_size
                        << endl;
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
                size_t group_id_x = it.get_group(0);
                size_t group_id_y = it.get_group(1);
                size_t local_id_x = it.get_local_id(0);
                size_t local_id_y = it.get_local_id(1);
                size_t global_id = it.get_global_linear_id();

                auto sg = it.get_sub_group();
                size_t sg_size = sg.get_local_range()[0];
                size_t sg_group_id = sg.get_group_id()[0];
                size_t sg_id = sg.get_local_id()[0];

                out << " group_id = (" << setw(3) << group_id_x << ", " << setw(3) << group_id_y << ")"
                        << " local_id = (" << setw(3) << local_id_x << ", " << setw(3) << local_id_y << ")"
                        << " global_id = " << setw(3) << global_id
                        << " sg_group_id = " << setw(3) << sg_group_id
                        << " sg_id = " << setw(3) << sg_id
                        << " sg_size = " << setw(3) << sg_size
                        << endl;
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
