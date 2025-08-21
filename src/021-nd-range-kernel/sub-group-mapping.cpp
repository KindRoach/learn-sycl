#include <sycl/sycl.hpp>

#include "util/device.hpp"


template<int SG_SIZE>
void print_sub_group_mapping_1d(sycl::queue &q) {
    std::cout << "=========================" << std::endl;

    using namespace sycl;
    q.submit([&](auto &h) {
        stream out(65536, 256, h);
        h.parallel_for(
                nd_range(range{64}, range{64}),
                [=](nd_item<1> it)[[sycl::reqd_sub_group_size(SG_SIZE)]] {

                    int group_id_x = it.get_group(0);

                    int local_id_x = it.get_local_id(0);

                    int global_id = it.get_global_linear_id();

                    auto sg = it.get_sub_group();
                    int sg_size = sg.get_local_range()[0];
                    int sg_group_id = sg.get_group_id()[0];
                    int sg_id = sg.get_local_id()[0];

                    out << " group_id = " << setw(2) << group_id_x
                        << " local_id = " << setw(2) << local_id_x
                        << " global_id = " << setw(2) << global_id
                        << " sg_group_id = " << setw(2) << sg_group_id
                        << " sg_id = " << setw(2) << sg_id
                        << " sg_size = " << setw(2) << sg_size
                        << endl;
                });
    }).wait();
}

template<int SG_SIZE>
void print_sub_group_mapping_2d(sycl::queue &q) {
    std::cout << "=========================" << std::endl;

    using namespace sycl;
    q.submit([&](auto &h) {
        stream out(65536, 256, h);
        h.parallel_for(
                nd_range(range{8, 8}, range{8, 8}),
                [=](nd_item<2> it)[[sycl::reqd_sub_group_size(SG_SIZE)]] {

                    int group_id_x = it.get_group(0);
                    int group_id_y = it.get_group(1);

                    int local_id_x = it.get_local_id(0);
                    int local_id_y = it.get_local_id(1);

                    int global_id = it.get_global_linear_id();

                    auto sg = it.get_sub_group();
                    int sg_size = sg.get_local_range()[0];
                    int sg_group_id = sg.get_group_id()[0];
                    int sg_id = sg.get_local_id()[0];

                    out << " group_id = (" << setw(2) << group_id_x << ", " << setw(2) << group_id_y << ")"
                        << " local_id = (" << setw(2) << local_id_x << ", " << setw(2) << local_id_y << ")"
                        << " global_id = " << setw(2) << global_id
                        << " sg_group_id = " << setw(2) << sg_group_id
                        << " sg_id = " << setw(2) << sg_id
                        << " sg_size = " << setw(2) << sg_size
                        << endl;
                });
    }).wait();
}


int main() {
    using namespace sycl;
    queue q{gpu_selector_by_cu};
    print_sub_group_mapping_1d<16>(q);
    print_sub_group_mapping_2d<16>(q);
}
