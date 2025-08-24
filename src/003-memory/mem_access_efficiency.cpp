#include <sycl/sycl.hpp>

#include "util/device.hpp"
#include "util/bench.hpp"

template<
    typename T,
    size_t WG_SIZE,
    size_t SG_SIZE,
    size_t WI_SIZE
>
void access_mem_workitem_continuous(sycl::queue &q, T *device_prt, size_t size) {
    q.parallel_for(
        sycl::nd_range<1>{size / WI_SIZE, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t i = item.get_global_linear_id();
            T *base = device_prt + i * WI_SIZE;
            for (size_t j = 0; j < WI_SIZE; j++) {
                base[j] += 1;
            }
        });
}

template<
    typename T,
    size_t WG_SIZE,
    size_t SG_SIZE,
    size_t WI_SIZE
>
void access_mem_subgroup_continuous(sycl::queue &q, T *device_prt, size_t size) {
    q.parallel_for(
        sycl::nd_range<1>{size / WI_SIZE, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t wg_offset = item.get_group(0) * WG_SIZE * WI_SIZE;
            size_t sg_offset = item.get_sub_group().get_group_id()[0] * SG_SIZE * WI_SIZE;
            size_t wi_offset = item.get_sub_group().get_local_id()[0];

            T *base = device_prt + wg_offset + sg_offset + wi_offset;

            for (size_t j = 0; j < WI_SIZE; j++) {
                base[j * SG_SIZE] += 1;
            }
        });
}

template<typename T>
void acc_check(sycl::queue &q, T *device_prt, size_t size) {
    // copy device mem to host
    std::vector<T> host_vec(size);
    q.memcpy(host_vec.data(), device_prt, size * sizeof(T)).wait();
    T x0 = host_vec[0];

    for (size_t i = 0; i < size; i++) {
        T &x = host_vec[i];
        if (x != x0) {
            std::cerr << "ACC FAILED at host[" << i << "]=" << x << std::endl;
            exit(1);
        }
    }
}

int main() {
    sycl::queue q{gpu_selector_by_cu};

    using dtype = int32_t;
    size_t size = 100 * 1024 * 1024; // 100M elements

    auto *device_prt = sycl::malloc_device<dtype>(size, q);

    q.fill(device_prt, dtype{0}, size).wait();
    benchmark_sycl_kernel(1000, q, [&](sycl::queue &q) {
        access_mem_workitem_continuous<dtype, 64, 32, 16>(q, device_prt, size);
    });
    acc_check(q, device_prt, size);

    q.fill(device_prt, dtype{0}, size).wait();
    benchmark_sycl_kernel(1000, q, [&](sycl::queue &q) {
        access_mem_subgroup_continuous<dtype, 64, 32, 16>(q, device_prt, size);
    });
    acc_check(q, device_prt, size);
}
