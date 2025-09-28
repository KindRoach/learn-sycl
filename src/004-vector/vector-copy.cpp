#include <sycl/sycl.hpp>

#include "util/util.hpp"

template<typename T>
void vector_copy_naive(sycl::queue &q, T *src, T *out, size_t size) {
    q.parallel_for({size}, [=](sycl::id<1> idx) {
        size_t offset = idx.get(0);
        out[offset] = src[offset];
    });
}

template<
    typename T,
    size_t WG_SIZE,
    size_t SG_SIZE
>
void vector_copy_nd_range(sycl::queue &q, T *src, T *out, size_t size) {
    check_divisible(size, WG_SIZE, "Global size must be divisible by work-group size");

    q.parallel_for(
        sycl::nd_range<1>{size, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t offset = item.get_global_linear_id();
            out[offset] = src[offset];
        });
}

template<
    typename T,
    size_t WG_SIZE,
    size_t SG_SIZE,
    size_t WI_SIZE
>
void vector_copy_workitem_continuous(sycl::queue &q, T *src, T *out, size_t size) {
    check_divisible(size, WG_SIZE * WI_SIZE, "Size must be divisible by WG_SIZE * WI_SIZE");

    q.parallel_for(
        sycl::nd_range<1>{size / WI_SIZE, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t i = item.get_global_linear_id();
            T *src_base = src + i * WI_SIZE;
            T *out_base = out + i * WI_SIZE;
            for (size_t j = 0; j < WI_SIZE; j++) {
                out_base[j] = src_base[j];
            }
        });
}

template<
    typename T,
    size_t WG_SIZE,
    size_t SG_SIZE,
    size_t WI_SIZE
>
void vector_copy_with_vec(sycl::queue &q, T *src, T *out, size_t size) {
    check_divisible(size, WG_SIZE * WI_SIZE, "Size must be divisible by WG_SIZE * WI_SIZE");

    q.parallel_for(
        sycl::nd_range<1>{size / WI_SIZE, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t i = item.get_global_linear_id();
            sycl::vec<T, WI_SIZE> vec;
            vec.load(i, src);
            vec.store(i, out);
        });
}

template<
    typename T,
    size_t WG_SIZE,
    size_t SG_SIZE,
    size_t WI_SIZE
>
void vector_copy_subgroup_continuous(sycl::queue &q, T *src, T *out, size_t size) {
    check_divisible(size, WG_SIZE * WI_SIZE, "Size must be divisible by WG_SIZE * WI_SIZE");

    q.parallel_for(
        sycl::nd_range<1>{size / WI_SIZE, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t wg_offset = item.get_group(0) * WG_SIZE * WI_SIZE;
            size_t sg_offset = item.get_sub_group().get_group_id()[0] * SG_SIZE * WI_SIZE;
            size_t wi_offset = item.get_sub_group().get_local_id()[0];

            T *src_base = src + wg_offset + sg_offset + wi_offset;
            T *out_base = out + wg_offset + sg_offset + wi_offset;

            for (size_t j = 0; j < WI_SIZE * SG_SIZE; j += SG_SIZE) {
                out_base[j] = src_base[j];
            }
        });
}

int main() {
    using dtype = float;
    constexpr uint16_t wg_size = 256;
    constexpr uint8_t sg_size = 32;
    constexpr uint8_t wi_size = 4;

    size_t secs = 10;
    size_t size = 100 * 1024 * 1024; // 100M elements

    std::vector<dtype> vec(size);
    random_fill(vec);

    sycl::queue q{gpu_selector_by_cu, sycl::property::queue::in_order()};
    auto *d_src = sycl::malloc_device<dtype>(size, q);
    auto *d_dst = sycl::malloc_device<dtype>(size, q);
    q.memcpy(d_src, vec.data(), size * sizeof(dtype)).wait();

    using func_t = std::function<void(sycl::queue &, dtype *, dtype *, size_t)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {"vector_copy_naive", vector_copy_naive<dtype>},
        {"vector_copy_nd_range", vector_copy_nd_range<dtype, wg_size, sg_size>},
        {"vector_copy_workitem_continuous", vector_copy_workitem_continuous<dtype, wg_size, sg_size, wi_size>},
        {"vector_copy_with_vec", vector_copy_with_vec<dtype, wg_size, sg_size, wi_size>},
        {"vector_copy_subgroup_continuous", vector_copy_subgroup_continuous<dtype, wg_size, sg_size, wi_size>},
    };

    for (auto [func_name,func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        q.fill(d_dst, dtype{0}, size).wait();
        benchmark_func_by_time(secs, [&]() {
            func(q, d_src, d_dst, size);
            q.wait();
        });
        sycl_acc_check(q, vec, d_dst);
    }
}
