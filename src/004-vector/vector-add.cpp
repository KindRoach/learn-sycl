#include <iostream>
#include <sycl/sycl.hpp>

#include "util/util.hpp"

template<typename T>
void vector_add_ref(const std::vector<T> &a, const std::vector<T> &b, std::vector<T> &c) {
    for (int i = 0; i < c.size(); ++i) {
        c[i] = a[i] + b[i];
    }
}

template<typename T>
void vector_add_naive(sycl::queue &q, T *a, T *b, T *c, size_t size) {
    q.parallel_for({size}, [=](sycl::id<1> idx) {
        size_t offset = idx.get(0);
        c[offset] = a[offset] + b[offset];
    });
}

template<
    typename T,
    size_t WG_SIZE,
    size_t SG_SIZE
>
void vector_add_nd_range(sycl::queue &q, T *a, T *b, T *c, size_t size) {
    check_divisible(size, WG_SIZE, "Global size must be divisible by work-group size");

    q.parallel_for(
        sycl::nd_range<1>{size, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t offset = item.get_global_linear_id();
            c[offset] = a[offset] + b[offset];
        });
}

template<
    typename T,
    size_t WG_SIZE,
    size_t SG_SIZE,
    size_t WI_SIZE
>
void vector_add_workitem_continue(sycl::queue &q, T *a, T *b, T *c, size_t size) {
    check_divisible(size, WG_SIZE * WI_SIZE, "Size must be divisible by WG_SIZE * WI_SIZE");

    q.parallel_for(
        sycl::nd_range<1>{size / WI_SIZE, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t offset = item.get_global_linear_id() * WI_SIZE;
            for (size_t i = 0; i < WI_SIZE; i++) {
                c[offset + i] = a[offset + i] + b[offset + i];
            }
        });
}

template<
    typename T,
    size_t WG_SIZE,
    size_t SG_SIZE,
    size_t WI_SIZE
>
void vector_add_with_vec(sycl::queue &q, T *a, T *b, T *c, size_t size) {
    check_divisible(size, WG_SIZE * WI_SIZE, "Size must be divisible by WG_SIZE * WI_SIZE");

    q.parallel_for(
        sycl::nd_range<1>{size / WI_SIZE, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t offset = item.get_global_linear_id();
            sycl::vec<T, WI_SIZE> vec_a, vec_b;
            vec_a.load(offset, a);
            vec_b.load(offset, b);
            vec_a += vec_b;
            vec_a.store(offset, c);
        });
}

template<
    typename T,
    size_t WG_SIZE,
    size_t SG_SIZE,
    size_t WI_SIZE
>
void vector_add_subgroup_continue(sycl::queue &q, T *a, T *b, T *c, size_t size) {
    check_divisible(size, WG_SIZE * WI_SIZE, "Size must be divisible by WG_SIZE * WI_SIZE");

    q.parallel_for(
        sycl::nd_range<1>{size / WI_SIZE, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t wg_offset = item.get_group(0) * WG_SIZE * WI_SIZE;
            size_t sg_offset = item.get_sub_group().get_group_id()[0] * SG_SIZE * WI_SIZE;
            size_t wi_offset = item.get_sub_group().get_local_id()[0];
            size_t offset = wg_offset + sg_offset + wi_offset;
            for (size_t j = 0; j < WI_SIZE * SG_SIZE; j += SG_SIZE) {
                c[offset + j] = a[offset + j] + b[offset + j];
            }
        });
}


int main(int argc, char *argv[]) {
    using dtype = float;
    constexpr uint16_t wg_size = 256;
    constexpr uint8_t sg_size = 32;
    constexpr uint8_t wi_size = 4;

    size_t secs = 10;
    size_t size = 100 * 1024 * 1024; // 100M elements

    std::vector<dtype> a(size), b(size), c(size);
    random_fill(a);
    random_fill(b);

    std::cout << "vector_add_ref:\n";
    benchmark_func_by_time(secs, [&] { vector_add_ref(a, b, c); });

    sycl::queue q{gpu_selector_by_cu, sycl::property::queue::in_order()};
    auto *d_a = sycl::malloc_device<dtype>(size, q);
    auto *d_b = sycl::malloc_device<dtype>(size, q);
    auto *d_c = sycl::malloc_device<dtype>(size, q);
    q.memcpy(d_a, a.data(), size * sizeof(dtype)).wait();
    q.memcpy(d_b, b.data(), size * sizeof(dtype)).wait();

    using func_t = std::function<void(sycl::queue &, dtype *, dtype *, dtype *, size_t)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {"vector_add_naive", vector_add_naive<dtype>},
        {"vector_add_nd_range", vector_add_nd_range<dtype, wg_size, sg_size>},
        {"vector_add_workitem_continue", vector_add_workitem_continue<dtype, wg_size, sg_size, wi_size>},
        {"vector_add_with_vec", vector_add_with_vec<dtype, wg_size, sg_size, wi_size>},
        {"vector_add_subgroup_continue", vector_add_subgroup_continue<dtype, wg_size, sg_size, wi_size>},
    };

    for (auto [func_name,func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        q.fill(d_c, dtype{0}, size).wait();
        benchmark_func_by_time(secs, [&]() {
            func(q, d_a, d_b, d_c, size);
            q.wait();
        });
        sycl_acc_check(q, c, d_c);
    }

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);
}
