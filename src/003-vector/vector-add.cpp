#include <iostream>
#include <sycl/sycl.hpp>

#include "util/bench.hpp"
#include "util/device.hpp"
#include "util/vector.hpp"

void vector_add_ref(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c) {
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
    int16_t WG_SIZE,
    int8_t SG_SIZE
>
void vector_add_nd_range(sycl::queue &q, T *a, T *b, T *c, size_t size) {
    q.parallel_for(
        sycl::nd_range<1>{size, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t offset = item.get_global_linear_id();
            c[offset] = a[offset] + b[offset];
        });
}

template<
    typename T,
    int16_t WG_SIZE,
    int8_t SG_SIZE,
    int8_t WI_SIZE
>
void vector_add_workitem_continue(sycl::queue &q, T *a, T *b, T *c, size_t size) {
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
    int16_t WG_SIZE,
    int8_t SG_SIZE,
    int8_t WI_SIZE
>
void vector_add_with_vec(sycl::queue &q, T *a, T *b, T *c, size_t size) {
    q.parallel_for(
        sycl::nd_range<1>{size / WI_SIZE, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t offset = item.get_global_linear_id() * WI_SIZE;
            sycl::vec<T, WI_SIZE> vec_a, vec_b;
            vec_a.load(0, a + offset);
            vec_b.load(0, b + offset);
            vec_a += vec_b;
            vec_a.store(0, c + offset);
        });
}

template<
    typename T,
    int16_t WG_SIZE,
    int8_t SG_SIZE,
    int8_t WI_SIZE
>
void vector_add_subgroup_continue(sycl::queue &q, T *a, T *b, T *c, size_t size) {
    q.parallel_for(
        sycl::nd_range<1>{size / WI_SIZE, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            size_t wg_offset = item.get_group(0) * WG_SIZE * WI_SIZE;
            size_t sg_offset = item.get_sub_group().get_group_id()[0] * SG_SIZE * WI_SIZE;
            size_t wi_offset = item.get_sub_group().get_local_id()[0];
            size_t offset = wg_offset + wi_offset + sg_offset;
            for (size_t j = 0; j < WI_SIZE * SG_SIZE; j += SG_SIZE) {
                c[offset + j] = a[offset + j] + b[offset + j];
            }
        });
}

template<typename T>
void acc_check(sycl::queue &q, std::vector<T> &gt, T *device_ptr, size_t size) {
    std::vector<T> actual(size);
    q.memcpy(actual.data(), device_ptr, size * sizeof(T)).wait();
    acc_check(gt, actual);
}

int main(int argc, char *argv[]) {
    using dtype = float;
    size_t loop = 1000;
    size_t size = 100 * 1024 * 1024; // 100M elements

    std::vector<dtype> a(size), b(size), c(size);
    random_fill(a);
    random_fill(b);

    std::cout << "vector_add_ref:\n";
    benchmark_func(loop, [&] { vector_add_ref(a, b, c); });

    sycl::queue q{gpu_selector_by_cu, sycl::property::queue::in_order()};
    auto *p_a = sycl::malloc_device<dtype>(size, q);
    auto *p_b = sycl::malloc_device<dtype>(size, q);
    auto *p_c = sycl::malloc_device<dtype>(size, q);
    q.memcpy(p_a, a.data(), size * sizeof(dtype)).wait();
    q.memcpy(p_b, b.data(), size * sizeof(dtype)).wait();

    using vector_copy = std::function<void(sycl::queue &, dtype *, dtype *, dtype *, size_t)>;
    std::vector<std::tuple<std::string, vector_copy> > funcs{
        {"vector_add_naive", vector_add_naive<dtype>},
        {"vector_add_nd_range", vector_add_nd_range<dtype, 256, 32>},
        {"vector_add_workitem_continue", vector_add_workitem_continue<dtype, 256, 32, 4>},
        {"vector_add_with_vec", vector_add_with_vec<dtype, 256, 32, 4>},
        {"vector_add_subgroup_continue", vector_add_subgroup_continue<dtype, 256, 32, 4>},
    };

    for (auto [func_name,func]: funcs) {
        q.fill(p_c, 0, size * sizeof(dtype)).wait();
        std::cout << func_name << ":\n";
        benchmark_sycl_kernel(loop, q, [&](sycl::queue &q) {
            func(q, p_a, p_b, p_c, size);
        });
        acc_check(q, c, p_c, size);
    }
}
