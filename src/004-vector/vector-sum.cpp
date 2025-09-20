#include <iostream>
#include <numeric>
#include <sycl/sycl.hpp>

#include "util/bench.hpp"
#include "util/device.hpp"
#include "util/validate.hpp"
#include "util/vector.hpp"

template<typename T>
void vector_sum_ref(const std::vector<T> &vec, std::vector<T> &out) {
    out[0] = std::accumulate(vec.begin(), vec.end(), T{0});
}

template<typename T>
void vector_sum_atomic(sycl::queue &q, T *vec, T *out, size_t size) {
    q.single_task([=]() {
        out[0] = T{0};
    });

    q.parallel_for(size, [=](sycl::id<1> i) {
        auto v = sycl::atomic_ref<T,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>(out[0]);
        v += vec[i];
    });
}

template<typename T>
void vector_sum_reduction(sycl::queue &q, T *vec, T *out, size_t size) {
    q.single_task([=]() {
        out[0] = T{0};
    });

    q.submit([&](sycl::handler &h) {
        auto red = sycl::reduction(out, sycl::plus<>());
        h.parallel_for(size, red, [=](sycl::id<1> i, auto &acc) {
            acc.combine(vec[i]);
        });
    });
}

template<
    typename T,
    size_t WG_SIZE,
    size_t SG_SIZE
>
void vector_sum_group_reduce_recursion(sycl::queue &q, T *vec, T *out, size_t size) {
    size_t group_num = (size + WG_SIZE - 1) / WG_SIZE;
    if (group_num > 1) {
        T *temp = sycl::malloc_device<T>(group_num, q);
        q.parallel_for(
            sycl::nd_range<1>{WG_SIZE * group_num, WG_SIZE},
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                auto group = item.get_group();
                size_t i = item.get_global_linear_id();
                T x = i < size ? vec[i] : T{0};
                float group_sum = reduce_over_group(group, x, sycl::plus<>());
                if (group.leader()) {
                    temp[group.get_group_linear_id()] = group_sum;
                }
            });
        vector_sum_group_reduce_recursion<T, WG_SIZE, SG_SIZE>(q, temp, out, group_num);
        sycl::free(temp, q);
    } else {
        q.parallel_for(
            sycl::nd_range<1>{WG_SIZE * group_num, WG_SIZE},
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                auto group = item.get_group();
                size_t i = item.get_global_linear_id();
                T x = i < size ? vec[i] : 0;
                float group_sum = reduce_over_group(group, x, sycl::plus<>());
                if (group.leader()) {
                    out[0] = group_sum;
                }
            });
    }
}

template<
    typename T,
    size_t WG_SIZE,
    size_t SG_SIZE
>
void vector_sum_group_reduce_atomic_collect(sycl::queue &q, T *vec, T *out, size_t size) {
    check_divisible(size, WG_SIZE, "Global size must be divisible by work-group size");

    q.single_task([=]() {
        out[0] = T{0};
    });

    q.parallel_for(
        sycl::nd_range<1>{size, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            auto group = item.get_group();
            size_t i = item.get_global_linear_id();
            float group_sum = reduce_over_group(group, vec[i], sycl::plus<>());
            if (group.leader()) {
                auto v = sycl::atomic_ref<T,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(out[0]);
                v += group_sum;
            }
        });
}

template<
    typename T,
    size_t WG_SIZE,
    size_t SG_SIZE,
    size_t WI_SIZE
>
void vector_sum_group_reduce_atomic_collect_vec(sycl::queue &q, T *vec, T *out, size_t size) {
    check_divisible(size, WG_SIZE * WI_SIZE, "Size must be divisible by WG_SIZE * WI_SIZE");

    q.single_task([=]() {
        out[0] = T{0};
    });

    q.parallel_for(
        sycl::nd_range<1>{size / WI_SIZE, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            auto group = item.get_group();
            size_t i = item.get_global_linear_id();

            sycl::vec<T, WI_SIZE> vec_i;
            vec_i.load(i, vec);

            T sum_i = T{0};
            for (int j = 0; j < WI_SIZE; ++j) {
                sum_i += vec_i[j];
            }

            float group_sum = reduce_over_group(group, sum_i, sycl::plus<>());
            if (group.leader()) {
                auto v = sycl::atomic_ref<T,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(out[0]);
                v += group_sum;
            }
        });
}

template<
    typename T,
    size_t WG_SIZE,
    size_t SG_SIZE,
    size_t WI_SIZE
>
void vector_sum_group_reduce_atomic_collect_sg(sycl::queue &q, T *vec, T *out, size_t size) {
    check_divisible(size, WG_SIZE * WI_SIZE, "Size must be divisible by WG_SIZE * WI_SIZE");

    q.single_task([=]() {
        out[0] = T{0};
    });

    q.parallel_for(
        sycl::nd_range<1>{size / WI_SIZE, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            auto group = item.get_group();
            size_t i = item.get_global_linear_id();

            size_t wg_offset = item.get_group(0) * WG_SIZE * WI_SIZE;
            size_t sg_offset = item.get_sub_group().get_group_id()[0] * SG_SIZE * WI_SIZE;
            size_t wi_offset = item.get_sub_group().get_local_id()[0];
            size_t offset = wg_offset + wi_offset + sg_offset;

            T sum_i = T{0};
            for (size_t j = 0; j < WI_SIZE * SG_SIZE; j += SG_SIZE) {
                sum_i += vec[offset + j];
            }

            float group_sum = reduce_over_group(group, sum_i, sycl::plus<>());
            if (group.leader()) {
                auto v = sycl::atomic_ref<T,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(out[0]);
                v += group_sum;
            }
        });
}


int main() {
    using dtype = float;
    constexpr uint16_t wg_size = 256;
    constexpr uint8_t sg_size = 32;
    constexpr uint8_t wi_size = 4;

    size_t secs = 10;
    size_t loop = 1000;
    size_t size = 100 * 1024 * 1024; // 100M elements

    std::vector<dtype> vec(size), out(1);
    random_fill(vec);

    std::cout << "vector_sum_ref:\n";
    benchmark_func_by_time(secs, [&] { vector_sum_ref(vec, out); });

    sycl::queue q{gpu_selector_by_cu, sycl::property::queue::in_order()};
    auto *vec_p = sycl::malloc_device<dtype>(size, q);
    auto *out_p = sycl::malloc_device<dtype>(1, q);
    q.memcpy(vec_p, vec.data(), size * sizeof(dtype)).wait();

    using func_t = std::function<void(sycl::queue &, dtype *, dtype *, size_t)>;
    std::vector<std::tuple<std::string, func_t, size_t> > funcs{
        {
            "vector_sum_atomic",
            vector_sum_atomic<dtype>,
            loop / 100
        },
        {
            "vector_sum_reduction",
            vector_sum_reduction<dtype>,
            loop
        },
        {
            "vector_sum_group_reduce_atomic_collect",
            vector_sum_group_reduce_atomic_collect<dtype, wg_size, sg_size>,
            loop
        },
        {
            "vector_sum_group_reduce_recursion",
            vector_sum_group_reduce_recursion<dtype, wg_size, sg_size>,
            loop
        },
        {
            "vector_sum_group_reduce_atomic_collect_vec",
            vector_sum_group_reduce_atomic_collect_vec<dtype, wg_size, sg_size, wi_size>,
            loop
        },
        {
            "vector_sum_group_reduce_atomic_collect_sg",
            vector_sum_group_reduce_atomic_collect_sg<dtype, wg_size, sg_size, wi_size>,
            loop
        },
    };

    for (auto [func_name,func, loop]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        q.fill(out_p, dtype{0}, 1).wait();
        benchmark_func_by_time(secs, [&]() {
            func(q, vec_p, out_p, size);
            q.wait();
        });
        acc_check(q, out, out_p);
    }
}
