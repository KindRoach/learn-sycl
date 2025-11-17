#include <iostream>
#include <numeric>
#include <sycl/sycl.hpp>

#include "cpp-bench-utils/utils.hpp"

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
    cbu::check_divisible(size, WG_SIZE, "Global size must be divisible by work-group size");

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
    cbu::check_divisible(size, WG_SIZE * WI_SIZE, "Size must be divisible by WG_SIZE * WI_SIZE");

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
    cbu::check_divisible(size, WG_SIZE * WI_SIZE, "Size must be divisible by WG_SIZE * WI_SIZE");

    q.single_task([=]() {
        out[0] = T{0};
    });

    q.parallel_for(
        sycl::nd_range<1>{size / WI_SIZE, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            auto group = item.get_group();
            size_t wg_offset = item.get_group(0) * WG_SIZE * WI_SIZE;
            size_t sg_offset = item.get_sub_group().get_group_id()[0] * SG_SIZE * WI_SIZE;
            size_t wi_offset = item.get_sub_group().get_local_id()[0];
            size_t offset = wg_offset + wi_offset + sg_offset;

            T sum_i = T{0};
            for (size_t i = 0; i < WI_SIZE * SG_SIZE; i += SG_SIZE) {
                sum_i += vec[offset + i];
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
    using namespace cbu;
    using dtype = float;
    constexpr uint16_t wg_size = 256;
    constexpr uint8_t sg_size = 32;
    constexpr uint8_t wi_size = 4;

    size_t secs = 10;
    size_t size = 100 * 1024 * 1024; // 100M elements

    std::vector<dtype> vec(size), out(1);
    random_fill(vec);

    std::cout << "vector_sum_ref:\n";
    BenchmarkOptions opt{
        .total_mem_bytes = size * sizeof(dtype),
        .total_flop = size - 1
    };
    benchmark_func_by_time(secs, [&] { vector_sum_ref(vec, out); }, opt);

    sycl::queue q{gpu_selector_by_cu, sycl::property::queue::in_order()};
    auto *d_vec = sycl::malloc_device<dtype>(size, q);
    auto *d_out = sycl::malloc_device<dtype>(1, q);
    q.memcpy(d_vec, vec.data(), size * sizeof(dtype)).wait();

    using func_t = std::function<void(sycl::queue &, dtype *, dtype *, size_t)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {
            "vector_sum_atomic",
            vector_sum_atomic<dtype>,
        },
        {
            "vector_sum_reduction",
            vector_sum_reduction<dtype>,
        },
        {
            "vector_sum_group_reduce_atomic_collect",
            vector_sum_group_reduce_atomic_collect<dtype, wg_size, sg_size>,
        },
        {
            "vector_sum_group_reduce_recursion",
            vector_sum_group_reduce_recursion<dtype, wg_size, sg_size>,
        },
        {
            "vector_sum_group_reduce_atomic_collect_vec",
            vector_sum_group_reduce_atomic_collect_vec<dtype, wg_size, sg_size, wi_size>,
        },
        {
            "vector_sum_group_reduce_atomic_collect_sg",
            vector_sum_group_reduce_atomic_collect_sg<dtype, wg_size, sg_size, wi_size>,
        },
    };

    for (auto [func_name,func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        q.fill(d_out, dtype{0}, 1).wait();
        benchmark_func_by_time(secs, [&]() {
            func(q, d_vec, d_out, size);
            q.wait();
        }, opt);
        sycl_acc_check(q, out, d_out);
    }

    sycl::free(d_vec, q);
    sycl::free(d_out, q);
}
