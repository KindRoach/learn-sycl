#include <iostream>
#include <numeric>

#include "util/bench.hpp"
#include "util/device.hpp"
#include "util/vector.hpp"

template<typename T>
void vector_dot_ref(const std::vector<T> &a, const std::vector<T> &b, std::vector<T> &out) {
    T sum = T{0};
    for (int i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    out[0] = sum;
}

template<typename T>
void vector_dot_reduction(sycl::queue &q, T *a, T *b, T *out, size_t size) {
    q.single_task([=]() {
        out[0] = T{0};
    });

    q.submit([&](sycl::handler &h) {
        auto red = sycl::reduction(out, sycl::plus<>());
        h.parallel_for(size, red, [=](sycl::id<1> i, auto &acc) {
            acc.combine(a[i] * b[i]);
        });
    });
}

template<
    typename T,
    uint16_t WG_SIZE,
    uint8_t SG_SIZE
>
void vector_sum_group_reduce_atomic_collect(sycl::queue &q, T *a, T *b, T *out, size_t size) {
    q.single_task([=]() {
        out[0] = T{0};
    });

    q.parallel_for(
        sycl::nd_range<1>{size, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            auto group = item.get_group();
            size_t i = item.get_global_linear_id();
            float group_sum = reduce_over_group(group, a[i] * b[i], sycl::plus<>());
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
    uint16_t WG_SIZE,
    uint8_t SG_SIZE,
    uint8_t WI_SIZE
>
void vector_sum_group_reduce_atomic_collect_vec(sycl::queue &q, T *a, T *b, T *out, size_t size) {
    q.single_task([=]() {
        out[0] = T{0};
    });

    q.parallel_for(
        sycl::nd_range<1>{size / WI_SIZE, WG_SIZE},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            auto group = item.get_group();
            size_t i = item.get_global_linear_id();

            sycl::vec<T, WI_SIZE> vec_a, vec_b;
            vec_a.load(i, a);
            vec_b.load(i, b);
            vec_a *= vec_b;

            T sum_i = T{0};
            for (int j = 0; j < WI_SIZE; ++j) {
                sum_i += vec_a[j];
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

    size_t loop = 1000;
    size_t size = 100 * 1024 * 1024; // 100M elements

    std::vector<dtype> a(size), b(size), out(1);
    random_fill(a);
    random_fill(b);

    std::cout << "vector_sum_ref:\n";
    benchmark_func(loop, [&] { vector_dot_ref(a, b, out); });

    sycl::queue q{gpu_selector_by_cu, sycl::property::queue::in_order()};
    auto *p_a = sycl::malloc_device<dtype>(size, q);
    auto *p_b = sycl::malloc_device<dtype>(size, q);
    auto *p_out = sycl::malloc_device<dtype>(1, q);
    q.memcpy(p_a, a.data(), size * sizeof(dtype)).wait();
    q.memcpy(p_b, b.data(), size * sizeof(dtype)).wait();

    using func_t = std::function<void(sycl::queue &, dtype *, dtype *, dtype *, size_t)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {
            "vector_dot_reduction",
            vector_dot_reduction<dtype>
        },
        {
            "vector_sum_group_reduce_atomic_collect",
            vector_sum_group_reduce_atomic_collect<dtype, wg_size, sg_size>
        },
        {
            "vector_sum_group_reduce_atomic_collect_vec",
            vector_sum_group_reduce_atomic_collect_vec<dtype, wg_size, sg_size, wi_size>
        },
    };

    for (auto [func_name,func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        q.fill(p_out, dtype{0}, 1).wait();
        benchmark_sycl_kernel(loop, q, [&](sycl::queue &q) {
            func(q, p_a, p_b, p_out, size);
        });
        acc_check(q, out, p_out);
    }
}
