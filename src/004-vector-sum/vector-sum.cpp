#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>
#include <sycl/sycl.hpp>

#include "util.hpp"

using namespace sycl;

constexpr size_t N = 512 * 512;
constexpr size_t B = 16 * 16;

float vector_sum_native(const std::vector<float> &a) {
    return std::accumulate(a.begin(), a.end(), 0.f);
}

void vector_sum(queue &q, buffer<float, 1> &a_buf, buffer<float, 1> &o_buf) {
    // Submit the kernel to the queue
    q.submit([&](handler &h) {
        accessor a{a_buf, h, read_only};
        accessor o{o_buf, h, write_only, no_init};

        // BEGIN CODE SNIP
        h.parallel_for(range{N}, [=](id<1> i) {
            atomic_ref<float, memory_order::relaxed,
                memory_scope::system,
                access::address_space::global_space>(
                o[0]) += a[i];
        });
        // END CODE SNIP
    });
}

void vector_sum_nd_range(queue &q, buffer<float, 1> &a_buf, buffer<float, 1> &o_buf) {
    // Submit the kernel to the queue
    q.submit([&](handler &h) {
        accessor a{a_buf, h, read_only};
        accessor o{o_buf, h, write_only, no_init};

        // BEGIN CODE SNIP
        h.parallel_for(nd_range<1>{N, B}, [=](nd_item<1> it) {
            auto grp = it.get_group();
            float group_sum = reduce_over_group(grp, a[it.get_global_id(0)], plus<>());
            if (grp.leader()) {
                atomic_ref<float, memory_order::relaxed,
                    memory_scope::system,
                    access::address_space::global_space>(
                    o[0]) += group_sum;
            }
        });
        // END CODE SNIP
    });
}

void vector_sum_reduction(queue &q, buffer<float, 1> &a_buf, buffer<float, 1> &o_buf) {
    // Submit the kernel to the queue
    q.submit([&](handler &h) {
        accessor a{a_buf, h, read_only};

        auto red = reduction(o_buf, h, plus<>());
        // BEGIN CODE SNIP
        h.parallel_for(range{N}, red,
                       [=](id<1> i, auto &sum) {
                           sum += a[i];
                       });
        // END CODE SNIP
    });
}

void test_acc() {
    // Initialize input and output memory on the host
    std::vector<float> a(N), o(1);
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    auto rng = [&]() { return dist(gen); };
    std::generate(a.begin(), a.end(), rng);

    // calculate groud truth
    float gt = vector_sum_native(a);

    queue cpu_q{cpu_selector_v};
    queue gpu_q{gpu_selector_by_cu};
    std::vector<std::tuple<
        std::string,
        std::function<void(queue &, buffer<float, 1> &, buffer<float, 1> &)>,
        queue> > tests = {
        {"CPU SYCL", vector_sum, cpu_q},
        {"CPU SYCL ND-range", vector_sum_nd_range, cpu_q},
        {"CPU SYCL Reduction", vector_sum_reduction, cpu_q},
        {"GPU SYCL", vector_sum, gpu_q},
        {"GPU SYCL ND-range", vector_sum_nd_range, gpu_q},
        {"GPU SYCL Reduction", vector_sum_reduction, gpu_q}
    };

    for (auto &[name,kernel, q]: tests) {
        {
            std::fill(o.begin(), o.end(), 0);
            buffer<float, 1> a_buf(a.data(), range<1>(N)),
                    o_buf(o.data(), range<1>(1));

            kernel(q, a_buf, o_buf);
        }

        auto success = floatVectorEquals(o, {gt});
        std::cout << name << ": " << (success ? "SUCCESS" : "FAILURE") << std::endl;
    }
}

void test_perfomance() {
    std::vector<float> a(N), o(1);

    std::cout << "CPU single core: ";
    benchmark_func([&] { vector_sum_native(a); });

    queue cpu_q{cpu_selector_v};
    queue gpu_q{gpu_selector_by_cu};
    std::vector<std::tuple<
        std::string,
        std::function<void(queue &, buffer<float, 1> &, buffer<float, 1> &)>,
        queue> > tests = {
        {"CPU SYCL", vector_sum, cpu_q},
        {"CPU SYCL ND-range", vector_sum_nd_range, cpu_q},
        {"CPU SYCL Reduction", vector_sum_reduction, cpu_q},
        {"GPU SYCL", vector_sum, gpu_q},
        {"GPU SYCL ND-range", vector_sum_nd_range, gpu_q},
        {"GPU SYCL Reduction", vector_sum_reduction, gpu_q}
    };

    for (auto &[name,kernel, q]: tests) {
        {
            std::fill(o.begin(), o.end(), 0);
            buffer<float, 1> a_buf(a.data(), range<1>(N)),
                    o_buf(o.data(), range<1>(1));

            std::cout << name << ": ";
            benchmark_sycl_kernel([&](queue &q) { kernel(q, a_buf, o_buf); }, q);
        }
    }
}


int main() {
    test_acc();
    test_perfomance();
}
