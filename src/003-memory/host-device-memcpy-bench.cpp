#include <iostream>
#include <sycl/sycl.hpp>

#include "cpp-bench-utils/utils.hpp"

template <typename T>
void bench_memcpy(sycl::queue& q, size_t size)
{
    using namespace cbu;

    std::vector<T> host_vec(size);
    random_fill(host_vec);

    auto* device_vec = sycl::malloc_device<T>(size, q);

    BenchmarkOptions opt{
        .total_mem_bytes = size * sizeof(T),
    };

    float size_mb = static_cast<float>(size * sizeof(T)) / (1024.0f * 1024.0f);
    std::cout << "\n========== Data size: " << size_mb << " MB ==========\n";

    std::cout << "\nbench_memcpy - host to device:\n";
    benchmark_func_by_time(10, [&]()
    {
        q.memcpy(device_vec, host_vec.data(), size * sizeof(T)).wait();
        q.wait();
    }, opt);

    std::cout << "\nbench_memcpy - device to host:\n";
    benchmark_func_by_time(10, [&]()
    {
        q.memcpy(host_vec.data(), device_vec, size * sizeof(T)).wait();
        q.wait();
    }, opt);

    sycl::free(device_vec, q);
}

int main(int argc, char* argv[])
{
    sycl::queue q{cbu::gpu_selector_by_cu, sycl::property::queue::in_order()};
    constexpr size_t mb = 1024 * 1024;
    for (auto size : {1 * mb, 16 * mb, 128 * mb, 1024 * mb})
    {
        bench_memcpy<float>(q, size / sizeof(float));
    }
}
