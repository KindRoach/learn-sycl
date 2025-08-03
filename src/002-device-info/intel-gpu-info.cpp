#include <sycl/sycl.hpp>

#include "device-util.hpp"

int main()
{

    for (const auto &device : sycl::device::get_devices())
    {
        if (device.get_info<sycl::info::device::device_type>() != sycl::info::device_type::gpu)
            continue;

        sycl::queue q(device);

        auto device_name = q.get_device().get_info<sycl::info::device::name>();
        auto numSlices = q.get_device().get_info<sycl::ext::intel::info::device::gpu_slices>();
        auto numSubslicesPerSlice = q.get_device().get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
        auto numEUsPerSubslice = q.get_device().get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>();
        auto numThreadsPerEU = q.get_device().get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();
        auto global_mem_size = q.get_device().get_info<sycl::info::device::global_mem_size>();
        auto local_mem_size = q.get_device().get_info<sycl::info::device::local_mem_size>();
        auto max_work_group_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
        auto sub_group_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();

        std::cout << "Intel GPU Characteristics:\n";
        std::cout << "\tGPU Model : " << device_name << "\n";
        std::cout << "\tBackend: " << backend_to_string(device.get_backend()) << "\n";
        std::cout << "\tXeCore count : " << numSlices * numSubslicesPerSlice << "\n";
        std::cout << "\tVector Engines per XeCore : " << numEUsPerSubslice << "\n";
        std::cout << "\tVector Engine count : " << numSlices * numSubslicesPerSlice * numEUsPerSubslice << "\n";
        std::cout << "\tHardware Threads per Vector Engine : " << numThreadsPerEU << "\n";
        std::cout << "\tHardware Threads count : " << numSlices * numSubslicesPerSlice * numEUsPerSubslice * numThreadsPerEU << "\n";
        std::cout << "\tGPU Memory Size : " << global_mem_size << "\n";
        std::cout << "\tShared Local Memory per Work-group : " << local_mem_size << "\n";
        std::cout << "\tMax Work-group size : " << max_work_group_size << "\n";
        print_sub_groups(device);
    }
}