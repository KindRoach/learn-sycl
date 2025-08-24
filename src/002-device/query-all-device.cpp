#include <sycl/device.hpp>

#include "util/device.hpp"


int main() {
    for (const auto &device: sycl::device::get_devices()) {
        std::cout << "Device: " << std::endl
                << "\tName: " << device.get_info<sycl::info::device::name>() << std::endl
                << "\tVendor: " << device.get_info<sycl::info::device::vendor>() << std::endl
                << "\tBackend: " << backend_to_string(device.get_backend()) << std::endl
                << "\tType: " << device_type_to_string(device.get_info<sycl::info::device::device_type>()) << std::endl
                << "\tCompute Unit: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl
                << "\tSLM Size(bytes): " << device.get_info<sycl::info::device::local_mem_size>() << std::endl;
        print_sub_groups(device);
    }
}
