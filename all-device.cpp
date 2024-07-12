#include <sycl/device.hpp>

int main() {
    for (const auto &device: sycl::device::get_devices()) {
        std::cout << "Device: " << std::endl
                << "\tName: " << device.get_info<sycl::info::device::name>() << std::endl
                << "\tVendor: " << device.get_info<sycl::info::device::vendor>() << std::endl
                << "\tCompute Unit: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    }
}
