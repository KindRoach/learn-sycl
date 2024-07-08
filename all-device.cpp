#include <sycl/device.hpp>

int main() {
    for (const auto &device: sycl::device::get_devices()) {
        std::cout << "Device: "
                << "\t" << device.get_info<sycl::info::device::name>() << std::endl
                << "\t" << device.get_info<sycl::info::device::vendor>() << std::endl;
    }
}
