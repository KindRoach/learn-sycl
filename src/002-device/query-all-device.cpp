#include <sycl/device.hpp>

std::string device_type_to_string(sycl::info::device_type type) {
    switch (type) {
        case sycl::info::device_type::cpu:
            return "CPU";
        case sycl::info::device_type::gpu:
            return "GPU";
        case sycl::info::device_type::accelerator:
            return "Accelerator";
        case sycl::info::device_type::custom:
            return "Custom";
        case sycl::info::device_type::host:
            return "Host";
        case sycl::info::device_type::all:
            return "All";
        default:
            return "Unknown";
    }
}

std::string backend_to_string(sycl::backend backend) {
    switch (backend) {
        case sycl::backend::opencl:
            return "OpenCL";
        case sycl::backend::ext_oneapi_cuda:
            return "CUDA";
        case sycl::backend::ext_oneapi_level_zero:
            return "Level-Zero";
        default:
            return "Unknown";
    }
}

void print_sub_groups(const sycl::device &d) {
    std::cout << "\tSubgroup sizes:";
    for (const auto &x: d.get_info<sycl::info::device::sub_group_sizes>()) {
        std::cout << " " << x;
    }
    std::cout << std::endl;
}


int main() {
    for (const auto &device: sycl::device::get_devices()) {
        std::cout << "Device: " << std::endl
                << "\tName: " << device.get_info<sycl::info::device::name>() << std::endl
                << "\tVendor: " << device.get_info<sycl::info::device::vendor>() << std::endl
                << "\tBackend: " << backend_to_string(device.get_backend()) << std::endl
                << "\tType: " << device_type_to_string(device.get_info<sycl::info::device::device_type>()) << std::endl
                << "\tCompute Unit: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
        print_sub_groups(device);
    }
}
