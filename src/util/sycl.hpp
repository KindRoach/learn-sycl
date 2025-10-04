#pragma once

#include <sycl/sycl.hpp>

#include "bench.hpp"
#include "util/vector.hpp"

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

void print_sub_groups(const sycl::device &d) {
    std::cout << "\tSubgroup sizes:";
    for (const auto &x: d.get_info<sycl::info::device::sub_group_sizes>()) {
        std::cout << " " << x;
    }
    std::cout << std::endl;
}

inline int gpu_selector_by_cu(const sycl::device &dev) {
    int priority = 0;

    if (dev.is_gpu()) {
        unsigned int cu = dev.get_info<sycl::info::device::max_compute_units>();
        priority += static_cast<int>(cu);
    }

    if (dev.get_backend() == sycl::backend::ext_oneapi_level_zero) {
        priority += 1;
    }

    return priority;
}

template<typename T>
void sycl_acc_check(sycl::queue &q, std::vector<T> &gt, T *device_ptr) {
    size_t size = gt.size();
    std::vector<T> actual(size);
    q.memcpy(actual.data(), device_ptr, size * sizeof(T)).wait();
    acc_check(gt, actual);
}

void sycl_print_item_info(sycl::nd_item<1> it) {
    size_t group_range = it.get_group_range(0);
    size_t local_range = it.get_local_range(0);

    size_t group_id = it.get_group(0);
    size_t local_id = it.get_local_id(0);
    size_t global_id = it.get_global_id(0);

    auto sg = it.get_sub_group();
    size_t sg_size = sg.get_local_range()[0];
    size_t sg_group_id = sg.get_group_id()[0];
    size_t sg_local_id = sg.get_local_id()[0];

    sycl::ext::oneapi::experimental::printf(
        "nd_range=[%zu, %zu];"
        " global_id=%zu;"
        " g_id=%zu;"
        " l_id=%zu;"
        " sg_group_id=%zu;"
        " sg_local_id=%zu;"
        " sg_size=%zu\n",
        group_range, local_range,
        global_id,
        group_id,
        local_id,
        sg_group_id,
        sg_local_id,
        sg_size
    );
}


void sycl_print_item_info(sycl::nd_item<2> it) {
    size_t group_range_x = it.get_group_range(0);
    size_t group_range_y = it.get_group_range(1);

    size_t local_range_x = it.get_local_range(0);
    size_t local_range_y = it.get_local_range(1);

    size_t group_id_x = it.get_group(0);
    size_t group_id_y = it.get_group(1);

    size_t local_id_x = it.get_local_id(0);
    size_t local_id_y = it.get_local_id(1);

    size_t global_id_x = it.get_global_id(0);
    size_t global_id_y = it.get_global_id(1);

    auto sg = it.get_sub_group();
    size_t sg_size = sg.get_local_range()[0];
    size_t sg_group_id = sg.get_group_id()[0];
    size_t sg_local_id = sg.get_local_id()[0];

    sycl::ext::oneapi::experimental::printf(
        "nd_range=[(%zux%zu),(%zux%zu)];"
        " global_id=(%zu,%zu);"
        " g_id=(%zu,%zu);"
        " l_id=(%zu,%zu);"
        " sg_group_id=%zu;"
        " sg_local_id=%zu;"
        " sg_size=%zu\n",
        group_range_x, group_range_y, local_range_x, local_range_y,
        global_id_x, global_id_y,
        group_id_x, group_id_y,
        local_id_x, local_id_y,
        sg_group_id,
        sg_local_id,
        sg_size
    );
}

using sycl_kernel = std::function<void(sycl::queue &)>;

void benchmark_sycl_kernel(
    int num_iter, sycl::queue &queue,
    const sycl_kernel &submitKernel,
    double warmup_ratio = 0.1
) {
    if (num_iter <= 1) {
        std::cerr << "Warning: num_iter less than 2, running kernel once.\n";
        submitKernel(queue);
        queue.wait();
        return;
    }

    // Warm-up phase
    int warm_up_iter = std::max(1, static_cast<int>(num_iter * warmup_ratio));
    for (int i = 0; i < warm_up_iter; ++i) { submitKernel(queue); }
    queue.wait();

    // benchmark phase
    num_iter -= warm_up_iter;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iter; ++i) {
        submitKernel(queue);
    }
    queue.wait();
    auto end = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    print_human_readable_time_usage(num_iter, totalDuration);
}
