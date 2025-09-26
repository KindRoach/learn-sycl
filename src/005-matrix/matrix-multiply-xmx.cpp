#include <sycl/sycl.hpp>

#include "util/util.hpp"

template<typename dtype, typename acc_type>
void matrix_multiply_ref(
    std::vector<dtype> &a,
    std::vector<dtype> &b,
    std::vector<acc_type> &c,
    size_t m, size_t n, size_t k) {
    size_t lda = k, ldb = n, ldc = n;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            acc_type sum = 0;
            for (size_t p = 0; p < k; p++) {
                auto a_i_p = static_cast<acc_type>(mat(a.data(), lda, i, p));
                auto b_p_j = static_cast<acc_type>(mat(b.data(), ldb, p, j));
                sum += a_i_p * b_p_j;
            }
            mat(c.data(), ldc, i, j) = sum;
        }
    }
}

template<typename KernelName>
size_t get_sg_size(sycl::queue &q) {
    auto KernelID = sycl::get_kernel_id<KernelName>();
    auto KB = get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {KernelID});
    auto kernel = KB.get_kernel(KernelID);
    return kernel.template get_info<sycl::info::kernel_device_specific::max_sub_group_size>(q.get_device());
}

struct joint_matrix_kernel;

template<typename dtype, typename acc_type, size_t WG_T_NUM, size_t TM, size_t TN, size_t TK>
void matrix_multiply_joint(sycl::queue &q, dtype *a, dtype *b, acc_type *c, size_t m, size_t n, size_t k) {
    using namespace sycl::ext::oneapi::experimental::matrix;

    size_t sg_size = get_sg_size<joint_matrix_kernel>(q);
    sycl::range<2> local = {WG_T_NUM, WG_T_NUM * sg_size};
    sycl::range<2> global = {m / (TM * WG_T_NUM), n / (TN * WG_T_NUM)};
    global *= local;

    q.parallel_for<joint_matrix_kernel>(
        sycl::nd_range<2>{global, local},
        [=](sycl::nd_item<2> item) {
            sycl::sub_group sg = item.get_sub_group();
            size_t g_i = item.get_global_id(0);
            size_t g_j = item.get_global_id(1) / sg_size;
            size_t sg_start_i = g_i * TM;
            size_t sg_start_j = g_j * TN;

            auto pA = sycl::multi_ptr<dtype, sycl::access::address_space::global_space>(a);
            auto pB = sycl::multi_ptr<dtype, sycl::access::address_space::global_space>(b);
            auto pC = sycl::multi_ptr<acc_type, sycl::access::address_space::global_space>(c);

            joint_matrix<sycl::sub_group, dtype, use::a, TM, TK, layout::row_major> tile_a;
            joint_matrix<sycl::sub_group, dtype, use::b, TK, TN, layout::row_major> tile_b;
            joint_matrix<sycl::sub_group, acc_type, use::accumulator, TM, TN> tile_c;

            joint_matrix_fill(sg, tile_c, 0);
            for (size_t kk = 0; kk < k; kk += TK) {
                joint_matrix_load(sg, tile_a, pA + sg_start_i * k + kk, k);
                joint_matrix_load(sg, tile_b, pB + kk * n + sg_start_j, n);
                joint_matrix_mad(sg, tile_c, tile_a, tile_b, tile_c);
            }

            joint_matrix_store(sg, tile_c, pC + sg_start_i * n + sg_start_j, n, layout::row_major);
        }).wait();
}

int main() {
    using dtype = sycl::half;
    using acc_type = float;

    size_t secs = 0;
    size_t m = 1024, n = 1024, k = 1024; // 2G FLOPs

    std::vector<dtype> a(m * k), b(k * n);
    std::vector<acc_type> c(m * n);
    random_fill(a);
    random_fill(b);

    sycl::queue q{gpu_selector_by_cu, sycl::property::queue::in_order()};
    auto *d_a = sycl::malloc_device<dtype>(a.size(), q);
    auto *d_b = sycl::malloc_device<dtype>(b.size(), q);
    auto *d_c = sycl::malloc_device<acc_type>(c.size(), q);
    q.memcpy(d_a, a.data(), a.size() * sizeof(dtype)).wait();
    q.memcpy(d_b, b.data(), b.size() * sizeof(dtype)).wait();

    std::cout << "matrix_multiply_ref:\n";
    benchmark_func_by_time(secs, [&]() {
        matrix_multiply_ref<dtype>(a, b, c, m, n, k);
    });

    using func_t = std::function<void(sycl::queue &, dtype *, dtype *, acc_type *, size_t, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {"matrix_multiply_joint", matrix_multiply_joint<dtype, acc_type, 4, 16, 16, 16>},
    };

    for (auto [func_name,func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        q.fill(d_c, dtype{0}, c.size()).wait();
        benchmark_func_by_time(secs, [&]() {
            func(q, d_a, d_b, d_c, m, n, k);
            q.wait();
        });
        sycl_acc_check(q, c, d_c);
    }
}
