#include <sycl/sycl.hpp>

#include "cpp-bench-utils/utils.hpp"

namespace xmx = sycl::ext::oneapi::experimental::matrix;

template <typename dtype, typename acc_type, xmx::layout b_layout>
void matrix_multiply_ref(
    std::vector<dtype>& a,
    std::vector<dtype>& b,
    std::vector<acc_type>& c,
    size_t m, size_t n, size_t k)
{
    using namespace cbu;
    size_t lda = k, ldb = b_layout == xmx::layout::row_major ? n : k, ldc = n;
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            acc_type sum = 0;
            for (size_t p = 0; p < k; p++)
            {
                acc_type a_ele = static_cast<acc_type>(mat(a.data(), lda, i, p));
                acc_type b_ele = static_cast<acc_type>(mat(b.data(), ldb,
                                                           b_layout == xmx::layout::row_major ? p : j,
                                                           b_layout == xmx::layout::row_major ? j : p));
                sum += a_ele * b_ele;
            }
            mat(c.data(), ldc, i, j) = sum;
        }
    }
}

template <typename KernelName>
size_t get_sg_size(sycl::queue& q)
{
    auto KernelID = sycl::get_kernel_id<KernelName>();
    auto KB = get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {KernelID});
    auto kernel = KB.get_kernel(KernelID);
    return kernel.template get_info<sycl::info::kernel_device_specific::max_sub_group_size>(q.get_device());
}

template <xmx::layout b_layout>
struct matrix_multiply_joint_kernel;

template <typename dtype, typename acc_type, xmx::layout b_layout, size_t WG_T_NUM, size_t TM, size_t TN, size_t TK>
void matrix_multiply_joint(sycl::queue& q, dtype* a, dtype* b, acc_type* c, size_t m, size_t n, size_t k)
{
    using namespace cbu;
    check_divisible(m, TM * WG_T_NUM, "M must be divisible by TM * WG_T_NUM");
    check_divisible(n, TN * WG_T_NUM, "N must be divisible by TN * WG_T_NUM");
    check_divisible(k, TK, "K must be divisible by TK");

    using kernel_name = matrix_multiply_joint_kernel<b_layout>;

    size_t lda = k, ldb = b_layout == xmx::layout::row_major ? n : k, ldc = n;
    size_t sg_size = get_sg_size<kernel_name>(q);
    sycl::range<2> local = {WG_T_NUM, WG_T_NUM * sg_size}; // WG_T_NUM * WG_T_NUM tiles
    sycl::range<2> global = {m / (TM * WG_T_NUM), n / (TN * WG_T_NUM)};
    global *= local;

    q.parallel_for<kernel_name>(
        sycl::nd_range<2>{global, local},
        [=](sycl::nd_item<2> item)
        {
            sycl::sub_group sg = item.get_sub_group();
            size_t g_i = item.get_global_id(0);
            size_t g_j = item.get_global_id(1) / sg_size;

            using mp = sycl::multi_ptr<dtype, sycl::access::address_space::global_space>;
            using mp_acc = sycl::multi_ptr<acc_type, sycl::access::address_space::global_space>;
            xmx::joint_matrix<sycl::sub_group, dtype, xmx::use::a, TM, TK, xmx::layout::row_major> tile_a;
            xmx::joint_matrix<sycl::sub_group, dtype, xmx::use::b, TK, TN, b_layout> tile_b;
            xmx::joint_matrix<sycl::sub_group, acc_type, xmx::use::accumulator, TM, TN> tile_c;

            xmx::joint_matrix_fill(sg, tile_c, 0);
            for (size_t kk = 0; kk < k; kk += TK)
            {
                auto pA = mp(mat_ptr(a, lda, g_i * TM, kk));
                auto pB = mp(mat_ptr(b, ldb,
                                     b_layout == xmx::layout::row_major ? kk : g_j * TN,
                                     b_layout == xmx::layout::row_major ? g_j * TN : kk));

                joint_matrix_load(sg, tile_a, pA, lda);
                joint_matrix_load(sg, tile_b, pB, ldb);
                joint_matrix_mad(sg, tile_c, tile_a, tile_b, tile_c);
            }

            auto pC = mp_acc(mat_ptr(c, ldc, g_i * TM, g_j * TN));
            joint_matrix_store(sg, tile_c, pC, ldc, xmx::layout::row_major);
        }).wait();
}

template <xmx::layout b_layout>
void test_matrix_multiply()
{
    using namespace cbu;
    std::string b_major = b_layout == xmx::layout::row_major ? "row major" : "col major";
    std::cout << "-------------- matrix b in " << b_major << " --------------\n";

    using dtype = sycl::half;
    using acc_type = float;

    size_t secs = 10;
    size_t m = 2 * 1024, n = 512, k = 1024;

    std::vector<dtype> a(m * k), b(k * n);
    std::vector<acc_type> c(m * n);
    random_fill(a);
    random_fill(b);

    sycl::queue q{gpu_selector_by_cu, sycl::property::queue::in_order()};
    auto* d_a = sycl::malloc_device<dtype>(a.size(), q);
    auto* d_b = sycl::malloc_device<dtype>(b.size(), q);
    auto* d_c = sycl::malloc_device<acc_type>(c.size(), q);
    q.memcpy(d_a, a.data(), a.size() * sizeof(dtype)).wait();
    q.memcpy(d_b, b.data(), b.size() * sizeof(dtype)).wait();

    std::cout << "matrix_multiply_ref:\n";
    benchmark_func_by_time(secs, [&]()
    {
        matrix_multiply_ref<dtype, acc_type, b_layout>(a, b, c, m, n, k);
    });

    using func_t = std::function<void(sycl::queue&, dtype*, dtype*, acc_type*, size_t, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t>> funcs{
        {"matrix_multiply_joint", matrix_multiply_joint<dtype, acc_type, b_layout, 4, 16, 16, 16>},
    };

    for (auto [func_name,func] : funcs)
    {
        std::cout << "\n" << func_name << ":\n";
        q.fill(d_c, dtype{0}, c.size()).wait();
        benchmark_func_by_time(secs, [&]()
        {
            func(q, d_a, d_b, d_c, m, n, k);
            q.wait();
        });
        sycl_acc_check(q, c, d_c);
    }

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);
}

int main()
{
    test_matrix_multiply<xmx::layout::row_major>();
    test_matrix_multiply<xmx::layout::col_major>();
}
