#include <sycl/sycl.hpp>

#include "cpp-bench-utils/utils.hpp"

inline std::string to_string(sycl::ext::oneapi::experimental::matrix::matrix_type type) {
    using namespace sycl::ext::oneapi::experimental::matrix;
    switch (type) {
        case matrix_type::bf16: return "bf16";
        case matrix_type::fp16: return "fp16";
        case matrix_type::tf32: return "tf32";
        case matrix_type::fp32: return "fp32";
        case matrix_type::fp64: return "fp64";
        case matrix_type::sint8: return "sint8";
        case matrix_type::sint16: return "sint16";
        case matrix_type::sint32: return "sint32";
        case matrix_type::sint64: return "sint64";
        case matrix_type::uint8: return "uint8";
        case matrix_type::uint16: return "uint16";
        case matrix_type::uint32: return "uint32";
        case matrix_type::uint64: return "uint64";
        default: throw std::invalid_argument("Unknown matrix_type");
    }
}

int main() {
    namespace matrix = sycl::ext::oneapi::experimental::matrix;
    namespace info = sycl::ext::oneapi::experimental::info;
    sycl::queue q{cbu::gpu_selector_by_cu};
    std::vector<matrix::combination> combinations = q.get_device().get_info<info::device::matrix_combinations>();

    if (combinations.empty()) {
        std::cout << "No XMX found." << std::endl;
        return 0;
    }

    for (auto &comb: combinations) {
        std::cout << "type a,b,c,d=" << to_string(comb.atype) << "," << to_string(comb.btype) << "," <<
                to_string(comb.ctype) << "," << to_string(comb.dtype) << ",";
        std::cout << "m,k,n=" << comb.msize << "," << comb.ksize << "," << comb.nsize << "," << std::endl;
    }
}
