#include <sycl/sycl.hpp>

#include "util/device.hpp"

class MyKernel {
public:
    MyKernel(int val, sycl::stream out) : val(val), out(out) {
    };

    void operator()(sycl::id<1> i) const {
        out << "Value: " << val << ", Index: " << i.get(0) << sycl::endl;
    }

private:
    int val;
    sycl::stream out;
};

int main() {
    sycl::queue q{gpu_selector_by_cu};

    q.submit([](sycl::handler &h) {
        sycl::stream stream(65536, 256, h);
        MyKernel kernel(42, stream);
        h.parallel_for(8, kernel);
    }).wait();
}
