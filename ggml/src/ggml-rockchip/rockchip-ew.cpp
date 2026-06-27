#include "rockchip-ew.h"

namespace ggml_rockchip {

void rk_run_elementwise(rk_device& dev, rk_ew_op op, 
                         const void* lhs, const void* rhs, 
                         void* dst, size_t count) {
    (void)dev;
    (void)op;
    (void)lhs;
    (void)rhs;
    (void)dst;
    (void)count;
}

} // namespace ggml_rockchip
