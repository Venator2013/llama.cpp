#pragma once
#include "rockchip-drm.h"
#include "ggml.h"

namespace ggml_rockchip {

enum rk_ew_op {
    RK_EW_ADD = 0,
    RK_EW_MUL = 1,
    RK_EW_SUB = 2,
    RK_EW_MAX = 3
};

void rk_run_elementwise(rk_device& dev, rk_ew_op op, 
                         const void* lhs, const void* rhs, 
                         void* dst, size_t count);

} // namespace ggml_rockchip
