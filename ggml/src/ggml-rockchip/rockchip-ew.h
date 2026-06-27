#pragma once
#include "rockchip-drm.h"
#include <cstdint>
#include <cstddef>

namespace ggml_rockchip {

enum rk_ew_op {
    RK_EW_ADD = 0,
    RK_EW_MUL = 1,
    RK_EW_SUB = 2,
    RK_EW_MAX = 3
};

void rk_run_elementwise(rk_device& dev, rk_ew_op op, 
                         uint64_t dma_src1, uint64_t dma_src2, 
                         uint64_t dma_dst, size_t size);

} // namespace ggml_rockchip
