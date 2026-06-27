#pragma once
#include "rockchip-drm.h"
#include "ggml.h"

namespace ggml_rockchip {

struct wmma_params {
    size_t m, n, k;
    size_t align_in;
    size_t align_out;
    size_t data_in_width;
    size_t data_in_height;
    size_t dataout_width;
    size_t dataout_height;
    size_t feature_grains;
    size_t weight_bytes_per_kernel;
    size_t data_bank;
    size_t line_stride;
    size_t surf_stride;
    size_t dst_surf_stride;
    size_t notch_val;
};

void rk_compute_matmul(rk_device& dev, struct ggml_tensor* op);

} // namespace ggml_rockchip
