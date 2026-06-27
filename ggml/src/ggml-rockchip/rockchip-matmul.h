#pragma once
#include "rockchip-drm.h"
#include "ggml.h"

namespace ggml_rockchip {

void rk_compute_matmul(rk_device& dev, struct ggml_tensor* op);

} // namespace ggml_rockchip
