#pragma once
#include "rockchip-drm.h"
#include <vector>

namespace ggml_rockchip {

void rk_fill_lut(std::vector<uint64_t>& cmds, const int16_t* lut, size_t lut_size);
void rk_generate_silu_lut(int16_t* lut, size_t lut_size, float& inv_scale);

} // namespace ggml_rockchip
