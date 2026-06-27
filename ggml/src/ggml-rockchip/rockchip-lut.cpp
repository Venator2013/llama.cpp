#include "rockchip-lut.h"

namespace ggml_rockchip {

void rk_fill_lut(std::vector<uint64_t>& cmds, const int16_t* lut, size_t lut_size) {
    (void)cmds;
    (void)lut;
    (void)lut_size;
}

void rk_generate_silu_lut(int16_t* lut, size_t lut_size, float& inv_scale) {
    (void)lut;
    (void)lut_size;
    (void)inv_scale;
}

} // namespace ggml_rockchip
