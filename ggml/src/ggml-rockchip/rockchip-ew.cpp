#include "rockchip-ew.h"
#include "rockchip-regs.h"
#include <vector>
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace ggml_rockchip {

const uint32_t RKNPU_TARGET_DPU = 4096;
const uint32_t RKNPU_TARGET_DPU_RDMA = 8192;

bool rk_run_elementwise(rk_device& dev, rk_ew_op op, 
                         uint64_t dma_src1, uint64_t dma_src2, 
                         uint64_t dma_dst, size_t size) {
    if (size == 0) return true;

    if (!dev.task_buf.va || !dev.cmd_buf.va) {
        std::fprintf(stderr, "ggml-rockchip: error: persistent DRM control buffers are not allocated.\n");
        return false;
    }

    const size_t MAX_ELEMENTS_PER_SUBMISSION = 4096;
    size_t processed = 0;

    std::vector<uint64_t> q;
    q.reserve(16);

    while (processed < size) {
        size_t chunk_size = std::min(size - processed, MAX_ELEMENTS_PER_SUBMISSION);

        uint64_t chunk_src1 = dma_src1 + processed * sizeof(uint16_t);
        uint64_t chunk_src2 = dma_src2 + processed * sizeof(uint16_t);
        uint64_t chunk_dst  = dma_dst  + processed * sizeof(uint16_t);

        // Dimensions setup
        uint32_t burst_len = 15;
        uint32_t output_mode = 2;
        uint32_t flying_mode = 1;
        uint32_t channel = 7;
        uint32_t dataout_height = 0;
        
        uint32_t dataout_width = (chunk_size + 7) / 8 - 1;

        uint32_t precision_float16 = 2;
        uint32_t ew_cvt_type = 0;
        uint32_t ew_data_mode = 1;
        uint32_t ew_data_size = 2; // FP16
        uint32_t ew_relu_bypass = 1;
        uint32_t ew_op_cvt_bypass = 0;
        uint32_t ew_lut_bypass = 1;
        uint32_t ew_op_src = 1;
        uint32_t ew_op_bypass = 0;
        uint32_t ew_bypass = 0;

        // Determine operation details
        uint32_t ew_alu_algo = 0;
        uint32_t ew_op_type = 0;

        switch (op) {
            case RK_EW_ADD:
                ew_alu_algo = 2;
                ew_op_type = 0;
                break;
            case RK_EW_MUL:
                ew_alu_algo = 0;
                ew_op_type = 1;
                ew_op_cvt_bypass = 1; // Multiplication bypasses operator conversion
                break;
            case RK_EW_SUB:
                ew_alu_algo = 4;
                ew_op_type = 0;
                break;
            case RK_EW_MAX:
                ew_alu_algo = 0;
                ew_op_type = 0;
                break;
        }

        uint32_t erdma_data_size_16bit = 2;

        q.clear();

        // 1. DPU Block Setup (Target: 4096 = DPU)
        uint32_t feat_mode_cfg = (burst_len << 5) | (output_mode << 1) | flying_mode;
        emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_FEATURE_MODE_CFG, feat_mode_cfg);

        uint32_t data_format = (precision_float16 << 29) | (precision_float16 << 26) | precision_float16;
        emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DATA_FORMAT, data_format);

        uint32_t cube_channel = (channel << 16) | channel;
        emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DATA_CUBE_CHANNEL, cube_channel);
        emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DATA_CUBE_WIDTH, dataout_width);

        uint32_t ew_cfg = (ew_cvt_type << 31) | (ew_data_mode << 28) | (ew_data_size << 22) |
                          (ew_alu_algo << 16) | (ew_op_type << 2) | (ew_relu_bypass << 9) |
                          (ew_op_cvt_bypass << 8) | (ew_lut_bypass << 7) | (ew_op_src << 6) |
                          (ew_op_bypass << 1) | ew_bypass;
        emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_EW_CFG, ew_cfg);

        emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DST_BASE_ADDR, chunk_dst);

        // 2. DPU RDMA Block Setup (Target: 8192 = DPU_RDMA)
        emit_raw(q, RKNPU_TARGET_DPU_RDMA, REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, dataout_width);
        emit_raw(q, RKNPU_TARGET_DPU_RDMA, REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, dataout_height);
        emit_raw(q, RKNPU_TARGET_DPU_RDMA, REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, channel);

        uint32_t erdma_cfg = (1 << 30) | (erdma_data_size_16bit << 2);
        emit_raw(q, RKNPU_TARGET_DPU_RDMA, REG_DPU_RDMA_RDMA_ERDMA_CFG, erdma_cfg);

        emit_raw(q, RKNPU_TARGET_DPU_RDMA, REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, chunk_src1);
        emit_raw(q, RKNPU_TARGET_DPU_RDMA, REG_DPU_RDMA_RDMA_EW_BASE_ADDR, chunk_src2);

        // 3. Hardware Submit Commands
        q.push_back(0x2001000178495044); // REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG (0x5044) = 0x00017849
        q.push_back(0x0081000000180008); // REG_PC_OPERATION_ENABLE (0x0008) = 0x0018

        // Copy to DRM Command Buffer
        std::memcpy(dev.cmd_buf.va, q.data(), q.size() * sizeof(uint64_t));

        // Submit task (blocking mode)
        if (!dev.submit(dev.task_buf, dev.cmd_buf, q.size(), 1)) {
            std::fprintf(stderr, "ggml-rockchip: error: NPU submission failed in rk_run_elementwise.\n");
            return false;
        }

        processed += chunk_size;
    }

    return true;
}

} // namespace ggml_rockchip
