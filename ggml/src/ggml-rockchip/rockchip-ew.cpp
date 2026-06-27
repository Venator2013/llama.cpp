#include "rockchip-ew.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace ggml_rockchip {

static void emit_raw(std::vector<uint64_t>& q, uint32_t target, uint32_t reg, uint32_t value) {
    target = target + 0x1;
    uint64_t packed_value = ((uint64_t)(target & 0xFFFF) << 48) | ((uint64_t)(value & 0xFFFFFFFF) << 16) | (reg & 0xFFFF);
    q.push_back(packed_value);
}

void rk_run_elementwise(rk_device& dev, rk_ew_op op, 
                         uint64_t dma_src1, uint64_t dma_src2, 
                         uint64_t dma_dst, size_t size) {
    if (size == 0) return;

    // Kernel-Buffer für Submission allokieren
    rk_buffer task_buf = dev.alloc(1024, RKNPU_MEM_KERNEL_MAPPING, "task_buf");
    rk_buffer cmd_buf = dev.alloc(2048, 0, "cmd_buf");

    if (!task_buf.va || !cmd_buf.va) {
        std::fprintf(stderr, "ggml-rockchip: error: failed to allocate task or cmd buffer in rk_run_elementwise.\n");
        if (task_buf.va) dev.free(task_buf);
        if (cmd_buf.va) dev.free(cmd_buf);
        return;
    }

    // Dimensions-Setup
    uint32_t burst_len = 15;
    uint32_t output_mode = 2;
    uint32_t flying_mode = 1;
    uint32_t channel = 7;
    uint32_t dataout_height = 0;
    
    // NPU verarbeitet in 8er Channels. Daher:
    uint32_t dataout_width = (size + 7) / 8 - 1;

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

    // Operationen-Details bestimmen
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
            ew_op_cvt_bypass = 1; // Multiplikation umgeht Operator-Konvertierung
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

    std::vector<uint64_t> q;

    // 1. DPU Block Setup (Target: 4096 = DPU)
    uint32_t feat_mode_cfg = (burst_len << 5) | (output_mode << 1) | flying_mode;
    emit_raw(q, 4096, 0x400c, feat_mode_cfg);  // REG_DPU_FEATURE_MODE_CFG

    uint32_t data_format = (precision_float16 << 29) | (precision_float16 << 26) | precision_float16;
    emit_raw(q, 4096, 0x4010, data_format);    // REG_DPU_DATA_FORMAT

    uint32_t cube_channel = (channel << 16) | channel;
    emit_raw(q, 4096, 0x403c, cube_channel);   // REG_DPU_DATA_CUBE_CHANNEL
    emit_raw(q, 4096, 0x4030, dataout_width);  // REG_DPU_DATA_CUBE_WIDTH

    uint32_t ew_cfg = (ew_cvt_type << 31) | (ew_data_mode << 28) | (ew_data_size << 22) |
                      (ew_alu_algo << 16) | (ew_op_type << 2) | (ew_relu_bypass << 9) |
                      (ew_op_cvt_bypass << 8) | (ew_lut_bypass << 7) | (ew_op_src << 6) |
                      (ew_op_bypass << 1) | ew_bypass;
    emit_raw(q, 4096, 0x4070, ew_cfg);         // REG_DPU_EW_CFG

    emit_raw(q, 4096, 0x4020, dma_dst);        // REG_DPU_DST_BASE_ADDR

    // 2. DPU RDMA Block Setup (Target: 8192 = DPU_RDMA)
    emit_raw(q, 8192, 0x500c, dataout_width);   // REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH
    emit_raw(q, 8192, 0x5010, dataout_height);  // REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT
    emit_raw(q, 8192, 0x5014, channel);         // REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL

    uint32_t erdma_cfg = (1 << 30) | (erdma_data_size_16bit << 2);
    emit_raw(q, 8192, 0x5034, erdma_cfg);       // REG_DPU_RDMA_RDMA_ERDMA_CFG

    emit_raw(q, 8192, 0x5018, dma_src1);        // REG_DPU_RDMA_RDMA_SRC_BASE_ADDR
    emit_raw(q, 8192, 0x5038, dma_src2);        // REG_DPU_RDMA_RDMA_EW_BASE_ADDR

    // 3. Hardware-Submit Befehle
    q.push_back(0x2001000178495044);           // REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG (0x5044) = 0x00017849
    q.push_back(0x0081000000180008);           // REG_PC_OPERATION_ENABLE (0x0008) = 0x0018

    // Kopieren in den Command-Buffer
    std::memcpy(cmd_buf.va, q.data(), q.size() * sizeof(uint64_t));

    // Submit Task (Blockierender Modus)
    if (!dev.submit(task_buf, cmd_buf, q.size(), 1)) {
        std::fprintf(stderr, "ggml-rockchip: error: NPU submission failed in rk_run_elementwise.\n");
    }

    // Speicher wieder freigeben
    dev.free(task_buf);
    dev.free(cmd_buf);
}

} // namespace ggml_rockchip
