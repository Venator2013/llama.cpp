#include "rockchip-drm.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

const uint32_t RKNPU_TARGET_CNA = 512;
const uint32_t RKNPU_TARGET_CORE = 2048;
const uint32_t RKNPU_TARGET_DPU = 4096;
const uint32_t RKNPU_TARGET_DPU_RDMA = 8192;

// Registers
#define REG_DPU_S_POINTER 0x4004
#define REG_CNA_CONV_CON1 0x100c
#define REG_CNA_CONV_CON2 0x1010
#define REG_CNA_CONV_CON3 0x1014
#define REG_CNA_DATA_SIZE0 0x1020
#define REG_CNA_DATA_SIZE1 0x1024
#define REG_CNA_DATA_SIZE2 0x1028
#define REG_CNA_DATA_SIZE3 0x102c
#define REG_CNA_WEIGHT_SIZE0 0x1030
#define REG_CNA_WEIGHT_SIZE1 0x1034
#define REG_CNA_WEIGHT_SIZE2 0x1038
#define REG_CNA_CBUF_CON0 0x1040
#define REG_CNA_CBUF_CON1 0x1044
#define REG_CNA_CVT_CON0 0x104c
#define REG_CNA_CVT_CON1 0x1050
#define REG_CNA_CVT_CON2 0x1054
#define REG_CNA_CVT_CON3 0x1058
#define REG_CNA_CVT_CON4 0x105c
#define REG_CNA_FEATURE_DATA_ADDR 0x1070
#define REG_CNA_DMA_CON0 0x1078
#define REG_CNA_DMA_CON1 0x107c
#define REG_CNA_DMA_CON2 0x1080
#define REG_CNA_FC_DATA_SIZE0 0x1084
#define REG_CNA_FC_DATA_SIZE1 0x1088
#define REG_CNA_DCOMP_ADDR0 0x1110
#define REG_CORE_MISC_CFG 0x3010
#define REG_CORE_DATAOUT_SIZE_0 0x3014
#define REG_CORE_DATAOUT_SIZE_1 0x3018
#define REG_DPU_FEATURE_MODE_CFG 0x400c
#define REG_DPU_DATA_FORMAT 0x4010
#define REG_DPU_DST_BASE_ADDR 0x4020
#define REG_DPU_DST_SURF_STRIDE 0x4024
#define REG_DPU_DATA_CUBE_WIDTH 0x4030
#define REG_DPU_DATA_CUBE_HEIGHT 0x4034
#define REG_DPU_DATA_CUBE_NOTCH_ADDR 0x4038
#define REG_DPU_DATA_CUBE_CHANNEL 0x403c
#define REG_DPU_BS_CFG 0x4040
#define REG_DPU_BS_OW_CFG 0x4050
#define REG_DPU_WDMA_SIZE_0 0x4058
#define REG_DPU_WDMA_SIZE_1 0x405c
#define REG_DPU_BN_CFG 0x4060
#define REG_DPU_EW_CFG 0x4070
#define REG_DPU_SURFACE_ADD 0x40c0

static float fp16_to_float(uint16_t val) {
    uint32_t sign = (val & 0x8000) << 16;
    int32_t exponent = ((val & 0x7c00) >> 10) - 15 + 127;
    uint32_t mantissa = (val & 0x03ff) << 13;
    if ((val & 0x7c00) == 0) {
        if ((val & 0x03ff) == 0) {
            exponent = 0;
        } else {
            exponent = 127 - 14;
            while ((mantissa & 0x00800000) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x007fffff;
        }
    } else if ((val & 0x7c00) == 0x7c00) {
        exponent = 255;
    }
    union { float f; uint32_t u; } u;
    u.u = sign | (exponent << 23) | mantissa;
    return u.f;
}

static uint16_t float_to_fp16(float val) {
    union { float f; uint32_t u; } u = { val };
    uint32_t sign = (u.u >> 16) & 0x8000;
    int32_t exponent = ((u.u >> 23) & 0xff) - 127;
    uint32_t mantissa = u.u & 0x007fffff;
    if (exponent <= -15) return sign;
    if (exponent >= 16) return sign | 0x7c00;
    exponent += 15;
    return sign | (exponent << 10) | (mantissa >> 13);
}

int main() {
    using namespace ggml_rockchip;
    std::printf("--- Starting RKNPU DRM/ioctl 64x64x64 MatMul Test ---\n");

    rk_device dev;
    if (!dev.init()) {
        std::fprintf(stderr, "Error: failed to initialize NPU device.\n");
        return 1;
    }

    size_t m = 16, n = 16, k = 256;
    size_t align_in = 256;
    size_t align_out = 32;

    size_t m_padded = 64;
    size_t data_in_width = 1;
    size_t line_stride = 4;
    size_t surf_stride = 60;
    size_t dst_surf_stride = 64;
    size_t notch_val = 0;
    size_t datacube_width = 1;
    size_t datacube_height = m_padded;
    size_t weight_bytes_per_kernel = align_in * 2;
    size_t wt_size_0 = weight_bytes_per_kernel * align_out;
    size_t data_bank = 2;
    size_t dataout_width = 1;

    std::vector<uint16_t> in_pack(m_padded * align_in, 0);
    std::vector<uint16_t> wt_pack(align_out * align_in, 0);

    // Populate features using c2_8 tiled packing
    for (size_t r = 0; r < m; ++r) {
        for (size_t col = 0; col < k; ++col) {
            size_t in_idx = (col / 8) * (m_padded * 8) + r * 8 + (col % 8);
            float val = ((r * 17 + col * 7) % 100) * 0.01f;
            in_pack[in_idx] = float_to_fp16(val);
        }
    }
    for (size_t r = 0; r < n; ++r) {
        for (size_t col = 0; col < k; ++col) {
            size_t wt_idx = (r / 16) * (align_in / 32) * 512 + (col / 32) * 512 + (r % 16) * 32 + (col % 32);
            float val = ((col * 13 + r * 19) % 100) * 0.01f;
            wt_pack[wt_idx] = float_to_fp16(val);
        }
    }

    std::printf("Debug initialization check:\n");
    std::printf("  wt_pack populated for r=0..15 col=0:\n");
    for (size_t r = 0; r < 16; ++r) {
        size_t wt_idx = (r / 16) * (align_in / 32) * 512 + (0 / 32) * 512 + (r % 16) * 32 + (0 % 32);
        std::printf("    r=%zu wt_idx=%zu: %f\n", r, wt_idx, fp16_to_float(wt_pack[wt_idx]));
    }

    size_t in_bytes = in_pack.size() * sizeof(uint16_t);
    size_t wt_bytes = wt_pack.size() * sizeof(uint16_t);
    size_t out_bytes = align_out * m_padded * sizeof(float);

    std::printf("Allocating GEM buffers...\n");
    rk_buffer task_buf = dev.alloc(1024, RKNPU_MEM_KERNEL_MAPPING, "task_buf");
    rk_buffer cmd_buf = dev.alloc(2048, 0, "cmd_buf");
    rk_buffer input_buf = dev.alloc(in_bytes, 0, "input_buf");
    rk_buffer weight_buf = dev.alloc(wt_bytes, 0, "weight_buf");
    rk_buffer output_buf = dev.alloc(out_bytes, 0, "output_buf");

    if (!task_buf.va || !cmd_buf.va || !input_buf.va || !weight_buf.va || !output_buf.va) {
        std::fprintf(stderr, "Error: buffer allocation failed.\n");
        return 1;
    }

    std::memcpy(input_buf.va, in_pack.data(), in_bytes);
    std::memcpy(weight_buf.va, wt_pack.data(), wt_bytes);
    std::memset(output_buf.va, 0, out_bytes);

    std::vector<uint64_t> q;
    q.reserve(64);

    uint32_t s_pointer = (1 << 3) | (1 << 2) | (1 << 1);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_S_POINTER, s_pointer);

    uint32_t conv_con1 = (2 << 7) | (2 << 4);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CONV_CON1, conv_con1);

    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CONV_CON2, (m_padded + 1) << 4);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CONV_CON3, (1 << 3) | 1);

    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DATA_SIZE0, (data_in_width << 16) | m_padded);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DATA_SIZE1, ((align_in - 1) << 16) | align_in);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DATA_SIZE2, dataout_width);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DATA_SIZE3, dataout_width * m_padded);

    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DMA_CON0, (15 << 16) | 15);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DMA_CON1, line_stride);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DMA_CON2, surf_stride);

    uint32_t cbuf_con0 = ((12 - data_bank) << 4) | data_bank;
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CBUF_CON0, cbuf_con0);

    uint32_t cbuf_entries = ((data_in_width * align_in + 31) / 32);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CBUF_CON1, cbuf_entries);

    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CVT_CON0, 0xB);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CVT_CON1, 1 << 16);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CVT_CON2, 1 << 16);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CVT_CON3, 1 << 16);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CVT_CON4, 1 << 16);

    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_FEATURE_DATA_ADDR, input_buf.dma_addr);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_FC_DATA_SIZE0, (data_in_width << 16) | m_padded);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_FC_DATA_SIZE1, align_in);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DCOMP_ADDR0, weight_buf.dma_addr);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_WEIGHT_SIZE0, wt_size_0);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_WEIGHT_SIZE1, weight_bytes_per_kernel);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_WEIGHT_SIZE2, (1 << 24) | (1 << 16) | align_out);

    emit_raw(q, RKNPU_TARGET_CORE, REG_CORE_MISC_CFG, (2 << 8) | 1);
    emit_raw(q, RKNPU_TARGET_CORE, REG_CORE_DATAOUT_SIZE_0, ((m_padded - 1) << 16) | (dataout_width - 1));
    emit_raw(q, RKNPU_TARGET_CORE, REG_CORE_DATAOUT_SIZE_1, align_out - 1);

    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_FEATURE_MODE_CFG, (15 << 5) | (2 << 1));
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DATA_FORMAT, (5 << 29) | (2 << 26) | 2);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DST_BASE_ADDR, output_buf.dma_addr);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DST_SURF_STRIDE, dst_surf_stride << 4);

    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DATA_CUBE_WIDTH, dataout_width - 1);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DATA_CUBE_HEIGHT, m_padded - 1);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DATA_CUBE_NOTCH_ADDR, (notch_val << 16) | notch_val);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DATA_CUBE_CHANNEL, ((align_out - 1) << 16) | (align_out - 1));

    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_BS_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_BS_OW_CFG, (3 << 8) | (3 << 5) | (3 << 2) | 2);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_WDMA_SIZE_0, align_out - 1);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_WDMA_SIZE_1, ((m_padded - 1) << 16) | (dataout_width - 1));
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_BN_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_EW_CFG, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_SURFACE_ADD, (dst_surf_stride * 4) << 4);

    q.push_back(0x00810000000d0008);

    std::memcpy(cmd_buf.va, q.data(), q.size() * sizeof(uint64_t));

    std::printf("Submitting matmul task to RKNPU Core 0...\n");
    if (!dev.submit(task_buf, cmd_buf, q.size(), 1)) {
        std::fprintf(stderr, "Error: NPU task submission failed.\n");
        return 1;
    }

    std::printf("Task completed. Comparing NPU raw vs CPU Reference (first 16x16):\n");
    const float* raw_ptr = static_cast<const float*>(output_buf.va);
    bool success = true;

    std::printf("  NPU output (first 256 elements raw):\n");
    for (size_t i = 0; i < 16; ++i) {
        std::printf("    [%3zu..%3zu]: ", i * 16, i * 16 + 15);
        for (size_t j = 0; j < 16; ++j) {
            std::printf("%7.2f ", raw_ptr[i * 16 + j]);
        }
        std::printf("\n");
    }

    std::printf("  CPU reference:\n");
    for (size_t row = 0; row < std::min(m, (size_t)16); ++row) {
        std::printf("    row %2zu: ", row);
        for (size_t col = 0; col < std::min(n, (size_t)16); ++col) {
            float sum = 0.0f;
            for (size_t i = 0; i < k; ++i) {
                float a_val = ((row * 17 + i * 7) % 100) * 0.01f;
                float b_val = ((i * 13 + col * 19) % 100) * 0.01f;
                sum += a_val * b_val;
            }
            std::printf("%7.2f ", sum);
        }
        std::printf("\n");
    }

    // Full comparison
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (size_t i = 0; i < k; ++i) {
                float a_val = ((row * 17 + i * 7) % 100) * 0.01f;
                float b_val = ((i * 13 + col * 19) % 100) * 0.01f;
                sum += a_val * b_val;
            }
            size_t npu_idx = (col / 4) * (m_padded * 4) + row * 4 + (col % 4);
            float npu_val = raw_ptr[npu_idx];
            if (std::abs(npu_val - sum) > 1.0f) {
                success = false;
            }
        }
    }

    if (success) {
        std::printf("\nSUCCESS: RKNPU matmul test passed successfully!\n");
    } else {
        std::fprintf(stderr, "\nFAILURE: matmul results did not match expected CPU reference!\n");
    }

    std::printf("Freeing buffers...\n");
    dev.free(task_buf);
    dev.free(cmd_buf);
    dev.free(input_buf);
    dev.free(weight_buf);
    dev.free(output_buf);

    return success ? 0 : 1;
}
