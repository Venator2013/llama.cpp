#include "rockchip-drm.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

// Hilfsfunktion zur Konvertierung Float -> FP16 (Bits)
static uint16_t float_to_fp16(float val) {
    union { float f; uint32_t u; } u = { val };
    uint32_t sign = (u.u >> 16) & 0x8000;
    int32_t exponent = ((u.u >> 23) & 0xff) - 127;
    uint32_t mantissa = u.u & 0x007fffff;

    if (exponent <= -15) {
        return sign; // Underflow zu 0
    } else if (exponent >= 16) {
        return sign | 0x7c00; // Overflow zu Inf
    }
    exponent += 15;
    return sign | (exponent << 10) | (mantissa >> 13);
}

// Hilfsfunktion zur Konvertierung FP16 -> Float
static float fp16_to_float(uint16_t val) {
    uint32_t sign = (val & 0x8000) << 16;
    int32_t exponent = (val & 0x7c00) >> 10;
    uint32_t mantissa = (val & 0x03ff) << 13;

    if (exponent == 0) {
        return 0.0f;
    } else if (exponent == 31) {
        exponent = 255;
    } else {
        exponent += (127 - 15);
    }
    union { uint32_t u; float f; } u;
    u.u = sign | (exponent << 23) | mantissa;
    return u.f;
}

// Hilfsfunktion zum Packen eines 64-Bit Register-Kommandos
static void emit_raw(std::vector<uint64_t>& q, uint32_t target, uint32_t reg, uint32_t value) {
    target = target + 0x1;
    uint64_t packed_value = ((uint64_t)(target & 0xFFFF) << 48) | ((uint64_t)(value & 0xFFFFFFFF) << 16) | (reg & 0xFFFF);
    q.push_back(packed_value);
}

int main() {
    std::printf("--- Starting RKNPU DRM/ioctl ADD operation test ---\n");

    ggml_rockchip::rk_device dev;
    if (!dev.init()) {
        std::fprintf(stderr, "Error: failed to initialize NPU device.\n");
        return 1;
    }

    size_t num_elements = 16;
    size_t data_size = num_elements * sizeof(uint16_t);

    std::printf("Allocating GEM buffers...\n");
    ggml_rockchip::rk_buffer task_buf = dev.alloc(1024, RKNPU_MEM_KERNEL_MAPPING, "task_buf");
    ggml_rockchip::rk_buffer cmd_buf = dev.alloc(1024, 0, "cmd_buf");
    ggml_rockchip::rk_buffer input_buf = dev.alloc(data_size, 0, "input_A");
    ggml_rockchip::rk_buffer weight_buf = dev.alloc(data_size, 0, "input_B");
    ggml_rockchip::rk_buffer output_buf = dev.alloc(data_size, 0, "output_C");

    if (!task_buf.va || !cmd_buf.va || !input_buf.va || !weight_buf.va || !output_buf.va) {
        std::fprintf(stderr, "Error: buffer allocation failed.\n");
        return 1;
    }

    // Vektor-Werte initialisieren: A = 2.0f, B = 3.0f
    uint16_t* va_A = static_cast<uint16_t*>(input_buf.va);
    uint16_t* va_B = static_cast<uint16_t*>(weight_buf.va);
    uint16_t* va_C = static_cast<uint16_t*>(output_buf.va);

    for (size_t i = 0; i < num_elements; ++i) {
        va_A[i] = float_to_fp16(2.0f);
        va_B[i] = float_to_fp16(3.0f);
        va_C[i] = 0; // Output leeren
    }

    // Register-Befehlsliste bauen
    std::vector<uint64_t> q;

    // 1. DPU Block Setup (Target: 4096 = DPU)
    emit_raw(q, 4096, 0x400c, 0x1e5);          // REG_DPU_FEATURE_MODE_CFG
    emit_raw(q, 4096, 0x4010, 0x48000002);     // REG_DPU_DATA_FORMAT (FP16 out/in/proc)
    emit_raw(q, 4096, 0x403c, 0x00070007);     // REG_DPU_DATA_CUBE_CHANNEL (8 channels)
    emit_raw(q, 4096, 0x4030, 1);              // REG_DPU_DATA_CUBE_WIDTH (2 columns - 1)
    emit_raw(q, 4096, 0x4070, 0x108202c0);     // REG_DPU_EW_CFG (ADD operation alu_algo=2)
    emit_raw(q, 4096, 0x4020, output_buf.dma_addr); // REG_DPU_DST_BASE_ADDR

    // 2. DPU RDMA Block Setup (Target: 8192 = DPU_RDMA)
    emit_raw(q, 8192, 0x500c, 1);              // REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH
    emit_raw(q, 8192, 0x5010, 0);              // REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT
    emit_raw(q, 8192, 0x5014, 7);              // REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL
    emit_raw(q, 8192, 0x5034, 0x40000008);     // REG_DPU_RDMA_RDMA_ERDMA_CFG
    emit_raw(q, 8192, 0x5018, input_buf.dma_addr);  // REG_DPU_RDMA_RDMA_SRC_BASE_ADDR
    emit_raw(q, 8192, 0x5038, weight_buf.dma_addr); // REG_DPU_RDMA_RDMA_EW_BASE_ADDR

    // 3. Hardware-Submit Befehle
    q.push_back(0x2001000178495044);           // REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG (0x5044) = 0x00017849
    q.push_back(0x0081000000180008);           // REG_PC_OPERATION_ENABLE (0x0008) = 0x0018

    // Befehlsliste in den Command-Buffer kopieren
    std::memcpy(cmd_buf.va, q.data(), q.size() * sizeof(uint64_t));

    std::printf("Submitting ADD task to RKNPU Core 0...\n");
    if (!dev.submit(task_buf, cmd_buf, q.size(), 1)) {
        std::fprintf(stderr, "Error: NPU task submission failed.\n");
        return 1;
    }

    std::printf("Task completed. Verifying results...\n");
    bool success = true;
    for (size_t i = 0; i < num_elements; ++i) {
        float val = fp16_to_float(va_C[i]);
        std::printf("  Index %zu: 2.0 + 3.0 = %f\n", i, val);
        if (std::abs(val - 5.0f) > 0.01f) {
            success = false;
        }
    }

    if (success) {
        std::printf("\nSUCCESS: RKNPU addition test passed successfully!\n");
    } else {
        std::fprintf(stderr, "\nFAILURE: addition results did not match expected 5.0!\n");
    }

    std::printf("Freeing buffers...\n");
    dev.free(task_buf);
    dev.free(cmd_buf);
    dev.free(input_buf);
    dev.free(weight_buf);
    dev.free(output_buf);

    return success ? 0 : 1;
}
