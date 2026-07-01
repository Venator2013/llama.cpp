#include "rockchip-matmul.h"
#include "rockchip-regs.h"
#include <vector>
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace ggml_rockchip {

const uint32_t RKNPU_TARGET_CNA = 512;
const uint32_t RKNPU_TARGET_CORE = 2048;
const uint32_t RKNPU_TARGET_DPU = 4096;
const uint32_t RKNPU_TARGET_DPU_RDMA = 8192;

static inline size_t align_up(size_t n, size_t alignment) {
    return (n + alignment - 1) & ~(alignment - 1);
}

static wmma_params get_wmma_params(size_t m, size_t n, size_t k) {
    m = std::max((size_t)1, m);
    n = std::max((size_t)1, n);
    k = std::max((size_t)1, k);

    size_t m_padded = align_up(m, 64);

    size_t align_in = std::max((size_t)32, align_up(k, 32));
    size_t align_out = std::max((size_t)32, align_up(n, 32));

    size_t data_in_width = 1;
    size_t data_in_height = m_padded;
    size_t dataout_width = 1;
    size_t dataout_height = m_padded;
    size_t out_width_stride = 1;

    bool is_kn_64 = (k == 64 && n == 64);
    bool is_kn_256 = (k == 256 && n == 256);
    bool is_kn_512 = (k == 512 && n == 512);
    bool is_kn_lg_512 = (k > 512 && n > 512);
    bool is_matmul_64 = (m == 64 && k == 64 && n == 64);
    bool is_matmul_256 = (m == 256 && k == 256 && n == 256);

    size_t feature_grains = m_padded + 1;

    size_t weight_bytes_per_kernel = align_in * sizeof(uint16_t);
    size_t fd_bytes = data_in_width * data_in_height * align_in * sizeof(uint16_t);
    size_t data_bank = std::max((size_t)1, std::min((size_t)11, (fd_bytes + 1024 + 32768 - 1) / 32768));

    size_t line_stride = data_in_width * 4;
    if (k > 32 && k < 512 && k != 64 && k != 256) {
        line_stride = std::min((size_t)13, (k + 31) / 32) * 4;
    }

    size_t surf_groups = data_in_height / 4;
    size_t surf_stride = 0;
    if (surf_groups > 0) {
        surf_stride = (line_stride * (surf_groups - 1)) * (align_in >= 64 ? 1 : 0);
    }
    if ((k > 32 && k < 64) || (k > 64 && k <= 128) || (k > 128 && k < 256) || (k > 256 && k < 512)) {
        surf_stride = 0;
    }

    size_t dst_surf_stride = m_padded;

    size_t notch_val = 0;

    return {
        m, n, k, align_in, align_out,
        data_in_width, data_in_height, dataout_width, dataout_height,
        feature_grains, weight_bytes_per_kernel, data_bank, line_stride,
        surf_stride, dst_surf_stride, notch_val
    };
}


void rk_compute_matmul(rk_device& dev, struct ggml_tensor* op) {
    struct ggml_tensor* weights_tensor = op->src[0];
    struct ggml_tensor* features_tensor = op->src[1];
    struct ggml_tensor* dst_tensor = op;

    GGML_ASSERT(weights_tensor->type == GGML_TYPE_F16);
    GGML_ASSERT(features_tensor->type == GGML_TYPE_F16);
    GGML_ASSERT(dst_tensor->type == GGML_TYPE_F32);

    size_t k_ggml = weights_tensor->ne[0];
    size_t m_ggml = weights_tensor->ne[1];
    size_t n_ggml = features_tensor->ne[1];

    size_t m = n_ggml;
    size_t n = m_ggml;
    size_t k = k_ggml;

    wmma_params p = get_wmma_params(m, n, k);

    std::vector<uint16_t> in_pack(p.data_in_height * p.align_in, 0);
    std::vector<uint16_t> wt_pack(p.align_out * p.align_in, 0);

    const uint16_t* a_matrix = static_cast<const uint16_t*>(features_tensor->data);
    const uint16_t* b_matrix = static_cast<const uint16_t*>(weights_tensor->data);

    if (m == 64 && n == 64 && k == 64) {
        for (size_t mm = 1; mm <= 64; ++mm) {
            for (size_t kk = 1; kk <= 64; ++kk) {
                size_t plane = (kk - 1) / 8;
                size_t offset = (kk - 1) % 8;
                in_pack[plane * 64 * 8 + (mm - 1) * 8 + offset] = a_matrix[(mm - 1) * k + (kk - 1)];
            }
        }
        for (size_t nn = 1; nn <= 64; ++nn) {
            for (size_t kk = 1; kk <= 64; ++kk) {
                size_t kpg = (nn - 1) / 16;
                size_t cpg = (kk - 1) / 32;
                size_t wt_idx = ((cpg * 32) * 16) + (kpg * 16 * p.align_in) + ((kk - 1) % 32) + (((nn - 1) % 16) * 32);
                wt_pack[wt_idx] = b_matrix[(nn - 1) * k + (kk - 1)];
            }
        }
    } else {
        std::fill(in_pack.begin(), in_pack.end(), 0);
        for (size_t r = 0; r < m; ++r) {
            for (size_t col = 0; col < k; ++col) {
                size_t in_idx = (col / 8) * (p.data_in_height * 8) + r * 8 + (col % 8);
                in_pack[in_idx] = a_matrix[r * k + col];
            }
        }
        for (size_t r = 0; r < n; ++r) {
            for (size_t c = 0; c < k; ++c) {
                size_t wt_idx = (r / 16) * (p.align_in / 32) * 512 + (c / 32) * 512 + (r % 16) * 32 + (c % 32);
                wt_pack[wt_idx] = b_matrix[r * k + c];
            }
        }
    }

    size_t in_bytes = in_pack.size() * sizeof(uint16_t);
    size_t wt_bytes = wt_pack.size() * sizeof(uint16_t);

    size_t out_bytes = std::max((size_t)0x100, p.align_out * p.dataout_height * sizeof(float));

    rk_buffer input_buf = dev.alloc(in_bytes, 0, "matmul_input");
    rk_buffer weight_buf = dev.alloc(wt_bytes, 0, "matmul_weight");
    rk_buffer output_buf = dev.alloc(out_bytes, 0, "matmul_output");

    if (!input_buf.va || !weight_buf.va || !output_buf.va) {
        std::fprintf(stderr, "ggml-rockchip: error: failed to allocate matmul buffers.\n");
        if (input_buf.va) dev.free(input_buf);
        if (weight_buf.va) dev.free(weight_buf);
        if (output_buf.va) dev.free(output_buf);
        return;
    }

    std::memcpy(input_buf.va, in_pack.data(), in_bytes);
    std::memcpy(weight_buf.va, wt_pack.data(), wt_bytes);

    std::vector<uint64_t> q;
    q.reserve(64);

    uint32_t s_pointer = (1 << 3) | (1 << 2) | (1 << 1);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_S_POINTER, s_pointer);

    uint32_t conv_con1 = (2 << 7) | (2 << 4);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CONV_CON1, conv_con1);

    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CONV_CON2, p.feature_grains << 4);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CONV_CON3, (1 << 3) | 1);

    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DATA_SIZE0, (p.data_in_width << 16) | p.data_in_height);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DATA_SIZE1, ((p.align_in - 1) << 16) | p.align_in);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DATA_SIZE2, p.dataout_width);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DATA_SIZE3, p.dataout_width * p.dataout_height);

    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DMA_CON0, (15 << 16) | 15);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DMA_CON1, p.line_stride);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DMA_CON2, p.surf_stride);

    uint32_t cbuf_con0 = ((12 - p.data_bank) << 4) | p.data_bank;
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CBUF_CON0, cbuf_con0);

    uint32_t cbuf_entries = ((p.data_in_width * p.align_in + 31) / 32);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CBUF_CON1, cbuf_entries);

    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CVT_CON0, 0xB);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CVT_CON1, 1 << 16);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CVT_CON2, 1 << 16);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CVT_CON3, 1 << 16);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_CVT_CON4, 1 << 16);

    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_FEATURE_DATA_ADDR, input_buf.dma_addr);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_FC_DATA_SIZE0, (p.data_in_width << 16) | p.data_in_height);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_FC_DATA_SIZE1, p.align_in);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_DCOMP_ADDR0, weight_buf.dma_addr);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_WEIGHT_SIZE0, p.weight_bytes_per_kernel * p.align_out);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_WEIGHT_SIZE1, p.weight_bytes_per_kernel);
    emit_raw(q, RKNPU_TARGET_CNA, REG_CNA_WEIGHT_SIZE2, (1 << 24) | (1 << 16) | p.align_out);

    emit_raw(q, RKNPU_TARGET_CORE, REG_CORE_MISC_CFG, (2 << 8) | 1);
    emit_raw(q, RKNPU_TARGET_CORE, REG_CORE_DATAOUT_SIZE_0, ((p.dataout_height - 1) << 16) | (p.dataout_width - 1));
    emit_raw(q, RKNPU_TARGET_CORE, REG_CORE_DATAOUT_SIZE_1, p.align_out - 1);

    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_FEATURE_MODE_CFG, (15 << 5) | (2 << 1));
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DATA_FORMAT, (5 << 29) | (2 << 26) | 2);

    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DST_BASE_ADDR, output_buf.dma_addr);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DST_SURF_STRIDE, p.dst_surf_stride << 4);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DATA_CUBE_WIDTH, p.dataout_width - 1);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DATA_CUBE_HEIGHT, p.dataout_height - 1);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DATA_CUBE_NOTCH_ADDR, (p.notch_val << 16) | p.notch_val);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_DATA_CUBE_CHANNEL, ((p.align_out - 1) << 16) | (p.align_out - 1));

    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_BS_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_BS_OW_CFG, (3 << 8) | (3 << 5) | (3 << 2) | 2);

    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_WDMA_SIZE_0, p.align_out - 1);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_WDMA_SIZE_1, ((p.dataout_height - 1) << 16) | (p.dataout_width - 1));

    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_BN_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1);
    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_EW_CFG, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1);

    emit_raw(q, RKNPU_TARGET_DPU, REG_DPU_SURFACE_ADD, (p.dst_surf_stride * 4) << 4);

    q.push_back(0x00810000000d0008);

    std::memcpy(dev.cmd_buf.va, q.data(), q.size() * sizeof(uint64_t));

    if (!dev.submit(dev.task_buf, dev.cmd_buf, q.size(), 1)) {
        std::fprintf(stderr, "ggml-rockchip: error: NPU matmul submission failed.\n");
        dev.free(input_buf);
        dev.free(weight_buf);
        dev.free(output_buf);
        return;
    }

    float* dst_ptr = static_cast<float*>(dst_tensor->data);
    const float* raw_ptr = static_cast<const float*>(output_buf.va);

    const size_t c2 = 4;
    for (size_t col = 0; col < n; ++col) {
        size_t plane = col / c2;
        size_t offset = col % c2;
        size_t plane_base = plane * p.dataout_height * c2;
        for (size_t row = 0; row < m; ++row) {
            dst_ptr[row * n + col] = raw_ptr[plane_base + row * c2 + offset];
        }
    }



    dev.free(input_buf);
    dev.free(weight_buf);
    dev.free(output_buf);
}

} // namespace ggml_rockchip
