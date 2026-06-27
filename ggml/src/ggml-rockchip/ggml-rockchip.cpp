#include "ggml-rockchip.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "rockchip-drm.h"
#include "rockchip-ew.h"
#include "rockchip-matmul.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#define UNUSED(x) (void)(x)

namespace ggml_rockchip {

// Device context for RKNPU
struct ggml_backend_rockchip_context {
    std::string name;
    rk_device * dev = nullptr;
    bool owns_dev = false;
    std::mutex mutex;
};

// Buffer context
struct ggml_backend_rockchip_buffer_context {
    rk_buffer virtual_buffer;
};

//
// Backend Implementation
//

static const char * ggml_backend_rockchip_name(ggml_backend_t backend) {
    UNUSED(backend);
    return "ROCKCHIP";
}

static void ggml_backend_rockchip_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_rockchip_context *>(backend->context);
    if (ctx->owns_dev) {
        delete ctx->dev;
    }
    delete ctx;
    delete backend;
}

static uint64_t get_dma_addr(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer) return 0;
    auto * buf_ctx = static_cast<ggml_backend_rockchip_buffer_context *>(tensor->buffer->context);
    if (!buf_ctx || !buf_ctx->virtual_buffer.va) return 0;
    
    uint8_t * tensor_ptr = static_cast<uint8_t *>(tensor->data);
    uint8_t * buffer_ptr = static_cast<uint8_t *>(buf_ctx->virtual_buffer.va);
    
    return buf_ctx->virtual_buffer.dma_addr + (tensor_ptr - buffer_ptr);
}

static enum ggml_status ggml_backend_rockchip_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    auto * ctx = static_cast<ggml_backend_rockchip_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->mutex);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || 
            node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE || 
            node->op == GGML_OP_TRANSPOSE) {
            continue;
        }
        
        if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL || node->op == GGML_OP_SUB) {
            rk_ew_op ew_op;
            switch (node->op) {
                case GGML_OP_ADD: ew_op = RK_EW_ADD; break;
                case GGML_OP_MUL: ew_op = RK_EW_MUL; break;
                case GGML_OP_SUB: ew_op = RK_EW_SUB; break;
                default: return GGML_STATUS_FAILED;
            }
            
            uint64_t dma_src1 = get_dma_addr(node->src[0]);
            uint64_t dma_src2 = get_dma_addr(node->src[1]);
            uint64_t dma_dst  = get_dma_addr(node);
            size_t size = ggml_nelements(node);
            
            if (dma_src1 == 0 || dma_src2 == 0 || dma_dst == 0) {
                std::fprintf(stderr, "ggml-rockchip: error: invalid DMA address in graph_compute (op %d)\n", (int)node->op);
                return GGML_STATUS_FAILED;
            }
            
            if (!rk_run_elementwise(*ctx->dev, ew_op, dma_src1, dma_src2, dma_dst, size)) {
                return GGML_STATUS_FAILED;
            }
        } else if (node->op == GGML_OP_MUL_MAT) {
            rk_compute_matmul(*ctx->dev, node);
        } else {
            std::fprintf(stderr, "ggml-rockchip: error: unsupported op %d in graph_compute.\n", (int)node->op);
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

//
// Buffer Interface
//

static void ggml_backend_rockchip_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_rockchip_buffer_context *>(buffer->context);
    auto * buft_ctx = static_cast<ggml_backend_rockchip_context *>(buffer->buft->device->context);
    
    buft_ctx->dev->free(ctx->virtual_buffer);
    delete ctx;
}

static void * ggml_backend_rockchip_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_rockchip_buffer_context *>(buffer->context);
    return ctx->virtual_buffer.va;
}

static enum ggml_status ggml_backend_rockchip_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    UNUSED(buffer);
    UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rockchip_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = static_cast<ggml_backend_rockchip_buffer_context *>(buffer->context);
    auto * buft_ctx = static_cast<ggml_backend_rockchip_context *>(buffer->buft->device->context);
    
    std::memcpy(static_cast<uint8_t *>(tensor->data) + offset, data, size);
    
    // Sync to device memory if needed (optional optimization)
    buft_ctx->dev->sync(ctx->virtual_buffer, RKNPU_MEM_SYNC_TO_DEVICE);
}

static void ggml_backend_rockchip_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * ctx = static_cast<ggml_backend_rockchip_buffer_context *>(buffer->context);
    auto * buft_ctx = static_cast<ggml_backend_rockchip_context *>(buffer->buft->device->context);
    
    buft_ctx->dev->sync(ctx->virtual_buffer, RKNPU_MEM_SYNC_FROM_DEVICE);
    std::memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
}

static void ggml_backend_rockchip_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<ggml_backend_rockchip_buffer_context *>(buffer->context);
    if (ctx->virtual_buffer.va) {
        std::memset(ctx->virtual_buffer.va, value, ctx->virtual_buffer.size);
    }
}

//
// Buffer Type Interface
//

static const char * ggml_backend_rockchip_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return "ROCKCHIP";
}

static ggml_backend_buffer_t ggml_backend_rockchip_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * dev_ctx = static_cast<ggml_backend_rockchip_context *>(buft->device->context);
    
    auto * ctx = new ggml_backend_rockchip_buffer_context();
    ctx->virtual_buffer = dev_ctx->dev->alloc(size, 0, "rockchip_virtual_buffer");
    
    if (!ctx->virtual_buffer.va) {
        delete ctx;
        return nullptr;
    }

    static const ggml_backend_buffer_i rockchip_buffer_interface = {
        /* .free_buffer   = */ ggml_backend_rockchip_buffer_free_buffer,
        /* .get_base      = */ ggml_backend_rockchip_buffer_get_base,
        /* .init_tensor   = */ ggml_backend_rockchip_buffer_init_tensor,
        /* .memset_tensor = */ nullptr,
        /* .set_tensor    = */ ggml_backend_rockchip_buffer_set_tensor,
        /* .get_tensor    = */ ggml_backend_rockchip_buffer_get_tensor,
        /* .set_tensor_2d = */ nullptr,
        /* .get_tensor_2d = */ nullptr,
        /* .cpy_tensor    = */ nullptr,
        /* .clear         = */ ggml_backend_rockchip_buffer_clear,
        /* .reset         = */ nullptr,
    };

    return ggml_backend_buffer_init(buft, rockchip_buffer_interface, ctx, size);
}

static size_t ggml_backend_rockchip_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 64;
}

static size_t ggml_backend_rockchip_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    UNUSED(buft);
    return ggml_nbytes(tensor);
}

//
// Device Interface
//

static const char * ggml_backend_rockchip_device_get_name(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "ROCKCHIP";
}

static const char * ggml_backend_rockchip_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "Rockchip NPU (Bare-Metal)";
}

static void ggml_backend_rockchip_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    UNUSED(dev);
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_rockchip_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_rockchip_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_rockchip_device_get_name(dev);
    props->description = ggml_backend_rockchip_device_get_description(dev);
    props->type = ggml_backend_rockchip_device_get_type(dev);
    ggml_backend_rockchip_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = nullptr;

    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static bool ggml_backend_rockchip_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    UNUSED(dev);
    
    if (op->op == GGML_OP_NONE) {
        return true;
    }
    
    if (op->op == GGML_OP_ADD || op->op == GGML_OP_MUL || op->op == GGML_OP_SUB) {
        bool src0_ok = op->src[0] && op->src[0]->type == GGML_TYPE_F16 && ggml_is_contiguous(op->src[0]);
        bool src1_ok = op->src[1] && op->src[1]->type == GGML_TYPE_F16 && ggml_is_contiguous(op->src[1]);
        bool dst_ok  = op->type == GGML_TYPE_F16 && ggml_is_contiguous(op);
        bool same_shape = ggml_are_same_shape(op->src[0], op->src[1]);
        return src0_ok && src1_ok && dst_ok && same_shape;
    }
    if (op->op == GGML_OP_MUL_MAT) {
        bool src0_ok = op->src[0] && op->src[0]->type == GGML_TYPE_F16 && ggml_is_contiguous(op->src[0]);
        bool src1_ok = op->src[1] && op->src[1]->type == GGML_TYPE_F16 && ggml_is_contiguous(op->src[1]);
        bool dst_ok  = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
        bool no_batch = op->src[0]->ne[2] == 1 && op->src[0]->ne[3] == 1 &&
                        op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1;

        // Check hardware register limits for matrix dimensions
        size_t k_ggml = op->src[0]->ne[0];
        size_t m_ggml = op->src[0]->ne[1];
        size_t n_ggml = op->src[1]->ne[1];

        bool dims_ok = (m_ggml <= 16384) && (n_ggml <= 2048) && (k_ggml <= 16384);

        return src0_ok && src1_ok && dst_ok && no_batch && dims_ok;
    }
    return false;
}

static ggml_backend_t ggml_backend_rockchip_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    UNUSED(params);
    
    auto * reg_ctx = static_cast<ggml_backend_rockchip_context *>(dev->context);
    
    auto * ctx = new ggml_backend_rockchip_context();
    ctx->name = "ROCKCHIP";
    ctx->dev = reg_ctx->dev;
    ctx->owns_dev = false;

    static const struct ggml_backend_i rockchip_backend_interface = {
        /* .get_name           = */ ggml_backend_rockchip_name,
        /* .free               = */ ggml_backend_rockchip_free,
        /* .set_tensor_async   = */ nullptr,
        /* .get_tensor_async   = */ nullptr,
        /* .set_tensor_2d_async= */ nullptr,
        /* .get_tensor_2d_async= */ nullptr,
        /* .cpy_tensor_async   = */ nullptr,
        /* .synchronize        = */ nullptr,
        /* .graph_plan_create  = */ nullptr,
        /* .graph_plan_free    = */ nullptr,
        /* .graph_plan_update  = */ nullptr,
        /* .graph_plan_compute = */ nullptr,
        /* .graph_compute      = */ ggml_backend_rockchip_graph_compute,
        /* .event_record       = */ nullptr,
        /* .event_wait         = */ nullptr,
        /* .graph_optimize     = */ nullptr,
    };

    return new ggml_backend{
        /* .guid    = */ {0},
        /* .iface   = */ rockchip_backend_interface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
}

} // namespace ggml_rockchip

using namespace ggml_rockchip;

GGML_API ggml_backend_reg_t ggml_backend_rockchip_reg(void) {
    static const struct ggml_backend_reg_i rockchip_reg_interface = {
        /* .get_name         = */ [](ggml_backend_reg_t reg) { UNUSED(reg); return "ROCKCHIP"; },
        /* .get_device_count = */ [](ggml_backend_reg_t reg) { UNUSED(reg); return (size_t)1; },
        /* .get_device       = */ [](ggml_backend_reg_t reg, size_t index) -> ggml_backend_dev_t {
            if (index != 0) return nullptr;

            static const struct ggml_backend_buffer_type_i rockchip_buffer_type_interface = {
                /* .get_name       = */ ggml_backend_rockchip_buffer_type_get_name,
                /* .alloc_buffer   = */ ggml_backend_rockchip_buffer_type_alloc_buffer,
                /* .get_alignment  = */ ggml_backend_rockchip_buffer_type_get_alignment,
                /* .get_max_size   = */ nullptr,
                /* .get_alloc_size = */ ggml_backend_rockchip_buffer_type_get_alloc_size,
                /* .is_host        = */ nullptr,
            };

            static struct ggml_backend_buffer_type rockchip_buffer_type = {
                /* .iface   = */ rockchip_buffer_type_interface,
                /* .device  = */ nullptr,
                /* .context = */ nullptr,
            };

            static const struct ggml_backend_device_i rockchip_device_interface = {
                /* .get_name             = */ ggml_backend_rockchip_device_get_name,
                /* .get_description      = */ ggml_backend_rockchip_device_get_description,
                /* .get_memory           = */ ggml_backend_rockchip_device_get_memory,
                /* .get_type             = */ ggml_backend_rockchip_device_get_type,
                /* .get_props            = */ ggml_backend_rockchip_device_get_props,
                /* .init_backend         = */ ggml_backend_rockchip_device_init_backend,
                /* .get_buffer_type      = */ [](ggml_backend_dev_t dev) { UNUSED(dev); return &rockchip_buffer_type; },
                /* .get_host_buffer_type = */ nullptr,
                /* .buffer_from_host_ptr = */ nullptr,
                /* .supports_op          = */ ggml_backend_rockchip_device_supports_op,
                /* .supports_buft        = */ [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) { UNUSED(dev); return buft == &rockchip_buffer_type; },
                /* .offload_op           = */ nullptr,
                /* .event_new            = */ nullptr,
                /* .event_free           = */ nullptr,
                /* .event_synchronize    = */ nullptr,
            };

            static struct ggml_backend_device rockchip_device = {
                /* .iface   = */ rockchip_device_interface,
                /* .reg     = */ reg,
                /* .context = */ nullptr,
            };

            if (rockchip_buffer_type.device == nullptr) {
                // Link buffer type context to device context
                rockchip_device.context = reg->context;
                rockchip_buffer_type.device = &rockchip_device;
            }

            return &rockchip_device;
        },
        /* .get_proc_address = */ nullptr,
    };

    static ggml_backend_rockchip_context * reg_ctx = nullptr;
    if (!reg_ctx) {
        reg_ctx = new ggml_backend_rockchip_context();
        reg_ctx->name = "ROCKCHIP";
        reg_ctx->dev = new rk_device();
        reg_ctx->owns_dev = true;
        reg_ctx->dev->init();
    }

    static struct ggml_backend_reg rockchip_backend_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ rockchip_reg_interface,
        /* .context     = */ reg_ctx,
    };

    return &rockchip_backend_reg;
}

#ifdef GGML_BACKEND_DL
GGML_BACKEND_DL_IMPL(ggml_backend_rockchip_reg)
#endif
