#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include "rockchip-ioctl.h"

namespace ggml_rockchip {

static inline void emit_raw(std::vector<uint64_t>& q, uint32_t target, uint32_t reg, uint32_t value) {
    target = target + 0x1;
    uint64_t packed_value = ((uint64_t)(target & 0xFFFF) << 48) | ((uint64_t)(value & 0xFFFFFFFF) << 16) | (reg & 0xFFFF);
    q.push_back(packed_value);
}

struct rk_buffer {
    void*    va = nullptr;       // Virtual address (from mmap)
    uint64_t dma_addr = 0;       // DMA address for NPU register configurations
    uint64_t obj_addr = 0;       // Kernel object address of the buffer
    uint32_t handle = 0;         // DRM GEM handle
    size_t   size = 0;
    uint32_t flags = 0;         // GEM Allocation flags
    std::string name;
};

class rk_device {
public:
    int fd = -1; // File descriptor to RKNPU DRM device (/dev/dri/card1)
    rk_buffer task_buf;
    rk_buffer cmd_buf;

    rk_device();
    ~rk_device();

    rk_device(const rk_device&) = delete;
    rk_device& operator=(const rk_device&) = delete;

    bool init();
    rk_buffer alloc(size_t size, uint32_t flags = 0, const std::string& name = "");
    void free(rk_buffer& buf);
    void sync(const rk_buffer& buf, uint32_t flags);
    void reset();
    
    // Submits a command list (regcmd) using the RKNPU task descriptor format
    bool submit(const rk_buffer& task_buf, const rk_buffer& cmd_buf, 
                size_t cmd_count, uint32_t core_mask = 1);
};

} // namespace ggml_rockchip
