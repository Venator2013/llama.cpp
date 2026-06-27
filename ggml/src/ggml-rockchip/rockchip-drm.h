#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include "rockchip-ioctl.h"

namespace ggml_rockchip {

struct rk_buffer {
    void*    va = nullptr;       // Virtual address (from mmap)
    uint64_t dma_addr = 0;       // DMA address for NPU register configurations
    uint64_t obj_addr = 0;       // Kernel object address of the buffer
    uint32_t handle = 0;         // DRM GEM handle
    size_t   size = 0;
    std::string name;
};

class rk_device {
public:
    int fd = -1; // File descriptor to RKNPU DRM device (/dev/dri/card1)

    rk_device();
    ~rk_device();

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
