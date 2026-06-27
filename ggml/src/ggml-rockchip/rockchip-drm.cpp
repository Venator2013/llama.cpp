#include "rockchip-drm.h"
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>

namespace ggml_rockchip {

rk_device::rk_device() : fd(-1) {}

rk_device::~rk_device() {
    if (fd >= 0) {
        close(fd);
    }
}

bool rk_device::init() {
    const char* paths[] = {
        "/dev/dri/card1",
        "/dev/dri/card0",
        "/dev/dri/renderD129",
        "/dev/dri/renderD128"
    };
    for (const char* path : paths) {
        fd = open(path, O_RDWR | O_CLOEXEC);
        if (fd >= 0) {
            // Verify device by querying RKNPU driver version
            struct rknpu_action action;
            std::memset(&action, 0, sizeof(action));
            action.flags = RKNPU_GET_DRV_VERSION;
            action.value = 0;
            if (ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &action) >= 0) {
                std::printf("ggml-rockchip: successfully opened RKNPU device at %s (Driver version: 0x%08x)\n", path, action.value);
                return true;
            }
            close(fd);
            fd = -1;
        }
    }
    std::fprintf(stderr, "ggml-rockchip: error: failed to find and open RKNPU DRM device.\n");
    return false;
}

rk_buffer rk_device::alloc(size_t size, uint32_t flags, const std::string& name) {
    rk_buffer buf;
    if (fd < 0) {
        std::fprintf(stderr, "ggml-rockchip: device not initialized during alloc.\n");
        return buf;
    }

    buf.size = size;
    buf.name = name;

    struct rknpu_mem_create mem_create;
    std::memset(&mem_create, 0, sizeof(mem_create));
    mem_create.flags = flags | RKNPU_MEM_NON_CACHEABLE;
    mem_create.size = size;

    if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create) < 0) {
        std::fprintf(stderr, "ggml-rockchip: MEM_CREATE ioctl failed for size %zu (%s)\n", size, name.c_str());
        return buf;
    }

    buf.handle = mem_create.handle;
    buf.obj_addr = mem_create.obj_addr;
    buf.dma_addr = mem_create.dma_addr;

    struct rknpu_mem_map mem_map;
    std::memset(&mem_map, 0, sizeof(mem_map));
    mem_map.handle = mem_create.handle;
    mem_map.offset = 0;

    if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mem_map) < 0) {
        std::fprintf(stderr, "ggml-rockchip: MEM_MAP ioctl failed for handle %u (%s)\n", mem_create.handle, name.c_str());
        free(buf);
        return buf;
    }

    buf.va = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mem_map.offset);
    if (buf.va == MAP_FAILED) {
        std::fprintf(stderr, "ggml-rockchip: mmap failed for size %zu (%s)\n", size, name.c_str());
        buf.va = nullptr;
        free(buf);
        return buf;
    }

    return buf;
}

void rk_device::free(rk_buffer& buf) {
    if (fd < 0) return;

    if (buf.va && buf.size) {
        munmap(buf.va, buf.size);
    }
    if (buf.handle) {
        struct rknpu_mem_destroy mem_destroy;
        std::memset(&mem_destroy, 0, sizeof(mem_destroy));
        mem_destroy.handle = buf.handle;
        mem_destroy.obj_addr = buf.obj_addr;
        if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &mem_destroy) < 0) {
            std::fprintf(stderr, "ggml-rockchip: MEM_DESTROY ioctl failed for handle %u\n", buf.handle);
        }
    }
    buf.va = nullptr;
    buf.dma_addr = 0;
    buf.obj_addr = 0;
    buf.handle = 0;
    buf.size = 0;
}

void rk_device::sync(const rk_buffer& buf, uint32_t flags) {
    if (fd < 0 || buf.handle == 0) return;

    struct rknpu_mem_sync mem_sync;
    std::memset(&mem_sync, 0, sizeof(mem_sync));
    mem_sync.flags = flags;
    mem_sync.obj_addr = buf.obj_addr;
    mem_sync.offset = 0;
    mem_sync.size = buf.size;
    if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &mem_sync) < 0) {
        std::fprintf(stderr, "ggml-rockchip: MEM_SYNC ioctl failed for GEM handle %u\n", buf.handle);
    }
}

void rk_device::reset() {
    if (fd < 0) return;

    struct rknpu_action action;
    std::memset(&action, 0, sizeof(action));
    action.flags = RKNPU_ACT_RESET;
    action.value = 0;
    if (ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &action) < 0) {
        std::fprintf(stderr, "ggml-rockchip: NPU reset failed\n");
    }
}

bool rk_device::submit(const rk_buffer& task_buf, const rk_buffer& cmd_buf, 
                       size_t cmd_count, uint32_t core_mask) {
    if (fd < 0) return false;

    auto* tasks = static_cast<struct rknpu_task*>(task_buf.va);
    if (!tasks) return false;

    std::memset(tasks, 0, sizeof(struct rknpu_task));
    tasks[0].flags = 0;
    tasks[0].op_idx = 4;
    tasks[0].enable_mask = 0x18;
    tasks[0].int_mask = 0x300;
    tasks[0].int_clear = 0x1ffff;
    tasks[0].int_status = 0;
    tasks[0].regcfg_amount = cmd_count;
    tasks[0].regcfg_offset = 0;
    tasks[0].regcmd_addr = cmd_buf.dma_addr;

    struct rknpu_submit submit_req;
    std::memset(&submit_req, 0, sizeof(submit_req));
    submit_req.flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG;
    submit_req.timeout = 6000;
    submit_req.task_start = 0;
    submit_req.task_number = 1;
    submit_req.task_counter = 0;
    submit_req.priority = 0;
    submit_req.task_obj_addr = task_buf.obj_addr;
    submit_req.regcfg_obj_addr = 0;
    submit_req.task_base_addr = 0;
    submit_req.user_data = 0;
    submit_req.core_mask = core_mask;
    submit_req.fence_fd = -1;
    
    // Core distribution for subcore_task
    submit_req.subcore_task[0].task_start = 0;
    submit_req.subcore_task[0].task_number = 1;
    submit_req.subcore_task[1].task_start = 1;
    submit_req.subcore_task[1].task_number = 0;
    submit_req.subcore_task[2].task_start = 2;
    submit_req.subcore_task[2].task_number = 0;

    int ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit_req);
    if (ret < 0) {
        std::fprintf(stderr, "ggml-rockchip: RKNPU_SUBMIT ioctl failed with code %d\n", ret);
        return false;
    }
    return true;
}

} // namespace ggml_rockchip
