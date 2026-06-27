#include "rockchip-drm.h"
#include <cstdio>
#include <cstring>

int main() {
    std::printf("--- Starting RKNPU DRM/ioctl verification test ---\n");

    ggml_rockchip::rk_device dev;
    if (!dev.init()) {
        std::fprintf(stderr, "error: failed to initialize NPU device.\n");
        return 1;
    }

    std::printf("Device initialized successfully. Attempting to allocate buffer...\n");
    size_t test_size = 1024;
    ggml_rockchip::rk_buffer buf = dev.alloc(test_size, 0, "test_buffer");
    if (!buf.va) {
        std::fprintf(stderr, "error: buffer allocation failed.\n");
        return 1;
    }

    std::printf("Buffer allocated successfully.\n");
    std::printf("  Handle: %u\n", buf.handle);
    std::printf("  DMA Address: 0x%016llx\n", (unsigned long long)buf.dma_addr);
    std::printf("  Object Address: 0x%016llx\n", (unsigned long long)buf.obj_addr);
    std::printf("  Virtual Address: %p\n", buf.va);

    // Test writing and reading memory
    const char* test_str = "RKNPU Bare-Metal DRM Verification Success!";
    std::printf("Writing test pattern: '%s'\n", test_str);
    std::memcpy(buf.va, test_str, std::strlen(test_str) + 1);

    // Synchronize to device memory
    std::printf("Syncing to device memory...\n");
    dev.sync(buf, RKNPU_MEM_SYNC_TO_DEVICE);

    // Synchronize back from device
    std::printf("Syncing back from device memory...\n");
    dev.sync(buf, RKNPU_MEM_SYNC_FROM_DEVICE);

    char read_back[128];
    std::memcpy(read_back, buf.va, std::strlen(test_str) + 1);
    std::printf("Read back pattern:    '%s'\n", read_back);

    if (std::strcmp(test_str, read_back) == 0) {
        std::printf("Pattern matched successfully!\n");
    } else {
        std::fprintf(stderr, "error: pattern mismatch!\n");
    }

    std::printf("Freeing buffer...\n");
    dev.free(buf);
    std::printf("Buffer freed. Test completed successfully.\n");

    return 0;
}
