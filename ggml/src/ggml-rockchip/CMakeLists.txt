# Finding OpenMP
find_package(OpenMP REQUIRED)

# Adding backend files
ggml_add_backend_library(ggml-rockchip
    ggml-rockchip.cpp
    rockchip-drm.cpp
    rockchip-ew.cpp
    rockchip-matmul.cpp
    rockchip-lut.cpp
)

# Linking headers
target_include_directories(ggml-rockchip
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Adding OpenMP in build
if(OpenMP_FOUND)
    add_definitions(-DHAVE_OPENMP=1)
    target_compile_options(ggml-rockchip PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(ggml-rockchip PRIVATE ${OpenMP_CXX_FLAGS})
endif()

if (GGML_BUILD_TESTS OR GGML_ROCKCHIP)
    add_executable(test-rockchip-drm test-rockchip-drm.cpp)
    target_link_libraries(test-rockchip-drm PRIVATE ggml-rockchip ggml-base)
endif()
