cmake_minimum_required(VERSION 3.20)

project(Qwen3Cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(QWEN3_NATIVE_ARCH "Optimize for the build machine's CPU" OFF)

# CUDA 13.3 supports C++23, but CMake 4.2's NVIDIA compiler table only
# advertises standards through C++20.
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.3)
    set(CMAKE_CUDA23_STANDARD_COMPILE_OPTION "-std=c++23")
    set(CMAKE_CUDA23_EXTENSION_COMPILE_OPTION "-std=c++23")
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(qwen3_core STATIC
    src/backend/inference/infer.cu
    src/backend/layers/layers.cpp
    src/backend/model/model.cpp
    src/backend/sampler/sampler.cpp
    src/backend/tokenizer/tokenizer.cpp
    src/backend/utils/loader.cpp
)
set_target_properties(qwen3_core PROPERTIES
    CUDA_STANDARD 23
    CUDA_STANDARD_REQUIRED ON
)
target_include_directories(qwen3_core
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(qwen3_core
    PUBLIC
        OpenMP::OpenMP_CXX
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(qwen3_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-mavx2;-mfma>
    )
    if(QWEN3_NATIVE_ARCH)
        target_compile_options(qwen3_core PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-march=native>
        )
    endif()
else()
    message(FATAL_ERROR "Qwen3.cpp currently supports GCC and Clang only")
endif()

add_executable(qwen3 src/main.cpp)
target_link_libraries(qwen3 PRIVATE qwen3_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(qwen3_tests tests/test_model.cpp)
    target_link_libraries(qwen3_tests PRIVATE qwen3_core)
    add_test(NAME qwen3_tests COMMAND qwen3_tests)

    add_executable(qwen3_kernel_tests tests/test_kernels.cu)
    set_target_properties(qwen3_kernel_tests PROPERTIES
        CUDA_STANDARD 23
        CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(qwen3_kernel_tests PRIVATE qwen3_core)
    add_test(NAME qwen3_kernel_tests COMMAND qwen3_kernel_tests)
    set_tests_properties(qwen3_kernel_tests PROPERTIES SKIP_RETURN_CODE 77)
endif()

install(TARGETS qwen3 RUNTIME DESTINATION bin)
