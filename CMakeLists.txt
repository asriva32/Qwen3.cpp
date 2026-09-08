cmake_minimum_required(VERSION 3.20)

project(Qwen3Cpp LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(QWEN3_NATIVE_ARCH "Optimize for the build machine's CPU" OFF)

find_package(OpenMP REQUIRED)

add_library(qwen3_core STATIC
    src/backend/layers/layers.cpp
    src/backend/model/model.cpp
    src/backend/model/cpu_impl.cpp
    src/backend/model/gpu_impl.cpp
    src/backend/sampler/sampler.cpp
    src/backend/tokenizer/tokenizer.cpp
    src/backend/utils/loader.cpp
)
target_compile_features(qwen3_core PUBLIC cxx_std_23)
target_include_directories(qwen3_core
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(qwen3_core
    PUBLIC
        OpenMP::OpenMP_CXX
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(qwen3_core PRIVATE -mavx2 -mfma)
    if(QWEN3_NATIVE_ARCH)
        target_compile_options(qwen3_core PRIVATE -march=native)
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
endif()

install(TARGETS qwen3 RUNTIME DESTINATION bin)
