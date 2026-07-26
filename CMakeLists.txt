cmake_minimum_required(VERSION 3.20)

project(Qwen3Cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_program(CARGO_EXECUTABLE cargo
    HINTS "$ENV{HOME}/.cargo/bin"
    REQUIRED
)

add_subdirectory(third_party/tokenizers-cpp)

include(CTest)

find_package(OpenMP REQUIRED)

add_library(qwen3_core STATIC
    src/inference.cpp
)
target_include_directories(qwen3_core PUBLIC include)
target_link_libraries(qwen3_core PUBLIC tokenizers_cpp OpenMP::OpenMP_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(qwen3_core PRIVATE -O3 -mavx2 -mfma -march=native)
endif()


add_executable(qwen3
    src/main.cpp
)

target_link_libraries(qwen3 PRIVATE qwen3_core)

if(BUILD_TESTING)
    add_executable(test_loader
        tst/test_loader.cpp
    )
    target_link_libraries(test_loader PRIVATE qwen3_core)

    add_executable(test_block
        tst/test_block.cpp
    )
    target_link_libraries(test_block PRIVATE qwen3_core)

    add_test(
        NAME loader
        COMMAND test_loader "${PROJECT_SOURCE_DIR}/Qwen3.bin"
    )

    add_test(
        NAME block
        COMMAND test_block
    )
endif()
