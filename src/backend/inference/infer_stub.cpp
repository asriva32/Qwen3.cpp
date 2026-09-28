#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "layers.h"

namespace {

[[noreturn]] void ThrowCudaUnavailable() {
    throw std::runtime_error(
        "GPU inference is unavailable in this CPU-only build"
    );
}

}  // namespace

bool cuda_backend_available() noexcept {
    return false;
}

extern "C" void* upload_cuda(void*, size_t) {
    ThrowCudaUnavailable();
}

extern "C" void* allocate_cuda_zeroed(size_t) {
    ThrowCudaUnavailable();
}

extern "C" void zero_cuda(void*, size_t) {
    ThrowCudaUnavailable();
}

extern "C" void free_cuda(void*) {}

extern "C" void set_cuda_device(int) {
    ThrowCudaUnavailable();
}

extern "C" void download_cuda(void*, const void*, size_t) {
    ThrowCudaUnavailable();
}

extern "C" void synchronize_cuda() {
    ThrowCudaUnavailable();
}

void rmsnorm_gpu(float*, const float*, const std::bfloat16_t*, float, int, int) {
    ThrowCudaUnavailable();
}

void matmul_gpu(
    float*, const float*, const std::bfloat16_t*, int, int, int
) {
    ThrowCudaUnavailable();
}

void qk_norm_rope_and_update_cache(
    float*, float*, const float*, std::bfloat16_t*, std::bfloat16_t*,
    const std::bfloat16_t*, const std::bfloat16_t*, int, int, int, int, int,
    float, float, int, int
) {
    ThrowCudaUnavailable();
}

void rotate_sink_tokens(
    std::bfloat16_t*, size_t, size_t, int, float, int
) {
    ThrowCudaUnavailable();
}

void ffn_gpu(
    float*, float*, float*, const float*, const std::bfloat16_t*,
    const std::bfloat16_t*, const std::bfloat16_t*, int, int, int
) {
    ThrowCudaUnavailable();
}

void attn_gpu(
    float*, float*, const float*, const std::bfloat16_t*,
    const std::bfloat16_t*, int, int, int, int, int, int
) {
    ThrowCudaUnavailable();
}

void add_gpu(float*, const float*, size_t) {
    ThrowCudaUnavailable();
}

void embedding_gpu(float*, const std::bfloat16_t*, int) {
    ThrowCudaUnavailable();
}

void argmax_gpu(std::int32_t*, const float*, int) {
    ThrowCudaUnavailable();
}
