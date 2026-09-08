#include "gpu_impl.h"

#include <stdexcept>

namespace {

[[noreturn]] void ThrowNotImplemented() {
    throw std::runtime_error("GPU backend is not implemented yet");
}

}  // namespace

GPUImpl::GPUImpl(const std::string& path, int context_length)
    : Model(path) {
    InitializeInference(context_length);
}

auto GPUImpl::InitializeBackend() -> void {
    ThrowNotImplemented();
}

auto GPUImpl::ResetBackend() -> void {
    ThrowNotImplemented();
}

auto GPUImpl::Prefill(std::span<const std::int32_t>, int, State&) -> void {
    ThrowNotImplemented();
}

auto GPUImpl::ForwardToken(std::int32_t, int, State&) -> void {
    ThrowNotImplemented();
}

auto GPUImpl::Generate(
    const std::vector<std::int32_t>&,
    size_t,
    bool,
    float,
    std::optional<std::uint64_t>
) -> GenerationResult {
    ThrowNotImplemented();
}
