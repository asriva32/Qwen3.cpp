#ifndef GPU_IMPL_H
#define GPU_IMPL_H

#include "model.h"

class GPUImpl final : public Model {
public:
    explicit GPUImpl(const std::string& path, int context_length = 512);

    void Prefill(std::span<const std::int32_t> tokens, int pos, State& state) override;
    void ForwardToken(std::int32_t token, int pos, State& state) override;
    GenerationResult Generate(
        const std::vector<std::int32_t>& prompt_tokens,
        size_t max_generated_tokens = 512,
        bool stop_on_eos = true,
        float temperature = 0.6f,
        std::optional<std::uint64_t> seed = std::nullopt
    ) override;

private:
    void InitializeBackend() override;
    void ResetBackend() override;
};

#endif
