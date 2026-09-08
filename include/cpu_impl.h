#ifndef CPU_IMPL_H
#define CPU_IMPL_H
#include <stdfloat>

#include "layers.h"
#include "model.h"
class CPUImpl final : public Model {
public:
    explicit CPUImpl(const std::string& path, int context_length = 512);
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

    std::vector<Block> blocks_;
    std::vector<std::bfloat16_t> embedding_;
    std::vector<std::bfloat16_t> final_norm_;
    std::vector<std::bfloat16_t> output_;
    std::vector<float> hidden_state_;
    std::vector<float> normalized_state_;
    std::vector<float> logits_;
};

#endif
