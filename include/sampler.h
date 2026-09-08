#ifndef SAMPLER_H
#define SAMPLER_H

#include <cstdint>
#include <optional>
#include <random>
#include <span>

class Sampler {
public:
    static constexpr float kDefaultTemperature = 0.6f;

    explicit Sampler(
        float temperature = kDefaultTemperature,
        std::optional<std::uint64_t> seed = std::nullopt
    );

    [[nodiscard]] std::int32_t Sample(std::span<const float> logits);
    [[nodiscard]] static std::int32_t Greedy(std::span<const float> logits);
    [[nodiscard]] float Temperature() const noexcept;

private:
    float temperature_;
    std::mt19937_64 generator_;
};

#endif
