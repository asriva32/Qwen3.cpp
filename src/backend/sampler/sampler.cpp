#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

std::uint64_t RandomSeed() {
    std::random_device random;
    const auto high = static_cast<std::uint64_t>(random()) << 32;
    return high ^ static_cast<std::uint64_t>(random());
}

void ValidateLogits(std::span<const float> logits) {
    if (logits.empty()) {
        throw std::invalid_argument("Cannot sample empty logits");
    }
    if (std::ranges::any_of(logits, [](float value) { return !std::isfinite(value); })) {
        throw std::invalid_argument("Cannot sample non-finite logits");
    }
}

}  // namespace

Sampler::Sampler(float temperature, std::optional<std::uint64_t> seed)
    : temperature_(temperature), generator_(seed ? *seed : RandomSeed()) {
    if (!std::isfinite(temperature_) || temperature_ < 0.0f) {
        throw std::invalid_argument("Temperature must be finite and non-negative");
    }
}

std::int32_t Sampler::Sample(std::span<const float> logits) {
    ValidateLogits(logits);
    if (temperature_ == 0.0f) {
        return Greedy(logits);
    }

    const float maximum = *std::ranges::max_element(logits);
    double total_weight = 0.0;
    for (const float logit : logits) {
        total_weight += std::exp(static_cast<double>(logit - maximum) / temperature_);
    }

    std::uniform_real_distribution<double> distribution(0.0, total_weight);
    const double sample = distribution(generator_);
    double cumulative = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        cumulative += std::exp(static_cast<double>(logits[i] - maximum) / temperature_);
        if (sample < cumulative) return static_cast<std::int32_t>(i);
    }
    return static_cast<std::int32_t>(logits.size() - 1);
}

std::int32_t Sampler::Greedy(std::span<const float> logits) {
    ValidateLogits(logits);
    return static_cast<std::int32_t>(
        std::ranges::max_element(logits) - logits.begin());
}

float Sampler::Temperature() const noexcept {
    return temperature_;
}
