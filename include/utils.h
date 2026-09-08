#ifndef UTILS_H
#define UTILS_H
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdfloat>
#include <string>
#include <vector>


struct GenerationStats {
    size_t prompt_tokens = 0;
    size_t generated_tokens = 0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    bool stopped_on_eos = false;
    bool reached_token_limit = false;

    double PrefillTokensPerSecond() const {
        return prefill_seconds > 0.0 ? prompt_tokens / prefill_seconds : 0.0;
    }

    double DecodeTokensPerSecond() const {
        return decode_seconds > 0.0 ? generated_tokens / decode_seconds : 0.0;
    }
};

struct GenerationResult {
    std::vector<std::int32_t> tokens;
    GenerationStats stats;
};

enum class TensorDType : std::uint32_t {
    Float32 = 1,
    BFloat16 = 3,
    UInt8 = 4,
    Int32 = 5,
};

struct TensorInfo {
    std::string name;
    TensorDType dtype;
    std::vector<size_t> shape;
    size_t data_offset = 0;
    size_t byte_size = 0;
};

template <typename T>
concept SupportedJsonValue =
    std::same_as<T, float>        ||
    std::same_as<T, std::int32_t> ||
    std::same_as<T, bool>         ||
    std::same_as<T, std::string>;

template <typename T>
concept SupportedTensorElement =
    std::same_as<T, float> ||
    std::same_as<T, std::int32_t> ||
    std::same_as<T, std::uint8_t> ||
    std::same_as<T, std::bfloat16_t>;

template <SupportedTensorElement T>
struct Tensor {
    TensorInfo info;
    std::vector<T> data;

    const T* ptr() const {
        return data.data();
    }

    T* ptr() {
        return data.data();
    }
};

template <SupportedTensorElement T>
constexpr TensorDType ExpectedDType() {
    if constexpr (std::same_as<T, std::bfloat16_t>) {
        return TensorDType::BFloat16;
    } else if constexpr (std::same_as<T, float>) {
        return TensorDType::Float32;
    } else if constexpr (std::same_as<T, std::int32_t>) {
        return TensorDType::Int32;
    } else {
        return TensorDType::UInt8;
    }
}

std::vector<std::uint8_t> LoadTensorBytes(
    const std::string& path,
    const TensorInfo& info,
    TensorDType expected_dtype,
    size_t element_size
);

#endif
