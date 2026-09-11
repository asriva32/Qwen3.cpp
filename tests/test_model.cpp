#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "model.h"
#include "layers.h"
#include "sampler.h"
#include "tokenizer.h"

namespace {

std::vector<std::bfloat16_t> MakeValues(size_t count, float scale) {
    std::vector<std::bfloat16_t> values(count);
    for (size_t i = 0; i < count; ++i) {
        const auto centered = static_cast<int>(i % 7) - 3;
        values[i] = static_cast<std::bfloat16_t>(scale * centered);
    }
    return values;
}

static_assert(!std::is_copy_constructible_v<Block>);
static_assert(!std::is_copy_assignable_v<Block>);
static_assert(std::is_nothrow_move_constructible_v<Block>);
static_assert(std::is_nothrow_move_assignable_v<Block>);

void TestBatchedMatmulMatchesSequentialMatmul() {
    constexpr int batch_size = 3;
    constexpr int n = 5;
    constexpr int m = 4;

    const std::array<float, batch_size * m> input{
        0.2f, -0.3f, 0.5f, 0.7f,
        -0.4f, 0.1f, 0.8f, -0.2f,
        0.3f, -0.8f, 0.6f, -0.1f,
    };
    const auto weights = MakeValues(n * m, 0.07f);
    std::array<float, batch_size * n> batched{};
    std::array<float, batch_size * n> sequential{};

    matmul_cpu(
        batched.data(), input.data(), weights.data(), n, m, batch_size
    );
    for (int batch = 0; batch < batch_size; ++batch) {
        matmul_cpu(
            sequential.data() + batch * n,
            input.data() + batch * m,
            weights.data(),
            n,
            m
        );
    }

    for (size_t i = 0; i < batched.size(); ++i) {
        if (std::abs(batched[i] - sequential[i]) > 1e-5f) {
            throw std::runtime_error(
                "batched matmul differs from sequential matmul at element " +
                std::to_string(i)
            );
        }
    }
}

void TestBlockMoveTransfersCacheOwnership() {
    const std::string config_json =
        R"({"act_type":"silu","arch":"Qwen3ForCausalLM","attention_bias":false,"bos_token_id":1,"dim":4,"dtype":"bf16","eos_token_id":2,"head_dim":2,"hidden_dim":6,"max_seq_len":8,"n_heads":2,"n_kv_heads":1,"n_layers":1,"norm_eps":0.000001,"qk_norm":true,"rope_theta":10000.0,"rotary_dim":2,"tie_word_embeddings":true,"vocab_size":16})";
    Config config(config_json);
    Block source(&config, Device::CPU);
    source.ResetCache();

    Block moved(std::move(source));
    moved.ResetCache();

    Block assigned(&config, Device::CPU);
    assigned = std::move(moved);
    assigned.ResetCache();
}

void TestConfigParsing() {
    const std::string metadata =
        R"({"act_type":"silu","arch":"Qwen3ForCausalLM","attention_bias":false,"bos_token_id":1,"dim":16,"dtype":"bf16","eos_token_id":2,"head_dim":8,"hidden_dim":32,"max_seq_len":128,"n_heads":2,"n_kv_heads":1,"n_layers":1,"norm_eps":0.000001,"qk_norm":true,"rope_theta":1000000.0,"rotary_dim":8,"tie_word_embeddings":true,"vocab_size":32})";
    const Config config(metadata);
    if (config.arch != "Qwen3ForCausalLM" || config.dim != 16 ||
        config.n_layers != 1 || config.max_seq_len != 128) {
        throw std::runtime_error("model config was parsed incorrectly");
    }
}

void TestGreedySampler() {
    const std::array logits{-2.0f, 4.0f, 1.0f};
    if (Sampler::Greedy(logits) != 1) {
        throw std::runtime_error("greedy sampler selected the wrong token");
    }
}

void TestTemperatureSampler() {
    const std::array logits{0.0f, 1.0f, 2.0f};
    Sampler first(0.6f, 12345);
    Sampler second(0.6f, 12345);
    bool sampled_non_argmax = false;
    for (int i = 0; i < 32; ++i) {
        const auto first_token = first.Sample(logits);
        if (first_token != second.Sample(logits)) {
            throw std::runtime_error("seeded sampling is not reproducible");
        }
        sampled_non_argmax |= first_token != 2;
    }
    if (!sampled_non_argmax) {
        throw std::runtime_error("temperature sampler never explored another token");
    }

    Sampler greedy(0.0f, 12345);
    if (greedy.Sample(logits) != 2) {
        throw std::runtime_error("zero-temperature sampling is not greedy");
    }
}

void TestTokenizer() {
    const Tokenizer tokenizer(
        {"a", "b", "c", "bc", "abc", " ", "<|special|>"},
        {{"b", "c"}, {"a", "bc"}}
    );
    const auto encoded = tokenizer.Encode("abc a<|special|>");
    const std::vector<std::int32_t> expected{4, 5, 0, 6};
    if (encoded != expected) {
        throw std::runtime_error("tokenizer did not apply ranked BPE merges");
    }
    if (tokenizer.Decode(encoded) != "abc a<|special|>") {
        throw std::runtime_error("tokenizer round trip failed");
    }
    if (FormatChatPrompt("Hello") !=
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n") {
        throw std::runtime_error("chat prompt formatting failed");
    }
}

}  // namespace

int main() {
    try {
        TestConfigParsing();
        TestBatchedMatmulMatchesSequentialMatmul();
        TestBlockMoveTransfersCacheOwnership();
        TestGreedySampler();
        TestTemperatureSampler();
        TestTokenizer();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
