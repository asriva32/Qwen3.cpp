#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "model.h"
#include "layers.h"
#include "sampler.h"
#include "tokenizer.h"

namespace {

class MetadataModel final : public Model {
public:
    using Model::Model;

    int initialize_count = 0;
    int reset_count = 0;

    void Prefill(std::span<const std::int32_t>, int, State&) override {}
    void ForwardToken(std::int32_t, int, State&) override {}
    GenerationResult Generate(
        const std::vector<std::int32_t>&,
        size_t,
        bool,
        float,
        std::optional<std::uint64_t>
    ) override {
        return {};
    }

private:
    void InitializeBackend() override { ++initialize_count; }
    void ResetBackend() override { ++reset_count; }
};

template <typename T>
void Write(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::vector<std::bfloat16_t> MakeValues(size_t count, float scale) {
    std::vector<std::bfloat16_t> values(count);
    for (size_t i = 0; i < count; ++i) {
        const auto centered = static_cast<int>(i % 7) - 3;
        values[i] = static_cast<std::bfloat16_t>(scale * centered);
    }
    return values;
}

BlockWeights MakePrefillWeights() {
    BlockWeights weights;
    weights.attn_norm.assign(4, static_cast<std::bfloat16_t>(1.0f));
    weights.q_norm.assign(2, static_cast<std::bfloat16_t>(1.0f));
    weights.k_norm.assign(2, static_cast<std::bfloat16_t>(1.0f));
    weights.wq = MakeValues(16, 0.07f);
    weights.wk = MakeValues(8, 0.05f);
    weights.wv = MakeValues(8, 0.09f);
    weights.wo = MakeValues(16, 0.04f);
    weights.mlp_norm.assign(4, static_cast<std::bfloat16_t>(1.0f));
    weights.w1 = MakeValues(24, 0.03f);
    weights.w2 = MakeValues(24, 0.02f);
    weights.w3 = MakeValues(24, 0.025f);
    return weights;
}

void TestBatchedPrefillMatchesSequentialForward() {
    const std::string config_json =
        R"({"act_type":"silu","arch":"Qwen3ForCausalLM","attention_bias":false,"bos_token_id":1,"dim":4,"dtype":"bf16","eos_token_id":2,"head_dim":2,"hidden_dim":6,"max_seq_len":8,"n_heads":2,"n_kv_heads":1,"n_layers":1,"norm_eps":0.000001,"qk_norm":true,"rope_theta":10000.0,"rotary_dim":2,"tie_word_embeddings":true,"vocab_size":16})";
    Config sequential_config(config_json);
    Config batched_config(config_json);
    Block sequential_block(&sequential_config, MakePrefillWeights());
    Block batched_block(&batched_config, MakePrefillWeights());
    State sequential_state(&sequential_config);
    State batched_state(&batched_config);

    std::array<float, 8> sequential_prefix{
        0.2f, -0.3f, 0.5f, 0.7f,
        -0.4f, 0.1f, 0.8f, -0.2f,
    };
    auto batched_prefix = sequential_prefix;
    for (int token = 0; token < 2; ++token) {
        sequential_block.Forward(
            sequential_prefix.data() + token * 4, token, 0, token, token + 1,
            sequential_state);
        batched_block.Forward(
            batched_prefix.data() + token * 4, token, 0, token, token + 1,
            batched_state);
    }

    std::array<float, 12> sequential{
        0.6f, 0.2f, -0.5f, 0.3f,
        -0.7f, 0.4f, 0.1f, 0.9f,
        0.3f, -0.8f, 0.6f, -0.1f,
    };
    auto batched = sequential;

    batched_block.ForwardPrefill(batched.data(), 3, 2, batched_state);
    for (int token = 0; token < 3; ++token) {
        const auto position = token + 2;
        sequential_block.Forward(
            sequential.data() + token * 4,
            position,
            0,
            position,
            position + 1,
            sequential_state
        );
    }

    for (size_t i = 0; i < batched.size(); ++i) {
        if (std::abs(batched[i] - sequential[i]) > 1e-5f) {
            throw std::runtime_error(
                "batched prefill differs from sequential forward at element " +
                std::to_string(i));
        }
    }

    const auto& batched_cache = batched_block.GetCache();
    const auto& sequential_cache = sequential_block.GetCache();
    for (size_t i = 0; i < 10; ++i) {
        if (batched_cache.k_[i] != sequential_cache.k_[i] ||
            batched_cache.v_[i] != sequential_cache.v_[i]) {
            throw std::runtime_error("batched prefill populated an incorrect KV cache");
        }
    }
}

void TestMetadataLoading() {
    const auto path = std::filesystem::temp_directory_path() / "qwen3_metadata_test.qwen3";
    const std::string metadata =
        R"({"act_type":"silu","arch":"Qwen3ForCausalLM","attention_bias":false,"bos_token_id":1,"dim":16,"dtype":"bf16","eos_token_id":2,"head_dim":8,"hidden_dim":32,"max_seq_len":128,"n_heads":2,"n_kv_heads":1,"n_layers":1,"norm_eps":0.000001,"qk_norm":true,"rope_theta":1000000.0,"rotary_dim":8,"tie_word_embeddings":true,"vocab_size":32})";

    {
        std::ofstream out(path, std::ios::binary);
        const std::array<char, 8> magic{'Q', 'W', 'E', 'N', '3', 'C', 'P', '\0'};
        out.write(magic.data(), magic.size());
        Write(out, std::uint32_t{1});
        Write(out, static_cast<std::uint64_t>(metadata.size()));
        out.write(metadata.data(), static_cast<std::streamsize>(metadata.size()));
        Write(out, std::uint64_t{0});
    }

    MetadataModel model(path.string());
    const Config* config = model.GetConfig();
    if (config->arch != "Qwen3ForCausalLM" || config->dim != 16 ||
        config->n_layers != 1 || !model.GetTensorIndex().empty()) {
        throw std::runtime_error("metadata model was parsed incorrectly");
    }

    model.InitializeInference(64);
    model.ResetInference();
    if (model.GetConfig()->max_seq_len != 64 || model.initialize_count != 1 ||
        model.reset_count != 1) {
        throw std::runtime_error("shared inference lifecycle did not invoke backend hooks");
    }

    bool rejected_invalid_context = false;
    try {
        model.InitializeInference(129);
    } catch (const std::invalid_argument&) {
        rejected_invalid_context = true;
    }
    if (!rejected_invalid_context) {
        throw std::runtime_error("model accepted an unsupported context length");
    }
    std::filesystem::remove(path);
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
        TestMetadataLoading();
        TestBatchedPrefillMatchesSequentialForward();
        TestGreedySampler();
        TestTemperatureSampler();
        TestTokenizer();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
