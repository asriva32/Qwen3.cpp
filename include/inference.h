#ifndef INFERENCE_H
#define INFERENCE_H

#include <cstdint>
#include <concepts>
#include <cstring>
#include <string>
#include <stdexcept>
#include <stdfloat>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cmath>
#include <memory>
#include <span>

namespace tokenizers {
    class Tokenizer;
}
// needs to match convert.py
enum class TensorDType : std::uint32_t {
    Float32 = 1,
    BFloat16 = 3,
    UInt8 = 4,
    Int32 = 5,
};

struct TensorInfo {
    std::string name;
    TensorDType dtype;
    std::vector<std::size_t> shape;
    std::size_t data_offset = 0;
    std::size_t byte_size = 0;
};

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

class Config {
public:
    std::string arch;
    std::string dtype;
    std::string act_type;

    std::int32_t dim = 0;
    std::int32_t hidden_dim = 0;
    std::int32_t head_dim = 0;
    std::int32_t n_layers = 0;
    std::int32_t n_heads = 0;
    std::int32_t n_kv_heads = 0;
    std::int32_t vocab_size = 0;
    std::int32_t max_seq_len = 0;
    std::int32_t bos_token_id = 0;
    std::int32_t eos_token_id = 0;
    std::int32_t rotary_dim = 0;

    float rope_theta = 0.0f;
    float norm_eps = 0.0f;

    bool tie_word_embeddings = false;
    bool attention_bias = false;
    bool qk_norm = false;
};

// Avoid as many allocations as possible
struct State {
    // From RMSNorm
    std::vector<float> lin1;
    std::vector<float> lin2;
    // From Block
    std::vector<float> norm_buffer;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> attn_output;
    std::vector<float> projected;
    std::vector<float> attn_scores;
    State(const std::shared_ptr<Config>& c);
};

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    std::vector<char> tokens;
    std::vector<std::int32_t> offsets;

    void LoadFromJsonBlob(const std::string& json_blob);
    void LoadFromJsonFile(const std::string& path);
    void SetSpecialTokenIds(const Config& config);
    bool HasBackend() const;
    std::size_t VocabSize() const;
    std::int32_t BosTokenId() const;
    std::int32_t EosTokenId() const;

    std::string Token(std::int32_t id) const;
    std::vector<std::int32_t> Encode(std::string_view prompt) const;
    std::string Decode(std::span<const std::int32_t> token_ids) const;

private:
    std::int32_t bos_token_id_ = 0;
    std::int32_t eos_token_id_ = 0;
    std::unique_ptr<tokenizers::Tokenizer> backend_;
};

struct KVCache {
    std::vector<std::bfloat16_t> k_; // (seq_len, n_kv_heads * head_dim)
    std::vector<std::bfloat16_t> v_; // (seq_len, n_kv_heads * head_dim)
    int seq_len = 0;
    int n_kv_heads = 0;
    int head_dim = 0;

    KVCache() = default;
    KVCache(int seq_len, int n_kv_heads, int head_dim) : 
        seq_len(seq_len), n_kv_heads(n_kv_heads), head_dim(head_dim) {
            k_.resize(seq_len * n_kv_heads * head_dim);
            v_.resize(seq_len * n_kv_heads * head_dim);
        }

    std::bfloat16_t *GetKCache() {
        return k_.data();
    }

    std::bfloat16_t *GetVCache() {
        return v_.data();
    }
};

struct BlockWeights {
    using bf16 = std::bfloat16_t;
    std::vector<bf16> attn_norm;
    std::vector<bf16> q_norm;
    std::vector<bf16> k_norm;
    std::vector<bf16> wq;
    std::vector<bf16> wk;
    std::vector<bf16> wv;
    std::vector<bf16> wo;
    std::vector<bf16> mlp_norm;
    std::vector<bf16> w1;
    std::vector<bf16> w2;
    std::vector<bf16> w3;
};

class Block {
public:
    void forward(float* x, int pos, int num_sink, int kv_pos, int kv_len, State &state);

    explicit Block(const std::shared_ptr<Config>& config);
    Block(const std::shared_ptr<Config>& config, BlockWeights weights);

    void ResetCache();

    Config* GetConfig() {
        return config.get();
    }

    const KVCache& GetCache() const {
        return cache;
    }
private:
    void ValidateWeights() const;

    std::shared_ptr<Config> config;
    BlockWeights weights;
    KVCache cache;
};

class Transformer {
public:
    std::unordered_map<std::string, TensorInfo> tensors;
    std::vector<Block> blocks;
    std::shared_ptr<Config> config;
};

class Sampler {
public:
    static std::int32_t Greedy(std::span<const float> logits);
};

struct GenerationStats {
    std::size_t prompt_tokens = 0;
    std::size_t generated_tokens = 0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    bool stopped_on_eos = false;
    bool reached_token_limit = false;

    double PrefillTokensPerSecond() const;
    double DecodeTokensPerSecond() const;
};

struct GenerationResult {
    std::string text;
    std::vector<std::int32_t> tokens;
    GenerationStats stats;
};

class Qwen3 {
public:
    void Load(const std::string& path);
    void InitializeInference(int context_length = 512);
    void ResetInference();

    const std::vector<float>& ForwardToken(std::int32_t token, int pos, State &state);
    GenerationResult Generate(
        std::string_view prompt,
        bool apply_chat_template = true,
        std::size_t max_generated_tokens = 512,
        bool stop_on_eos = true
    );

    template <SupportedTensorElement T>
    Tensor<T> LoadTensor(const std::string& name) const;

    const Config* GetConfig() const;
    const std::unordered_map<std::string, TensorInfo>& GetTensorIndex() const;
    const Tokenizer& GetTokenizer() const;

private:
    std::vector<std::bfloat16_t> LoadFloatData(const std::string& name) const;

    std::string model_path_;
    Transformer transformer_;
    Tokenizer tokenizer_;
    std::shared_ptr<Config> inference_config_;
    std::vector<std::bfloat16_t> embedding_;
    std::vector<std::bfloat16_t> final_norm_;
    std::vector<std::bfloat16_t> output_;
    std::vector<float> hidden_state_;
    std::vector<float> normalized_state_;
    std::vector<float> logits_;
    bool inference_initialized_ = false;
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
    std::size_t element_size
);


template <SupportedTensorElement T>
Tensor<T> Qwen3::LoadTensor(const std::string& name) const {
    const auto it = transformer_.tensors.find(name);
    if (it == transformer_.tensors.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }

    const TensorInfo& info = it->second;
    const std::vector<std::uint8_t> bytes = LoadTensorBytes(
        model_path_,
        info,
        ExpectedDType<T>(),
        sizeof(T)
    );

    std::vector<T> data(bytes.size() / sizeof(T));
    std::memcpy(data.data(), bytes.data(), bytes.size());
    return Tensor<T>{info, std::move(data)};
}

void RmsNorm(const float *x, const std::bfloat16_t *weights, float *out, float eps, int n);
void Softmax(const float *x, float *out, int n);
float Gelu(float x);
float Silu(float x);
void MatMul(float *out, const float *x, const std::bfloat16_t *y, int n, int m);
void ApplyRotaryEmb(float *out, int d, int head_dim, int pos, float theta, int rotary_dim);
void FeedForwardNetwork(float *out, float *lin1, float *lin2, const float *x, const std::bfloat16_t *w1, const std::bfloat16_t *w2, const std::bfloat16_t *w3, int hidden_dim, int dim);
void Attn(float *out, float *atth, const float *q, const std::bfloat16_t *k, const std::bfloat16_t *v, int head_dim, int n_kv_heads, int kv_len);
#endif
