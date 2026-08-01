#include "inference.h"
#include <omp.h>
#include <tokenizers_cpp.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <immintrin.h>
#include <limits>


// ----- Helpers -----
namespace {

constexpr char kMagic[8] = {'Q', 'W', 'E', 'N', '3', 'C', 'P', '\0'};
constexpr std::uint32_t kVersion = 1;

template <typename T>
auto ReadPod(std::istream& in) -> T {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Unexpected end of Qwen3.bin");
    }
    return value;
}

auto ReadString(std::istream& in, std::uint64_t size) -> std::string {
    if (size > static_cast<std::uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("String record is too large for this platform");
    }

    std::string value(static_cast<size_t>(size), '\0');
    in.read(value.data(), static_cast<std::streamsize>(value.size()));
    if (!in) {
        throw std::runtime_error("Unexpected end of Qwen3.bin while reading string");
    }
    return value;
}

auto CheckSize(size_t value, const std::string& field) -> size_t {
    if (value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(field + " does not fit in size_t");
    }
    return static_cast<size_t>(value);
}

auto SkipBytes(std::istream& in, size_t bytes) -> void {
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("Tensor payload is too large to seek over");
    }
    in.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
    if (!in) {
        throw std::runtime_error("Unexpected end of Qwen3.bin while skipping tensor payload");
    }
}

auto DTypeSize(TensorDType dtype) -> size_t {
    switch (dtype) {
        case TensorDType::Float32:
            return 4;
        case TensorDType::BFloat16:
            return 2;
        case TensorDType::UInt8:
            return 1;
        case TensorDType::Int32:
            return 4;
    }
    throw std::runtime_error("Unknown tensor dtype");
}

auto ShapeElementCount(const std::vector<size_t>& shape) -> size_t {
    auto count{1uz};
    for (const auto dim : shape) {
        if (dim != 0 && count > std::numeric_limits<size_t>::max() / dim) {
            throw std::runtime_error("Tensor shape element count overflow");
        }
        count *= dim;
    }
    return count;
}

auto ValidateTensorByteSize(const TensorInfo& info) -> void {
    const auto element_count = ShapeElementCount(info.shape);
    const auto dtype_size = DTypeSize(info.dtype);
    if (element_count > std::numeric_limits<size_t>::max() / dtype_size) {
        throw std::runtime_error("Tensor byte size overflow for " + info.name);
    }
    const auto expected = element_count * dtype_size;
    if (expected != info.byte_size) {
        throw std::runtime_error("Tensor byte size mismatch for " + info.name);
    }
}

auto JsonValue(const std::string& json, const std::string& key) -> std::string {
    const auto quoted_key = std::string{"\"" + key + "\":"};
    const auto key_pos = json.find(quoted_key);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("Missing metadata key: " + key);
    }
    auto pos = key_pos + quoted_key.size();
    while (pos < json.size() && json[pos] == ' ') {
        ++pos;
    }
    if (pos >= json.size()) {
        throw std::runtime_error("Malformed metadata value for: " + key);
    }

    if (json[pos] == '"') {
        const auto end = json.find('"', pos + 1);
        if (end == std::string::npos) {
            throw std::runtime_error("Unterminated string metadata value for: " + key);
        }
        return json.substr(pos + 1, end - pos - 1);
    }

    const auto end = json.find_first_of(",}", pos);
    if (end == std::string::npos) {
        throw std::runtime_error("Unterminated metadata value for: " + key);
    }
    return json.substr(pos, end - pos);
}
template<SupportedJsonValue T>
auto GetJsonValue(const std::string &json, const std::string& key) -> T {
    const auto value = JsonValue(json, key);
    if constexpr (std::same_as<T, float>) {
        return std::stof(JsonValue(json, key));
    } else if constexpr (std::same_as<T, bool>) {
        if (value == "true") {
            return true;
        } 
        if (value == "false") {
            return false;
        }
        throw std::runtime_error("Metadata value is not bool for: " + key);
    } else if constexpr (std::same_as<T, std::int32_t>) {
        return std::stoi(JsonValue(json, key));
    } else if constexpr (std::same_as<T, std::string>) {
        return value;
    }
}

auto ValidateConfig(const Config &config) -> void {
    if (config.dim <= 0 || config.hidden_dim <= 0 || config.head_dim <= 0 ||
        config.n_heads <= 0 || config.n_kv_heads <= 0 || config.max_seq_len <= 0) {
        throw std::invalid_argument("Block config dimensions must be positive");
    }
    if (config.n_heads % config.n_kv_heads != 0) {
        throw std::invalid_argument("Attention head count must be divisible by KV head count");
    }
    if (config.rotary_dim < 0 || config.rotary_dim > config.head_dim ||
        config.rotary_dim % 2 != 0) {
        throw std::invalid_argument("rotary_dim must be even and no larger than head_dim");
    }
}

auto ParseConfig(const std::string& json) -> Config {
    Config config;
    config.arch                = GetJsonValue<std::string>(json, "arch");
    config.dtype               = GetJsonValue<std::string>(json, "dtype");
    config.act_type            = GetJsonValue<std::string>(json, "act_type");
    config.dim                 = GetJsonValue<std::int32_t>(json, "dim");
    config.hidden_dim          = GetJsonValue<std::int32_t>(json, "hidden_dim");
    config.head_dim            = GetJsonValue<std::int32_t>(json, "head_dim");
    config.n_layers            = GetJsonValue<std::int32_t>(json, "n_layers");
    config.n_heads             = GetJsonValue<std::int32_t>(json, "n_heads");
    config.n_kv_heads          = GetJsonValue<std::int32_t>(json, "n_kv_heads");
    config.vocab_size          = GetJsonValue<std::int32_t>(json, "vocab_size");
    config.max_seq_len         = GetJsonValue<std::int32_t>(json, "max_seq_len");
    config.bos_token_id        = GetJsonValue<std::int32_t>(json, "bos_token_id");
    config.eos_token_id        = GetJsonValue<std::int32_t>(json, "eos_token_id");
    config.rotary_dim          = GetJsonValue<std::int32_t>(json, "rotary_dim");
    config.rope_theta          = GetJsonValue<float>(json, "rope_theta");
    config.norm_eps            = GetJsonValue<float>(json, "norm_eps");
    config.tie_word_embeddings = GetJsonValue<bool>(json, "tie_word_embeddings");
    config.attention_bias      = GetJsonValue<bool>(json, "attention_bias");
    config.qk_norm             = GetJsonValue<bool>(json, "qk_norm");
    ValidateConfig(config);
    return config;
}

auto ReadTensorInfo(std::istream& in) -> TensorInfo {
    auto info = TensorInfo{};
    const auto name_size = ReadPod<std::uint32_t>(in);
    info.name = ReadString(in, name_size);
    info.dtype = static_cast<TensorDType>(ReadPod<std::uint32_t>(in));

    const auto ndim = ReadPod<std::uint32_t>(in);
    info.shape.reserve(ndim);
    for (auto i{0uz}; i < ndim; ++i) {
        info.shape.push_back(CheckSize(ReadPod<size_t>(in), "tensor dimension"));
    }

    info.byte_size = CheckSize(ReadPod<size_t>(in), "tensor byte size");

    auto CheckedTell = [](std::istream& in) {
        const auto pos = in.tellg();
        if (pos < 0) {
            throw std::runtime_error("Failed to query input position");
        }
        return static_cast<size_t>(pos);
    };
    
    info.data_offset = CheckSize(CheckedTell(in), "tensor data offset");
    ValidateTensorByteSize(info);
    return info;
}

auto ValidateSupportedTensorType(const TensorInfo& info) -> void {
    auto IsTokenizerTensor = [](std::string_view name) -> bool {
        return name == "tokenizer.json" || name == "tokenizer.tokens" || name == "tokenizer.offsets";
    };

    if (IsTokenizerTensor(info.name)) {
        if (info.name == "tokenizer.json" && info.dtype == TensorDType::UInt8) {
            return;
        }
        if (info.name == "tokenizer.tokens" && info.dtype == TensorDType::UInt8) {
            return;
        }
        if (info.name == "tokenizer.offsets" && info.dtype == TensorDType::Int32) {
            return;
        }
        throw std::runtime_error("Unsupported tokenizer tensor dtype for " + info.name);
    }

    if (info.dtype != TensorDType::BFloat16) {
        throw std::runtime_error("Only bf16 model tensors are supported: " + info.name);
    }
}

auto ReadTensorBytes(
    const std::string& path,
    const TensorInfo& info,
    size_t element_size
) -> std::vector<std::uint8_t> {
    if (element_size == 0 || info.byte_size % element_size != 0) {
        throw std::runtime_error("Tensor byte size is not aligned: " + info.name);
    }

    if (info.data_offset > static_cast<size_t>(std::numeric_limits<std::streamoff>::max()) ||
        info.byte_size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Tensor is too large to read: " + info.name);
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open model file: " + path);
    }
    in.seekg(static_cast<std::streamoff>(info.data_offset), std::ios::beg);
    if (!in) {
        throw std::runtime_error("Failed to seek to tensor: " + info.name);
    }

    auto data = std::vector<std::uint8_t>(info.byte_size);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(info.byte_size));
    if (!in) {
        throw std::runtime_error("Failed to read tensor payload: " + info.name);
    }
    return data;
}

// ----- SIMD Helpers -----

auto Reduce(__m256 value) -> float {
    const __m128 low  = _mm256_extractf128_ps(value, 0);
    const __m128 high = _mm256_extractf128_ps(value, 1);

    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    return _mm_cvtss_f32(sum);
};

auto LoadBF16(const std::bfloat16_t *p) -> __m256 {
    const __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    __m256i wide = _mm256_cvtepu16_epi32(tmp);
    wide = _mm256_slli_epi32(wide, 16);
    return _mm256_castsi256_ps(wide);
};

auto FastDotProduct(const std::bfloat16_t *w_row, const float *x, int m) -> float {
    __m256 sum_low = _mm256_setzero_ps();
    __m256 sum_high = _mm256_setzero_ps();
    int j = 0;
    for (; j + 16 <= m; j += 16) {
        __m256 wv_low = LoadBF16(w_row + j);
        __m256 wv_high = LoadBF16(w_row + j + 8);
        __m256 xv_low = _mm256_loadu_ps(x + j);
        __m256 xv_high = _mm256_loadu_ps(x + j + 8);

        sum_low = _mm256_fmadd_ps(wv_low, xv_low, sum_low);
        sum_high = _mm256_fmadd_ps(wv_high, xv_high, sum_high);
    }
    float sum = Reduce(_mm256_add_ps(sum_low, sum_high));
    for (; j < m; j++) {
        sum += static_cast<float>(w_row[j]) * x[j];
    }
    return sum;
};

}  // namespace

auto LoadTensorBytes(
    const std::string& path,
    const TensorInfo& info,
    TensorDType expected_dtype,
    size_t element_size
) -> std::vector<std::uint8_t> {
    if (info.dtype != expected_dtype) {
        throw std::runtime_error("Tensor dtype mismatch: " + info.name);
    }
    return ReadTensorBytes(path, info, element_size);
}

auto HasTensor(
    const std::unordered_map<std::string, TensorInfo>& tensors,
    const std::string& name
) -> bool {
    return tensors.find(name) != tensors.end();
}

auto ReadFile(const std::filesystem::path& path) -> std::string {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open tokenizer file: " + path.string());
    }

    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("Failed to query tokenizer file size: " + path.string());
    }
    in.seekg(0, std::ios::beg);

    auto data = std::string(static_cast<size_t>(size), '\0');
    in.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!in) {
        throw std::runtime_error("Failed to read tokenizer file: " + path.string());
    }
    return data;
}

auto FindTokenizerJson(const std::string& model_path) -> std::filesystem::path {
    const auto path = std::filesystem::path(model_path);
    const auto parent = path.parent_path().empty()
        ? std::filesystem::current_path()
        : path.parent_path();

    const auto sibling = parent / "tokenizer.json";
    if (std::filesystem::exists(sibling)) {
        return sibling;
    }

    const auto project_default = parent / "Qwen3-0.6B" / "tokenizer.json";
    if (std::filesystem::exists(project_default)) {
        return project_default;
    }

    return {};
}

// ----- Tokenizer -----
Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;

Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
auto Tokenizer::operator=(Tokenizer&&) noexcept -> Tokenizer& = default;

auto Tokenizer::LoadFromJsonBlob(const std::string& json_blob) -> void {
    backend_ = tokenizers::Tokenizer::FromBlobJSON(json_blob);
    if (!backend_) {
        throw std::runtime_error("Failed to initialize tokenizer backend");
    }
}

auto Tokenizer::LoadFromJsonFile(const std::string& path) -> void {
    LoadFromJsonBlob(ReadFile(path));
}

auto Tokenizer::SetSpecialTokenIds(const Config& config) -> void {
    bos_token_id_ = config.bos_token_id;
    eos_token_id_ = config.eos_token_id;
}

auto Tokenizer::HasBackend() const -> bool {
    return backend_ != nullptr;
}

auto Tokenizer::VocabSize() const -> size_t {
    if (!offsets.empty()) {
        return offsets.size();
    }
    if (backend_) {
        return backend_->GetVocabSize();
    }
    return 0;
}

auto Tokenizer::BosTokenId() const -> std::int32_t {
    return bos_token_id_;
}

auto Tokenizer::EosTokenId() const -> std::int32_t {
    return eos_token_id_;
}

auto Tokenizer::Token(std::int32_t id) const -> std::string {
    if (backend_) {
        const auto token = backend_->IdToToken(id);
        if (!token.empty()) {
            return token;
        }
    }

    if (id < 0 || static_cast<size_t>(id) >= offsets.size()) {
        throw std::out_of_range("Token id is out of range");
    }
    const auto offset = offsets[id];
    if (offset < 0 || static_cast<size_t>(offset) >= tokens.size()) {
        throw std::runtime_error("Tokenizer offset is out of range");
    }
    return std::string(tokens.data() + offset);
}

auto Tokenizer::Encode(std::string_view prompt) const -> std::vector<std::int32_t> {
    if (!backend_) {
        throw std::runtime_error("Tokenizer backend is not initialized");
    }
    return backend_->Encode(std::string(prompt));
}

auto Tokenizer::Decode(std::span<const std::int32_t> token_ids) const -> std::string {
    if (!backend_) {
        throw std::runtime_error("Tokenizer backend is not initialized");
    }
    return backend_->Decode(std::vector<std::int32_t>(token_ids.begin(), token_ids.end()));
}

// ----- Qwen3 -----

Qwen3::Qwen3(const std::string& path, int context_length) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open model file: " + path);
    }

    char magic[8] = {};
    in.read(magic, sizeof(magic));
    if (!in || !std::equal(std::begin(magic), std::end(magic), std::begin(kMagic))) {
        throw std::runtime_error("Invalid Qwen3 model file magic");
    }

    const auto version = ReadPod<std::uint32_t>(in);
    if (version != kVersion) {
        throw std::runtime_error("Unsupported Qwen3 model file version");
    }

    const auto metadata_size = ReadPod<size_t>(in);
    const auto metadata_json = ReadString(in, metadata_size);
    inference_config_ = std::make_shared<Config>(ParseConfig(metadata_json));
    if (inference_config_->dtype != "bf16") {
        throw std::runtime_error("Only bf16 Qwen3.bin files are supported");
    }

    tokenizer_ = Tokenizer{};
    tokenizer_.SetSpecialTokenIds(*inference_config_);
    model_path_ = path;

    const auto tensor_count = ReadPod<size_t>(in);
    for (auto i{0uz}; i < tensor_count; ++i) {
        TensorInfo info = ReadTensorInfo(in);
        ValidateSupportedTensorType(info);
        if (!tensors.emplace(info.name, info).second) {
            throw std::runtime_error("Duplicate tensor record: " + info.name);
        }
        SkipBytes(in, info.byte_size);
    }

    if (HasTensor(tensors, "tokenizer.tokens")) {
        auto token_bytes = LoadTensor<std::uint8_t>("tokenizer.tokens");
        tokenizer_.tokens.assign(
            reinterpret_cast<const char*>(token_bytes.ptr()),
            reinterpret_cast<const char*>(token_bytes.ptr() + token_bytes.data.size())
        );
    }

    if (HasTensor(tensors, "tokenizer.offsets")) {
        auto token_offsets = LoadTensor<std::int32_t>("tokenizer.offsets");
        tokenizer_.offsets = std::move(token_offsets.data);
    }

    if (HasTensor(tensors, "tokenizer.json")) {
        auto tokenizer_json = LoadTensor<std::uint8_t>("tokenizer.json");
        tokenizer_.LoadFromJsonBlob(std::string(
            reinterpret_cast<const char*>(tokenizer_json.ptr()),
            tokenizer_json.data.size()
        ));
    } else {
        const auto tokenizer_path = FindTokenizerJson(path);
        if (!tokenizer_path.empty()) {
            tokenizer_.LoadFromJsonFile(tokenizer_path.string());
        }
        // Should we throw ?
    }
    InitializeInference(context_length);
}

auto Qwen3::GetConfig() const -> const Config* {
    return inference_config_.get();
}

auto Qwen3::GetTensorIndex() const -> const std::unordered_map<std::string, TensorInfo>& {
    return tensors;
}

auto Qwen3::GetTokenizer() const -> const Tokenizer& {
    return tokenizer_;
}

template<typename T>
auto Qwen3::LoadFloatData(const std::string& name) const -> std::vector<T> {
    return LoadTensor<T>(name).data;
}

auto Qwen3::InitializeInference(int context_length) -> void {
    if (context_length <= 0 || context_length > inference_config_->max_seq_len) {
        throw std::invalid_argument("Context length is outside the model's supported range");
    }
    // override context_length
    inference_config_->max_seq_len = context_length;

    // If quantized further will have to figure this out
    using dtype = std::bfloat16_t;

    embedding_  = LoadFloatData<dtype>("model.embed.weight");
    final_norm_ = LoadFloatData<dtype>("model.norm.weight");
    if (inference_config_->tie_word_embeddings) {
        output_.clear();
    } else {
        output_ = LoadFloatData<dtype>("model.output.weight");
    }

    const auto dim = static_cast<size_t>(inference_config_->dim);
    const auto vocab_size = static_cast<size_t>(inference_config_->vocab_size);
    if (embedding_.size() != vocab_size * dim || final_norm_.size() != dim ||
        (!inference_config_->tie_word_embeddings && output_.size() != vocab_size * dim)) {
        throw std::runtime_error("Invalid embedding, final norm, or output tensor shape");
    }

    blocks.reserve(inference_config_->n_layers);
    for (auto layer{0}; layer < inference_config_->n_layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        BlockWeights weights;
        weights.attn_norm = LoadFloatData<dtype>(prefix + ".attn.norm.weight");
        weights.q_norm    = LoadFloatData<dtype>(prefix + ".attn.q_norm.weight");
        weights.k_norm    = LoadFloatData<dtype>(prefix + ".attn.k_norm.weight");
        weights.wq        = LoadFloatData<dtype>(prefix + ".attn.wq.weight");
        weights.wk        = LoadFloatData<dtype>(prefix + ".attn.wk.weight");
        weights.wv        = LoadFloatData<dtype>(prefix + ".attn.wv.weight");
        weights.wo        = LoadFloatData<dtype>(prefix + ".attn.wo.weight");
        weights.mlp_norm  = LoadFloatData<dtype>(prefix + ".mlp.norm.weight");
        weights.w1        = LoadFloatData<dtype>(prefix + ".mlp.w1.weight");
        weights.w2        = LoadFloatData<dtype>(prefix + ".mlp.w2.weight");
        weights.w3        = LoadFloatData<dtype>(prefix + ".mlp.w3.weight");
        blocks.emplace_back(inference_config_, std::move(weights));
    }

    for (const auto &block: blocks) {
        block.ValidateWeights();
    }

    hidden_state_.resize(inference_config_->dim);
    normalized_state_.resize(inference_config_->dim);
    logits_.resize(inference_config_->vocab_size);
    // kind of unnecessary
    ResetInference();
}

auto Qwen3::ResetInference() -> void {
    for (Block& block : blocks) {
        block.ResetCache();
    }
    std::fill(hidden_state_.begin(), hidden_state_.end(), 0.0f);
    std::fill(normalized_state_.begin(), normalized_state_.end(), 0.0f);
    std::fill(logits_.begin(), logits_.end(), 0.0f);
}

auto Qwen3::ForwardToken(std::int32_t token, int pos, State &state) -> const std::vector<float>& {
    const auto& config = *inference_config_;
    if (token < 0 || token >= config.vocab_size) {
        throw std::out_of_range("Token id is outside the vocabulary");
    }
    if (pos < 0) {
        throw std::out_of_range("Token position must not be negative");
    }

    const auto* embedding_row =
        embedding_.data() + static_cast<size_t>(token) * config.dim;
    std::copy_n(embedding_row, config.dim, hidden_state_.begin());

    constexpr int kAttentionSinks = 4;
    const auto context_length = config.max_seq_len;
    const auto num_sink = pos >= context_length
        ? std::min(kAttentionSinks, context_length - 1)
        : 0;
    const auto kv_pos = pos < context_length
        ? pos
        : num_sink + (pos - num_sink) % (context_length - num_sink);
    const auto kv_len = std::min(pos + 1, context_length);

    for (Block& block : blocks) {
        block.forward(hidden_state_.data(), pos, num_sink, kv_pos, kv_len, state);
    }

    RmsNorm(
        normalized_state_.data(),
        hidden_state_.data(),
        final_norm_.data(),
        config.norm_eps,
        config.dim
    );
    const auto classifier = config.tie_word_embeddings
        ? embedding_.data()
        : output_.data();
    MatMul(
        logits_.data(),
        normalized_state_.data(),
        classifier,
        config.vocab_size,
        config.dim
    );
    return logits_;
}

auto Qwen3::Generate(
    std::string_view prompt,
    bool apply_chat_template,
    size_t max_generated_tokens,
    bool stop_on_eos,
    const std::function<void(std::span<const std::int32_t>)>& on_tokens
) -> GenerationResult {
    auto model_input = std::string{};
    if (apply_chat_template) {
        model_input = "<|im_start|>user\n";
        model_input.append(prompt);
        model_input += "<|im_end|>\n<|im_start|>assistant\n";
    } else {
        model_input.assign(prompt);
    }

    auto prompt_tokens = tokenizer_.Encode(model_input);
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(inference_config_->bos_token_id);
    }

    GenerationResult result;
    result.stats.prompt_tokens = prompt_tokens.size();

    auto pos = 0;
    const std::vector<float>* logits = nullptr;
    const auto prefill_start = std::chrono::steady_clock::now();
    State state(inference_config_);
    // TODO: implement batched prefill
    // [tokens, dim]
    for (const auto token : prompt_tokens) {
        logits = &ForwardToken(token, pos++, state);
    }
    // generated first token
    const auto prefill_end = std::chrono::steady_clock::now();
    result.stats.prefill_seconds =
        std::chrono::duration<double>(prefill_end - prefill_start).count();

    const auto decode_start = std::chrono::steady_clock::now();
    // Autoregressive loop
    while (result.tokens.size() < max_generated_tokens) {
        const auto token = Sampler::Greedy(*logits);
        if (stop_on_eos && token == inference_config_->eos_token_id) {
            result.stats.stopped_on_eos = true;
            break;
        }
        result.tokens.push_back(token);
        if (on_tokens) {
            on_tokens(result.tokens);
        }
        logits = &ForwardToken(token, pos++, state);
    }
    const auto decode_end = std::chrono::steady_clock::now();
    result.stats.generated_tokens = result.tokens.size();
    result.stats.reached_token_limit =
        !result.stats.stopped_on_eos &&
        result.tokens.size() == max_generated_tokens;
    result.stats.decode_seconds =
        std::chrono::duration<double>(decode_end - decode_start).count();
    result.text = tokenizer_.Decode(result.tokens);
    return result;
}

// TODO: maybe add temperature based sampling
auto Sampler::Greedy(std::span<const float> logits) -> std::int32_t {
    if (logits.empty()) {
        throw std::invalid_argument("Cannot sample empty logits");
    }
    return static_cast<std::int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
}

auto GenerationStats::PrefillTokensPerSecond() const -> double {
    return prefill_seconds > 0.0 ? prompt_tokens / prefill_seconds : 0.0;
}

auto GenerationStats::DecodeTokensPerSecond() const -> double {
    return decode_seconds > 0.0 ? generated_tokens / decode_seconds : 0.0;
}

// ----- Block -----
Block::Block(const std::shared_ptr<Config>& config) : config(config) {
    
    const auto q_dim = config->n_heads * config->head_dim;
    const auto kv_dim = config->n_kv_heads * config->head_dim;
    cache = KVCache(config->max_seq_len, config->n_kv_heads, config->head_dim);
}

Block::Block(const std::shared_ptr<Config>& config, BlockWeights weights)
    : Block(config) {
    this->weights = std::move(weights);
}

auto Block::ResetCache() -> void {
    std::fill(cache.k_.begin(), cache.k_.end(), 0.0f);
    std::fill(cache.v_.begin(), cache.v_.end(), 0.0f);
}

auto Block::ValidateWeights() const -> void {
    const auto dim = static_cast<size_t>(config->dim);
    const auto hidden_dim = static_cast<size_t>(config->hidden_dim);
    const auto head_dim = static_cast<size_t>(config->head_dim);
    const auto q_dim = static_cast<size_t>(config->n_heads) * head_dim;
    const auto kv_dim = static_cast<size_t>(config->n_kv_heads) * head_dim;

    const auto require_size = [](const std::vector<std::bfloat16_t>& weight, size_t expected,
                                 const char* name) {
        if (weight.size() != expected) {
            throw std::invalid_argument(
                std::string("Invalid ") + name + " size: expected " +
                std::to_string(expected) + ", got " + std::to_string(weight.size()));
        }
    };

    require_size(weights.attn_norm, dim, "attn_norm");
    require_size(weights.q_norm, head_dim, "q_norm");
    require_size(weights.k_norm, head_dim, "k_norm");
    require_size(weights.wq, q_dim * dim, "wq");
    require_size(weights.wk, kv_dim * dim, "wk");
    require_size(weights.wv, kv_dim * dim, "wv");
    require_size(weights.wo, dim * q_dim, "wo");
    require_size(weights.mlp_norm, dim, "mlp_norm");
    require_size(weights.w1, hidden_dim * dim, "w1");
    require_size(weights.w2, dim * hidden_dim, "w2");
    require_size(weights.w3, hidden_dim * dim, "w3");
}

auto Block::forward(
    float* x,
    int pos,
    int num_sink,
    int kv_pos,
    int kv_len,
    State &state
) -> void {
    if (!x) {
        throw std::invalid_argument("Block input must not be null");
    }
    auto& q = state.q;
    auto& k = state.k;
    auto& v = state.v;
    auto& norm_buffer = state.norm_buffer;
    auto& attn_scores = state.attn_scores;
    auto& attn_output = state.attn_output;
    auto& projected = state.projected;

    const auto& c = *config;
    if (pos < 0 || kv_len <= 0 || kv_len > c.max_seq_len || kv_pos < 0 ||
        kv_pos >= kv_len || num_sink < 0 || num_sink > kv_len ||
        num_sink >= c.max_seq_len || kv_pos < num_sink) {
        throw std::out_of_range("Invalid position or KV-cache range");
    }

    const auto q_dim = c.n_heads * c.head_dim;
    const auto kv_dim = c.n_kv_heads * c.head_dim;

    RmsNorm(norm_buffer.data(), x, weights.attn_norm.data() , c.norm_eps, c.dim);
    MatMul(q.data(), norm_buffer.data(), weights.wq.data(), q_dim, c.dim);
    MatMul(k.data(), norm_buffer.data(), weights.wk.data(), kv_dim, c.dim);
    MatMul(v.data(), norm_buffer.data(), weights.wv.data(), kv_dim, c.dim);

    for (auto head = 0; head < c.n_heads; ++head) {
        auto* q_head = q.data() + head * c.head_dim;
        RmsNorm(q_head, q_head, weights.q_norm.data(), c.norm_eps, c.head_dim);
    }
    for (auto head = 0; head < c.n_kv_heads; ++head) {
        auto* k_head = k.data() + head * c.head_dim;
        RmsNorm(k_head, k_head, weights.k_norm.data(), c.norm_eps, c.head_dim);
    }

    ApplyRotaryEmb(q.data(), q_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim);
    ApplyRotaryEmb(k.data(), kv_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim);

    for (auto i = 0; i < kv_dim; i++) {
        cache.k_[i + kv_pos * kv_dim] = static_cast<std::bfloat16_t>(k[i]);
        cache.v_[i + kv_pos * kv_dim] = static_cast<std::bfloat16_t>(v[i]);
    }

    // Keep sink tokens at a constant relative distance after the ring buffer fills.
    for (auto sink = 0; sink < num_sink; ++sink) {
        for (auto i = 0; i < kv_dim; i++) {
            k[i] = static_cast<float>(cache.k_[sink * kv_dim + i]);
        }
        ApplyRotaryEmb(k.data(), kv_dim, c.head_dim, 1, c.rope_theta, c.rotary_dim);

        for (auto i = 0; i < kv_dim; i++) {
            cache.k_[sink * kv_dim + i] = static_cast<std::bfloat16_t>(k[i]);
        }
    }

    const auto queries_per_kv_head = c.n_heads / c.n_kv_heads;
    int head;
    #pragma omp parallel for private(head)
    for (head = 0; head < c.n_heads; ++head) {
        const auto kv_head = head / queries_per_kv_head;
        FastAttn(
            attn_output.data() + head * c.head_dim,
            attn_scores.data() + static_cast<size_t>(head) * c.max_seq_len,
            q.data() + head * c.head_dim,
            cache.k_.data() + kv_head * c.head_dim,
            cache.v_.data() + kv_head * c.head_dim,
            c.head_dim,
            c.n_kv_heads,
            kv_len
        );
    }

    MatMul(projected.data(), attn_output.data(), weights.wo.data(), c.dim, q_dim);
    for (auto i = 0; i < c.dim; ++i) {
        x[i] += projected[i];
    }

    RmsNorm(norm_buffer.data(), x, weights.mlp_norm.data(), c.norm_eps, c.dim);
    FeedForwardNetwork(
        projected.data(),
        state.lin1.data(),
        state.lin2.data(),
        norm_buffer.data(),
        weights.w1.data(),
        weights.w2.data(),
        weights.w3.data(),
        c.hidden_dim,
        c.dim
    );
    for (auto i = 0; i < c.dim; ++i) {
        x[i] += projected[i];
    }
}

// ----- InferenceState -----

State::State(const std::shared_ptr<Config> &c) {
    lin1.resize(c->hidden_dim);
    lin2.resize(c->hidden_dim);
    // Block
    const auto q_dim = c->n_heads * c->head_dim;
    const auto kv_dim = c->n_kv_heads * c->head_dim;
    norm_buffer.resize(c->dim);
    q.resize(q_dim);
    k.resize(kv_dim);
    v.resize(kv_dim);
    attn_output.resize(q_dim);
    projected.resize(c->dim);
    attn_scores.resize(static_cast<size_t>(c->n_heads) * c->max_seq_len);
}

// ----- Layers -----

auto RmsNorm(
    float *out,
    const float *x, 
    const std::bfloat16_t *w, 
    float eps,
    int n
) -> void {
    
    __m256 rms_vec = _mm256_setzero_ps();
    auto i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        rms_vec = _mm256_fmadd_ps(x_vec, x_vec, rms_vec);
    }
    auto rms = Reduce(rms_vec);
    for (; i < n; i++) {
        rms += x[i] * x[i];
    }
    rms = sqrtf(rms / n + eps);
    
    const auto inverse_rms = 1.0f / rms;
    __m256 inverse_rms_vec = _mm256_set1_ps(inverse_rms);
    auto j = 0;
    for (; j + 8 <= n; j += 8) {
        __m256 value  = _mm256_setzero_ps();
        __m256 w_vec  = LoadBF16(w + j);
        __m256 x_vec  = _mm256_loadu_ps(x + j);

        value = _mm256_mul_ps(x_vec, inverse_rms_vec);
        value = _mm256_mul_ps(value, w_vec);
        _mm256_storeu_ps(out + j, value);
    }
    for (; j < n; j++) {
        out[j] = x[j] * inverse_rms * static_cast<float>(w[j]);
    }

}

auto Softmax(
    float *out,
    const float *x, 
    int n
) -> void {

    auto mx_score = std::numeric_limits<float>::lowest();
    for (int i = 0; i < n; i++) {
        mx_score = std::max(mx_score, x[i]);
    }
    auto score = 0.0f;

    for (auto i = 0; i < n; i++) {
        out[i] = expf(x[i] - mx_score);
        score += out[i];
    }

    for (auto i = 0; i < n; i++) {
        out[i] /= score;
    }

}

auto Silu(float x) -> float {
    return x / (1.0f + expf(-x));
}

auto MatMul(
    float *out,
    const float* x, 
    const std::bfloat16_t* w,
    int n,
    int m
) -> void {
    // (n, m) x (m, ) = (n, )
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < n; i++) {
        const auto *w_row = w + static_cast<size_t>(i) * m;
        out[i] = FastDotProduct(w_row, x, m);
    }
}

auto ApplyRotaryEmb(
    float *out,
    int d,
    int head_dim,
    int pos,
    float theta,
    int rotary_dim
) -> void {
    const auto rotary_half = rotary_dim / 2;
    for (int head_start = 0; head_start < d; head_start += head_dim) {
        for (int i = 0; i < rotary_half; ++i) {
            const auto freq = 1.0f / powf(theta, 2.0f * i / rotary_dim);
            const auto angle = pos * freq;
            const auto cosine = cosf(angle);
            const auto sine = sinf(angle);
            const auto first_index = head_start + i;
            const auto second_index = first_index + rotary_half;
            const auto first = out[first_index];
            const auto second = out[second_index];
            out[first_index] = first * cosine - second * sine;
            out[second_index] = second * cosine + first * sine;
        }
    }
}

auto FeedForwardNetwork(
    float *out,
    float *lin1,
    float *lin2,
    const float *x,
    const std::bfloat16_t *w1,
    const std::bfloat16_t *w2,
    const std::bfloat16_t *w3,
    int hidden_dim,
    int dim
) -> void {
    
    MatMul(lin1, x, w1, hidden_dim, dim);
    MatMul(lin2, x, w3, hidden_dim, dim);
    
    // this is like siluAndMul (?) 
    for (auto i = 0; i < hidden_dim; i++) {
        lin1[i] = Silu(lin1[i]) * lin2[i];
    }

    MatMul(out, lin1, w2, dim, hidden_dim);
    
}

auto FastAttn(
    float *out, // (dim, )
    float *atth, // (kv_len, ) - to hold attn scores
    const float *q, // (head_dim, )
    const std::bfloat16_t *k, // (kv_len, n_kv_heads, head_dim)
    const std::bfloat16_t *v, // (kv_len, n_kv_heads, head_dim)
    int head_dim,
    int n_kv_heads,
    int kv_len
) -> void {
    const auto stride = n_kv_heads * head_dim;
    const auto sqrt_head_dim = sqrtf(head_dim);
    for (auto i = 0; i < kv_len; i++) {
        const auto *k_row = k + static_cast<size_t>(i) * stride;  
        auto score = FastDotProduct(k_row, q, head_dim);
        score /= sqrt_head_dim;
        atth[i] = score;
    }

    Softmax(atth, atth, kv_len);
    for (auto i = 0; i < head_dim; i++) {
        out[i] = 0.0f;
    }
    // Vectorize this
    for (auto token = 0; token < kv_len; ++token) {
        const float score = atth[token];
        const auto* value = v + token * stride;
        __m256 score_vec = _mm256_set1_ps(score);

        auto i = 0;
        for (; i + 8 <= head_dim; i += 8) {
            __m256 value_vec = LoadBF16(value + i);
            __m256 out_vec = _mm256_loadu_ps(out + i);
            out_vec = _mm256_add_ps(_mm256_mul_ps(value_vec, score_vec), out_vec);
            _mm256_storeu_ps(out + i, out_vec);
        }

        for (; i < head_dim; ++i) {
            out[i] += score * static_cast<float>(value[i]);
        }
    }
}

auto Attn(
    float *out, // (dim, )
    float *atth, // (kv_len, ) - to hold attn scores
    const float *q, // (head_dim, )
    const std::bfloat16_t *k, // (kv_len, n_kv_heads, head_dim)
    const std::bfloat16_t *v, // (kv_len, n_kv_heads, head_dim)
    int head_dim,
    int n_kv_heads,
    int kv_len
) -> void {
    const auto stride = n_kv_heads * head_dim;
    const auto sqrt_head_dim = sqrtf(head_dim);
    for (auto i = 0; i < kv_len; i++) {
        auto score = 0.0f;
        for (auto j = 0; j < head_dim; j++) {
            score += q[j] * static_cast<float>(k[i * stride + j]);
        }
        score /= sqrt_head_dim;
        atth[i] = score;
    }

    Softmax(atth, atth, kv_len);

    for (auto i = 0; i < head_dim; i++) {
        auto res = 0.0f;
        for (auto j = 0; j < kv_len; j++) {
            res += atth[j] * static_cast<float>(v[j * stride + i]);
        }
        out[i] = res;
    }
}