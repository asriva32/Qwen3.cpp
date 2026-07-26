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
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("String record is too large for this platform");
    }

    std::string value(static_cast<std::size_t>(size), '\0');
    in.read(value.data(), static_cast<std::streamsize>(value.size()));
    if (!in) {
        throw std::runtime_error("Unexpected end of Qwen3.bin while reading string");
    }
    return value;
}



auto CheckedSize(std::size_t value, const std::string& field) -> std::size_t {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(field + " does not fit in size_t");
    }
    return static_cast<std::size_t>(value);
}

auto SkipBytes(std::istream& in, std::size_t bytes) -> void {
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("Tensor payload is too large to seek over");
    }
    in.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
    if (!in) {
        throw std::runtime_error("Unexpected end of Qwen3.bin while skipping tensor payload");
    }
}

auto DTypeSize(TensorDType dtype) -> std::size_t {
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

auto ShapeElementCount(const std::vector<std::size_t>& shape) -> std::size_t {
    auto count{1uz};
    for (const auto dim : shape) {
        if (dim != 0 && count > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("Tensor shape element count overflow");
        }
        count *= dim;
    }
    return count;
}

auto ValidateTensorByteSize(const TensorInfo& info) -> void {
    const auto element_count = ShapeElementCount(info.shape);
    const auto dtype_size = DTypeSize(info.dtype);
    if (element_count > std::numeric_limits<std::size_t>::max() / dtype_size) {
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

auto JsonI32(const std::string& json, const std::string& key) -> std::int32_t {
    const auto value = std::stoi(JsonValue(json, key));
    return value;
}

auto JsonFloat(const std::string& json, const std::string& key) -> float {
    return std::stof(JsonValue(json, key));
}

auto JsonBool(const std::string& json, const std::string& key) -> bool {
    const auto value = JsonValue(json, key);
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    throw std::runtime_error("Metadata value is not bool for: " + key);
}

// TODO
auto ValidateConfig(const std::shared_ptr<Config> &config) -> void {
    if (config->dim <= 0 || config->hidden_dim <= 0 || config->head_dim <= 0 ||
        config->n_heads <= 0 || config->n_kv_heads <= 0 || config->max_seq_len <= 0) {
        throw std::invalid_argument("Block config dimensions must be positive");
    }
    if (config->n_heads % config->n_kv_heads != 0) {
        throw std::invalid_argument("Attention head count must be divisible by KV head count");
    }
    if (config->rotary_dim < 0 || config->rotary_dim > config->head_dim ||
        config->rotary_dim % 2 != 0) {
        throw std::invalid_argument("rotary_dim must be even and no larger than head_dim");
    }
}

auto ParseConfig(const std::string& json) -> Config {
    Config config;
    config.arch = JsonValue(json, "arch");
    config.dtype = JsonValue(json, "dtype");
    config.act_type = JsonValue(json, "act_type");
    config.dim = JsonI32(json, "dim");
    config.hidden_dim = JsonI32(json, "hidden_dim");
    config.head_dim = JsonI32(json, "head_dim");
    config.n_layers = JsonI32(json, "n_layers");
    config.n_heads = JsonI32(json, "n_heads");
    config.n_kv_heads = JsonI32(json, "n_kv_heads");
    config.vocab_size = JsonI32(json, "vocab_size");
    config.max_seq_len = JsonI32(json, "max_seq_len");
    config.bos_token_id = JsonI32(json, "bos_token_id");
    config.eos_token_id = JsonI32(json, "eos_token_id");
    config.rotary_dim = JsonI32(json, "rotary_dim");
    config.rope_theta = JsonFloat(json, "rope_theta");
    config.norm_eps = JsonFloat(json, "norm_eps");
    config.tie_word_embeddings = JsonBool(json, "tie_word_embeddings");
    config.attention_bias = JsonBool(json, "attention_bias");
    config.qk_norm = JsonBool(json, "qk_norm");
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
        info.shape.push_back(CheckedSize(ReadPod<std::size_t>(in), "tensor dimension"));
    }

    info.byte_size = CheckedSize(ReadPod<std::size_t>(in), "tensor byte size");

    auto CheckedTell = [](std::istream& in) {
        const auto pos = in.tellg();
        if (pos < 0) {
            throw std::runtime_error("Failed to query input position");
        }
        return static_cast<std::size_t>(pos);
    };
    
    info.data_offset = CheckedSize(CheckedTell(in), "tensor data offset");
    ValidateTensorByteSize(info);
    return info;
}

auto IsTokenizerTensor(std::string_view name) -> bool {
    return name == "tokenizer.json" || name == "tokenizer.tokens" || name == "tokenizer.offsets";
}

auto ValidateSupportedTensorType(const TensorInfo& info) -> void {
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
    std::size_t element_size
) -> std::vector<std::uint8_t> {
    if (element_size == 0 || info.byte_size % element_size != 0) {
        throw std::runtime_error("Tensor byte size is not aligned: " + info.name);
    }

    if (info.data_offset > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max()) ||
        info.byte_size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
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

}  // namespace

auto LoadTensorBytes(
    const std::string& path,
    const TensorInfo& info,
    TensorDType expected_dtype,
    std::size_t element_size
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

    auto data = std::string(static_cast<std::size_t>(size), '\0');
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

auto Tokenizer::VocabSize() const -> std::size_t {
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

    if (id < 0 || static_cast<std::size_t>(id) >= offsets.size()) {
        throw std::out_of_range("Token id is out of range");
    }
    const auto offset = offsets[id];
    if (offset < 0 || static_cast<std::size_t>(offset) >= tokens.size()) {
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

auto Qwen3::Load(const std::string& path) -> void {
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

    const auto metadata_size = ReadPod<std::uint64_t>(in);
    const auto metadata_json = ReadString(in, metadata_size);
    transformer_.config = std::make_shared<Config>(ParseConfig(metadata_json));
    if (transformer_.config->dtype != "bf16") {
        throw std::runtime_error("Only bf16 Qwen3.bin files are supported");
    }

    inference_initialized_ = false;
    tokenizer_ = Tokenizer{};
    tokenizer_.SetSpecialTokenIds(*transformer_.config);
    model_path_ = path;

    const auto tensor_count = ReadPod<std::uint64_t>(in);
    for (auto i{0uz}; i < tensor_count; ++i) {
        TensorInfo info = ReadTensorInfo(in);
        ValidateSupportedTensorType(info);
        if (!transformer_.tensors.emplace(info.name, info).second) {
            throw std::runtime_error("Duplicate tensor record: " + info.name);
        }
        SkipBytes(in, info.byte_size);
    }

    if (HasTensor(transformer_.tensors, "tokenizer.tokens")) {
        auto token_bytes = LoadTensor<std::uint8_t>("tokenizer.tokens");
        tokenizer_.tokens.assign(
            reinterpret_cast<const char*>(token_bytes.ptr()),
            reinterpret_cast<const char*>(token_bytes.ptr() + token_bytes.data.size())
        );
    }

    if (HasTensor(transformer_.tensors, "tokenizer.offsets")) {
        auto token_offsets = LoadTensor<std::int32_t>("tokenizer.offsets");
        tokenizer_.offsets = std::move(token_offsets.data);
    }

    if (HasTensor(transformer_.tensors, "tokenizer.json")) {
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
    }
}

auto Qwen3::GetConfig() const -> const Config* {
    return transformer_.config.get();
}

auto Qwen3::GetTensorIndex() const -> const std::unordered_map<std::string, TensorInfo>& {
    return transformer_.tensors;
}

auto Qwen3::GetTokenizer() const -> const Tokenizer& {
    return tokenizer_;
}

auto Qwen3::LoadFloatData(const std::string& name) const -> std::vector<std::bfloat16_t> {
    return LoadTensor<std::bfloat16_t>(name).data;
}

auto Qwen3::InitializeInference(int context_length) -> void {
    if (!transformer_.config) {
        throw std::runtime_error("Load a model before initializing inference");
    }
    if (context_length <= 0 || context_length > transformer_.config->max_seq_len) {
        throw std::invalid_argument("Context length is outside the model's supported range");
    }
    if (inference_initialized_ && inference_config_->max_seq_len == context_length) {
        ResetInference();
        return;
    }
    inference_initialized_ = false;

    inference_config_ = std::make_shared<Config>(*transformer_.config);
    inference_config_->max_seq_len = context_length;

    embedding_ = LoadFloatData("model.embed.weight");
    final_norm_ = LoadFloatData("model.norm.weight");
    if (transformer_.config->tie_word_embeddings) {
        output_.clear();
    } else {
        output_ = LoadFloatData("model.output.weight");
    }

    const auto dim = static_cast<std::size_t>(transformer_.config->dim);
    const auto vocab_size = static_cast<std::size_t>(transformer_.config->vocab_size);
    if (embedding_.size() != vocab_size * dim || final_norm_.size() != dim ||
        (!transformer_.config->tie_word_embeddings && output_.size() != vocab_size * dim)) {
        throw std::runtime_error("Invalid embedding, final norm, or output tensor shape");
    }

    transformer_.blocks.reserve(transformer_.config->n_layers);
    for (auto layer{0}; layer < transformer_.config->n_layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        BlockWeights weights;
        weights.attn_norm = LoadFloatData(prefix + ".attn.norm.weight");
        weights.q_norm = LoadFloatData(prefix + ".attn.q_norm.weight");
        weights.k_norm = LoadFloatData(prefix + ".attn.k_norm.weight");
        weights.wq = LoadFloatData(prefix + ".attn.wq.weight");
        weights.wk = LoadFloatData(prefix + ".attn.wk.weight");
        weights.wv = LoadFloatData(prefix + ".attn.wv.weight");
        weights.wo = LoadFloatData(prefix + ".attn.wo.weight");
        weights.mlp_norm = LoadFloatData(prefix + ".mlp.norm.weight");
        weights.w1 = LoadFloatData(prefix + ".mlp.w1.weight");
        weights.w2 = LoadFloatData(prefix + ".mlp.w2.weight");
        weights.w3 = LoadFloatData(prefix + ".mlp.w3.weight");
        transformer_.blocks.emplace_back(inference_config_, std::move(weights));
    }

    hidden_state_.resize(transformer_.config->dim);
    normalized_state_.resize(transformer_.config->dim);
    logits_.resize(transformer_.config->vocab_size);
    inference_initialized_ = true;
    ResetInference();
}

auto Qwen3::ResetInference() -> void {
    for (Block& block : transformer_.blocks) {
        block.ResetCache();
    }
    std::fill(hidden_state_.begin(), hidden_state_.end(), 0.0f);
    std::fill(normalized_state_.begin(), normalized_state_.end(), 0.0f);
    std::fill(logits_.begin(), logits_.end(), 0.0f);
}

auto Qwen3::ForwardToken(std::int32_t token, int pos, State &state) -> const std::vector<float>& {
    if (!inference_initialized_) {
        throw std::runtime_error("Initialize inference before forwarding tokens");
    }
    const auto& model_config = *transformer_.config;
    const auto& runtime_config = *inference_config_;
    if (token < 0 || token >= model_config.vocab_size) {
        throw std::out_of_range("Token id is outside the vocabulary");
    }
    if (pos < 0) {
        throw std::out_of_range("Token position must not be negative");
    }

    const auto* embedding_row =
        embedding_.data() + static_cast<std::size_t>(token) * model_config.dim;
    std::copy_n(embedding_row, model_config.dim, hidden_state_.begin());

    constexpr int kAttentionSinks = 4;
    const auto context_length = runtime_config.max_seq_len;
    const auto num_sink = pos >= context_length
        ? std::min(kAttentionSinks, context_length - 1)
        : 0;
    const auto kv_pos = pos < context_length
        ? pos
        : num_sink + (pos - num_sink) % (context_length - num_sink);
    const auto kv_len = std::min(pos + 1, context_length);

    for (Block& block : transformer_.blocks) {
        block.forward(hidden_state_.data(), pos, num_sink, kv_pos, kv_len, state);
    }

    RmsNorm(
        hidden_state_.data(),
        final_norm_.data(),
        normalized_state_.data(),
        model_config.norm_eps,
        model_config.dim
    );
    const auto classifier = model_config.tie_word_embeddings
        ? embedding_.data()
        : output_.data();
    MatMul(
        logits_.data(),
        normalized_state_.data(),
        classifier,
        model_config.vocab_size,
        model_config.dim
    );
    return logits_;
}

auto Qwen3::Generate(
    std::string_view prompt,
    bool apply_chat_template,
    std::size_t max_generated_tokens,
    bool stop_on_eos
) -> GenerationResult {
    if (!inference_initialized_) {
        InitializeInference();
    } else {
        ResetInference();
    }

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
        prompt_tokens.push_back(transformer_.config->bos_token_id);
    }

    GenerationResult result;
    result.stats.prompt_tokens = prompt_tokens.size();

    auto pos = 0;
    const std::vector<float>* logits = nullptr;
    const auto prefill_start = std::chrono::steady_clock::now();
    State state(inference_config_);
    for (const auto token : prompt_tokens) {
        logits = &ForwardToken(token, pos++, state);
    }
    const auto prefill_end = std::chrono::steady_clock::now();
    result.stats.prefill_seconds =
        std::chrono::duration<double>(prefill_end - prefill_start).count();

    const auto decode_start = std::chrono::steady_clock::now();
    while (result.tokens.size() < max_generated_tokens) {
        const auto token = Sampler::Greedy(*logits);
        if (stop_on_eos && token == transformer_.config->eos_token_id) {
            result.stats.stopped_on_eos = true;
            break;
        }
        result.tokens.push_back(token);
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
    if (!config) {
        throw std::invalid_argument("Block config must not be null");
    }
    
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
    const auto dim = static_cast<std::size_t>(config->dim);
    const auto hidden_dim = static_cast<std::size_t>(config->hidden_dim);
    const auto head_dim = static_cast<std::size_t>(config->head_dim);
    const auto q_dim = static_cast<std::size_t>(config->n_heads) * head_dim;
    const auto kv_dim = static_cast<std::size_t>(config->n_kv_heads) * head_dim;

    const auto require_size = [](const std::vector<std::bfloat16_t>& weight, std::size_t expected,
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

    RmsNorm(x, weights.attn_norm.data(), norm_buffer.data(), c.norm_eps, c.dim);
    MatMul(q.data(), norm_buffer.data(), weights.wq.data(), q_dim, c.dim);
    MatMul(k.data(), norm_buffer.data(), weights.wk.data(), kv_dim, c.dim);
    MatMul(v.data(), norm_buffer.data(), weights.wv.data(), kv_dim, c.dim);

    for (auto head = 0; head < c.n_heads; ++head) {
        auto* q_head = q.data() + head * c.head_dim;
        RmsNorm(q_head, weights.q_norm.data(), q_head, c.norm_eps, c.head_dim);
    }
    for (auto head = 0; head < c.n_kv_heads; ++head) {
        auto* k_head = k.data() + head * c.head_dim;
        RmsNorm(k_head, weights.k_norm.data(), k_head, c.norm_eps, c.head_dim);
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
        Attn(
            attn_output.data() + head * c.head_dim,
            attn_scores.data() + static_cast<std::size_t>(head) * c.max_seq_len,
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

    RmsNorm(x, weights.mlp_norm.data(), norm_buffer.data(), c.norm_eps, c.dim);
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
    attn_scores.resize(static_cast<std::size_t>(c->n_heads) * c->max_seq_len);
}

// ----- Layers -----

auto RmsNorm(
    const float *x, 
    const std::bfloat16_t *weights, 
    float *out,
    float eps,
    int n
) -> void {
    
    auto rms = 0.0f;
    for (int i = 0; i < n; i++) {
        rms += x[i] * x[i];
    }
    rms = sqrtf(rms / n + eps);
    
    const auto inverse_rms = 1.0f / rms;

    for (auto i = 0; i < n; i++) {
        out[i] = x[i] * inverse_rms * static_cast<float>(weights[i]);
    }

}

auto Softmax(
    const float *x, 
    float *out, 
    int    n
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

auto Gelu(float x) -> float {
    return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
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
    auto Reduce = [](__m256 value) {
        const __m128 low  = _mm256_extractf128_ps(value, 0);
        const __m128 high = _mm256_extractf128_ps(value, 1);

        __m128 sum = _mm_add_ps(low, high);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);

        return _mm_cvtss_f32(sum);
    };

    auto LoadBF16 = [](const std::bfloat16_t *p) {
        const __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        __m256i wide = _mm256_cvtepu16_epi32(tmp);
        wide = _mm256_slli_epi32(wide, 16);
        return _mm256_castsi256_ps(wide);
    };
    
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < n; i++) {
        const std::bfloat16_t *w_row = w + static_cast<std::size_t>(i) * m;
        
        __m256 sum = _mm256_setzero_ps();
        int j;
        for (j = 0; j + 16 <= m; j += 16) {
            __m256 wv_low = LoadBF16(w_row + j);
            __m256 wv_high = LoadBF16(w_row + j + 8);
            __m256 xv_low = _mm256_loadu_ps(x + j);
            __m256 xv_high = _mm256_loadu_ps(x + j + 8);

            sum = _mm256_fmadd_ps(wv_low, xv_low, sum);
            sum = _mm256_fmadd_ps(wv_high, xv_high, sum);
        }
        out[i] = Reduce(sum);
        for (; j < m; j++) {
            out[i] += static_cast<float>(w_row[j]) * x[j];
        }
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
