#include "inference.h"

#include <tokenizers_cpp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>


// ----- Helpers -----
namespace {

constexpr char kMagic[8] = {'Q', 'W', 'E', 'N', '3', 'C', 'P', '\0'};
constexpr std::uint32_t kVersion = 1;

template <typename T>
T ReadPod(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Unexpected end of Qwen3.bin");
    }
    return value;
}

std::string ReadString(std::istream& in, std::uint64_t size) {
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

std::uint64_t CheckedTell(std::istream& in) {
    const std::streampos pos = in.tellg();
    if (pos < 0) {
        throw std::runtime_error("Failed to query input position");
    }
    return static_cast<std::uint64_t>(pos);
}

std::int64_t CheckedI64(std::uint64_t value, const std::string& field) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error(field + " does not fit in int64_t");
    }
    return static_cast<std::int64_t>(value);
}

void SkipBytes(std::istream& in, std::int64_t bytes) {
    if (bytes < 0 || bytes > std::numeric_limits<std::streamoff>::max()) {
        throw std::runtime_error("Tensor payload is too large to seek over");
    }
    in.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
    if (!in) {
        throw std::runtime_error("Unexpected end of Qwen3.bin while skipping tensor payload");
    }
}

std::size_t DTypeSize(TensorDType dtype) {
    switch (dtype) {
        case TensorDType::Float32:
            return 4;
        case TensorDType::UInt8:
            return 1;
        case TensorDType::Int32:
            return 4;
    }
    throw std::runtime_error("Unknown tensor dtype");
}

std::int64_t ShapeElementCount(const std::vector<std::int64_t>& shape) {
    std::int64_t count = 1;
    for (const std::int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("Tensor shape contains a negative dimension");
        }
        if (dim != 0 && count > std::numeric_limits<std::int64_t>::max() / dim) {
            throw std::runtime_error("Tensor shape element count overflow");
        }
        count *= dim;
    }
    return count;
}

void ValidateTensorByteSize(const TensorInfo& info) {
    const std::int64_t element_count = ShapeElementCount(info.shape);
    const std::int64_t dtype_size = static_cast<std::int64_t>(DTypeSize(info.dtype));
    if (element_count > std::numeric_limits<std::int64_t>::max() / dtype_size) {
        throw std::runtime_error("Tensor byte size overflow for " + info.name);
    }
    const std::int64_t expected = element_count * dtype_size;
    if (expected != info.byte_size) {
        throw std::runtime_error("Tensor byte size mismatch for " + info.name);
    }
}

std::string JsonValue(const std::string& json, const std::string& key) {
    const std::string quoted_key = "\"" + key + "\":";
    const std::size_t key_pos = json.find(quoted_key);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("Missing metadata key: " + key);
    }
    std::size_t pos = key_pos + quoted_key.size();
    while (pos < json.size() && json[pos] == ' ') {
        ++pos;
    }
    if (pos >= json.size()) {
        throw std::runtime_error("Malformed metadata value for: " + key);
    }

    if (json[pos] == '"') {
        const std::size_t end = json.find('"', pos + 1);
        if (end == std::string::npos) {
            throw std::runtime_error("Unterminated string metadata value for: " + key);
        }
        return json.substr(pos + 1, end - pos - 1);
    }

    const std::size_t end = json.find_first_of(",}", pos);
    if (end == std::string::npos) {
        throw std::runtime_error("Unterminated metadata value for: " + key);
    }
    return json.substr(pos, end - pos);
}

std::int32_t JsonI32(const std::string& json, const std::string& key) {
    const long value = std::stol(JsonValue(json, key));
    if (value < 0 || value > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("Metadata integer overflow for: " + key);
    }
    return static_cast<std::int32_t>(value);
}

float JsonFloat(const std::string& json, const std::string& key) {
    return std::stof(JsonValue(json, key));
}

bool JsonBool(const std::string& json, const std::string& key) {
    const std::string value = JsonValue(json, key);
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    throw std::runtime_error("Metadata value is not bool for: " + key);
}

Config ParseConfig(const std::string& json) {
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

TensorInfo ReadTensorInfo(std::istream& in) {
    TensorInfo info;
    const std::uint32_t name_size = ReadPod<std::uint32_t>(in);
    info.name = ReadString(in, name_size);
    info.dtype = static_cast<TensorDType>(ReadPod<std::uint32_t>(in));

    const std::uint32_t ndim = ReadPod<std::uint32_t>(in);
    info.shape.reserve(ndim);
    for (std::uint32_t i = 0; i < ndim; ++i) {
        info.shape.push_back(CheckedI64(ReadPod<std::uint64_t>(in), "tensor dimension"));
    }

    info.byte_size = CheckedI64(ReadPod<std::uint64_t>(in), "tensor byte size");
    info.data_offset = CheckedI64(CheckedTell(in), "tensor data offset");
    ValidateTensorByteSize(info);
    return info;
}

bool IsTokenizerTensor(const std::string& name) {
    return name == "tokenizer.json" || name == "tokenizer.tokens" || name == "tokenizer.offsets";
}

void ValidateSupportedTensorType(const TensorInfo& info) {
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

    if (info.dtype != TensorDType::Float32) {
        throw std::runtime_error("Only fp32 model tensors are supported: " + info.name);
    }
}

std::vector<std::uint8_t> ReadTensorBytes(
    const std::string& path,
    const TensorInfo& info,
    std::int64_t element_size
) {
    if (info.byte_size < 0 || static_cast<std::uint64_t>(info.byte_size) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Tensor is too large to allocate: " + info.name);
    }
    if (element_size <= 0 || info.byte_size % element_size != 0) {
        throw std::runtime_error("Tensor byte size is not aligned: " + info.name);
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open model file: " + path);
    }
    in.seekg(static_cast<std::streamoff>(info.data_offset), std::ios::beg);
    if (!in) {
        throw std::runtime_error("Failed to seek to tensor: " + info.name);
    }

    std::vector<std::uint8_t> data(static_cast<std::size_t>(info.byte_size));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(info.byte_size));
    if (!in) {
        throw std::runtime_error("Failed to read tensor payload: " + info.name);
    }
    return data;
}

}  // namespace

std::vector<std::uint8_t> LoadTensorBytes(
    const std::string& path,
    const TensorInfo& info,
    TensorDType expected_dtype,
    std::int64_t element_size
) {
    if (info.dtype != expected_dtype) {
        throw std::runtime_error("Tensor dtype mismatch: " + info.name);
    }
    return ReadTensorBytes(path, info, element_size);
}

bool HasTensor(const std::unordered_map<std::string, TensorInfo>& tensors, const std::string& name) {
    return tensors.find(name) != tensors.end();
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open tokenizer file: " + path.string());
    }

    in.seekg(0, std::ios::end);
    const std::streampos size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("Failed to query tokenizer file size: " + path.string());
    }
    in.seekg(0, std::ios::beg);

    std::string data(static_cast<std::size_t>(size), '\0');
    in.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!in) {
        throw std::runtime_error("Failed to read tokenizer file: " + path.string());
    }
    return data;
}

std::filesystem::path FindTokenizerJson(const std::string& model_path) {
    const std::filesystem::path path(model_path);
    const std::filesystem::path parent = path.parent_path().empty()
        ? std::filesystem::current_path()
        : path.parent_path();

    const std::filesystem::path sibling = parent / "tokenizer.json";
    if (std::filesystem::exists(sibling)) {
        return sibling;
    }

    const std::filesystem::path project_default = parent / "Qwen3-0.6B" / "tokenizer.json";
    if (std::filesystem::exists(project_default)) {
        return project_default;
    }

    return {};
}

// ----- Tokenizer -----

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

void Tokenizer::LoadFromJsonBlob(const std::string& json_blob) {
    backend_ = tokenizers::Tokenizer::FromBlobJSON(json_blob);
    if (!backend_) {
        throw std::runtime_error("Failed to initialize tokenizer backend");
    }
}

void Tokenizer::LoadFromJsonFile(const std::string& path) {
    LoadFromJsonBlob(ReadFile(path));
}

void Tokenizer::SetSpecialTokenIds(const Config& config) {
    bos_token_id_ = config.bos_token_id;
    eos_token_id_ = config.eos_token_id;
}

bool Tokenizer::HasBackend() const {
    return backend_ != nullptr;
}

std::size_t Tokenizer::VocabSize() const {
    if (!offsets.empty()) {
        return offsets.size();
    }
    if (backend_) {
        return backend_->GetVocabSize();
    }
    return 0;
}

std::int32_t Tokenizer::BosTokenId() const {
    return bos_token_id_;
}

std::int32_t Tokenizer::EosTokenId() const {
    return eos_token_id_;
}

std::string Tokenizer::Token(std::int32_t id) const {
    if (backend_) {
        const std::string token = backend_->IdToToken(id);
        if (!token.empty()) {
            return token;
        }
    }

    if (id < 0 || static_cast<std::size_t>(id) >= offsets.size()) {
        throw std::out_of_range("Token id is out of range");
    }
    const std::int32_t offset = offsets[static_cast<std::size_t>(id)];
    if (offset < 0 || static_cast<std::size_t>(offset) >= tokens.size()) {
        throw std::runtime_error("Tokenizer offset is out of range");
    }
    return std::string(tokens.data() + offset);
}

std::vector<std::int32_t> Tokenizer::Encode(std::string_view prompt) const {
    if (!backend_) {
        throw std::runtime_error("Tokenizer backend is not initialized");
    }
    return backend_->Encode(std::string(prompt));
}

std::string Tokenizer::Decode(std::span<const std::int32_t> token_ids) const {
    if (!backend_) {
        throw std::runtime_error("Tokenizer backend is not initialized");
    }
    return backend_->Decode(std::vector<std::int32_t>(token_ids.begin(), token_ids.end()));
}

// ----- Qwen3 -----

void Qwen3::Load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open model file: " + path);
    }

    char magic[8] = {};
    in.read(magic, sizeof(magic));
    if (!in || !std::equal(std::begin(magic), std::end(magic), std::begin(kMagic))) {
        throw std::runtime_error("Invalid Qwen3 model file magic");
    }

    const std::uint32_t version = ReadPod<std::uint32_t>(in);
    if (version != kVersion) {
        throw std::runtime_error("Unsupported Qwen3 model file version");
    }

    const std::uint64_t metadata_size = ReadPod<std::uint64_t>(in);
    const std::string metadata_json = ReadString(in, metadata_size);
    transformer_.config = std::make_shared<Config>(ParseConfig(metadata_json));
    if (transformer_.config->dtype != "fp32") {
        throw std::runtime_error("Only fp32 Qwen3.bin files are supported");
    }

    transformer_.tensors.clear();
    transformer_.blocks.clear();
    inference_config_.reset();
    embedding_.clear();
    final_norm_.clear();
    output_.clear();
    hidden_state_.clear();
    normalized_state_.clear();
    logits_.clear();
    inference_initialized_ = false;
    tokenizer_ = Tokenizer{};
    tokenizer_.SetSpecialTokenIds(*transformer_.config);
    model_path_ = path;

    const std::uint64_t tensor_count = ReadPod<std::uint64_t>(in);
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo info = ReadTensorInfo(in);
        ValidateSupportedTensorType(info);
        if (!transformer_.tensors.emplace(info.name, info).second) {
            throw std::runtime_error("Duplicate tensor record: " + info.name);
        }
        SkipBytes(in, info.byte_size);
    }

    if (HasTensor(transformer_.tensors, "tokenizer.tokens")) {
        Tensor<std::uint8_t> token_bytes = LoadTensor<std::uint8_t>("tokenizer.tokens");
        tokenizer_.tokens.assign(
            reinterpret_cast<const char*>(token_bytes.ptr()),
            reinterpret_cast<const char*>(token_bytes.ptr() + token_bytes.data.size())
        );
    }

    if (HasTensor(transformer_.tensors, "tokenizer.offsets")) {
        Tensor<std::int32_t> token_offsets = LoadTensor<std::int32_t>("tokenizer.offsets");
        tokenizer_.offsets = std::move(token_offsets.data);
    }

    if (HasTensor(transformer_.tensors, "tokenizer.json")) {
        Tensor<std::uint8_t> tokenizer_json = LoadTensor<std::uint8_t>("tokenizer.json");
        tokenizer_.LoadFromJsonBlob(std::string(
            reinterpret_cast<const char*>(tokenizer_json.ptr()),
            tokenizer_json.data.size()
        ));
    } else {
        const std::filesystem::path tokenizer_path = FindTokenizerJson(path);
        if (!tokenizer_path.empty()) {
            tokenizer_.LoadFromJsonFile(tokenizer_path.string());
        }
    }
}

const Config* Qwen3::GetConfig() const {
    return transformer_.config.get();
}

const std::unordered_map<std::string, TensorInfo>& Qwen3::GetTensorIndex() const {
    return transformer_.tensors;
}

const Tokenizer& Qwen3::GetTokenizer() const {
    return tokenizer_;
}

std::vector<float> Qwen3::LoadFloatData(const std::string& name) const {
    return LoadTensor<float>(name).data;
}

void Qwen3::InitializeInference(int context_length) {
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

    const std::size_t dim = static_cast<std::size_t>(transformer_.config->dim);
    const std::size_t vocab_size = static_cast<std::size_t>(transformer_.config->vocab_size);
    if (embedding_.size() != vocab_size * dim || final_norm_.size() != dim ||
        (!transformer_.config->tie_word_embeddings && output_.size() != vocab_size * dim)) {
        throw std::runtime_error("Invalid embedding, final norm, or output tensor shape");
    }

    transformer_.blocks.clear();
    transformer_.blocks.reserve(transformer_.config->n_layers);
    for (int layer = 0; layer < transformer_.config->n_layers; ++layer) {
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

void Qwen3::ResetInference() {
    for (Block& block : transformer_.blocks) {
        block.ResetCache();
    }
    std::fill(hidden_state_.begin(), hidden_state_.end(), 0.0f);
    std::fill(normalized_state_.begin(), normalized_state_.end(), 0.0f);
    std::fill(logits_.begin(), logits_.end(), 0.0f);
}

const std::vector<float>& Qwen3::ForwardToken(std::int32_t token, int pos) {
    if (!inference_initialized_) {
        throw std::runtime_error("Initialize inference before forwarding tokens");
    }
    const Config& model_config = *transformer_.config;
    const Config& runtime_config = *inference_config_;
    if (token < 0 || token >= model_config.vocab_size) {
        throw std::out_of_range("Token id is outside the vocabulary");
    }
    if (pos < 0) {
        throw std::out_of_range("Token position must not be negative");
    }

    const float* embedding_row =
        embedding_.data() + static_cast<std::size_t>(token) * model_config.dim;
    std::copy_n(embedding_row, model_config.dim, hidden_state_.begin());

    constexpr int kAttentionSinks = 4;
    const int context_length = runtime_config.max_seq_len;
    const int num_sink = pos >= context_length
        ? std::min(kAttentionSinks, context_length - 1)
        : 0;
    const int kv_pos = pos < context_length
        ? pos
        : num_sink + (pos - num_sink) % (context_length - num_sink);
    const int kv_len = std::min(pos + 1, context_length);

    for (Block& block : transformer_.blocks) {
        block.forward(hidden_state_.data(), pos, num_sink, kv_pos, kv_len);
    }

    RmsNorm(
        hidden_state_.data(),
        final_norm_.data(),
        normalized_state_.data(),
        model_config.norm_eps,
        model_config.dim
    );
    const float* classifier = model_config.tie_word_embeddings
        ? embedding_.data()
        : output_.data();
    MatMul(
        logits_.data(),
        classifier,
        normalized_state_.data(),
        model_config.vocab_size,
        model_config.dim
    );
    return logits_;
}

GenerationResult Qwen3::Generate(
    std::string_view prompt,
    bool apply_chat_template
) {
    if (!inference_initialized_) {
        InitializeInference();
    } else {
        ResetInference();
    }

    std::string model_input;
    if (apply_chat_template) {
        model_input = "<|im_start|>user\n";
        model_input.append(prompt);
        model_input += "<|im_end|>\n<|im_start|>assistant\n";
    } else {
        model_input.assign(prompt);
    }

    std::vector<std::int32_t> prompt_tokens = tokenizer_.Encode(model_input);
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(transformer_.config->bos_token_id);
    }

    GenerationResult result;
    result.stats.prompt_tokens = prompt_tokens.size();

    int pos = 0;
    const std::vector<float>* logits = nullptr;
    const auto prefill_start = std::chrono::steady_clock::now();
    for (const std::int32_t token : prompt_tokens) {
        logits = &ForwardToken(token, pos++);
    }
    const auto prefill_end = std::chrono::steady_clock::now();
    result.stats.prefill_seconds =
        std::chrono::duration<double>(prefill_end - prefill_start).count();

    const auto decode_start = std::chrono::steady_clock::now();
    while (true) {
        const std::int32_t token = Sampler::Greedy(*logits);
        if (token == transformer_.config->eos_token_id) {
            result.stats.stopped_on_eos = true;
            break;
        }
        result.tokens.push_back(token);
        logits = &ForwardToken(token, pos++);
    }
    const auto decode_end = std::chrono::steady_clock::now();
    result.stats.generated_tokens = result.tokens.size();
    result.stats.decode_seconds =
        std::chrono::duration<double>(decode_end - decode_start).count();
    result.text = tokenizer_.Decode(result.tokens);
    return result;
}

std::int32_t Sampler::Greedy(std::span<const float> logits) {
    if (logits.empty()) {
        throw std::invalid_argument("Cannot sample empty logits");
    }
    return static_cast<std::int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
}

double GenerationStats::PrefillTokensPerSecond() const {
    return prefill_seconds > 0.0 ? prompt_tokens / prefill_seconds : 0.0;
}

double GenerationStats::DecodeTokensPerSecond() const {
    return decode_seconds > 0.0 ? generated_tokens / decode_seconds : 0.0;
}

// ----- TransformerBlock -----
Block::Block(const std::shared_ptr<Config>& config) : config(config) {
    if (!config) {
        throw std::invalid_argument("Block config must not be null");
    }
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

    const int q_dim = config->n_heads * config->head_dim;
    const int kv_dim = config->n_kv_heads * config->head_dim;
    cache = KVCache(config->max_seq_len, config->n_kv_heads, config->head_dim);
    norm_buffer.resize(config->dim);
    q.resize(q_dim);
    k.resize(kv_dim);
    v.resize(kv_dim);
    attn_output.resize(q_dim);
    projected.resize(config->dim);
    attn_scores.resize(static_cast<std::size_t>(config->n_heads) * config->max_seq_len);
}

Block::Block(const std::shared_ptr<Config>& config, BlockWeights weights)
    : Block(config) {
    SetWeights(std::move(weights));
}

void Block::SetWeights(BlockWeights new_weights) {
    weights = std::move(new_weights);
    ValidateWeights();
}

void Block::ResetCache() {
    std::fill(cache.k_.begin(), cache.k_.end(), 0.0f);
    std::fill(cache.v_.begin(), cache.v_.end(), 0.0f);
}

void Block::ValidateWeights() const {
    const std::size_t dim = static_cast<std::size_t>(config->dim);
    const std::size_t hidden_dim = static_cast<std::size_t>(config->hidden_dim);
    const std::size_t head_dim = static_cast<std::size_t>(config->head_dim);
    const std::size_t q_dim = static_cast<std::size_t>(config->n_heads) * head_dim;
    const std::size_t kv_dim = static_cast<std::size_t>(config->n_kv_heads) * head_dim;

    const auto require_size = [](const std::vector<float>& weight, std::size_t expected,
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

void Block::forward(
    float* x,
    int pos,
    int num_sink,
    int kv_pos,
    int kv_len
) {
    if (!x) {
        throw std::invalid_argument("Block input must not be null");
    }
    ValidateWeights();

    const Config& c = *config;
    if (pos < 0 || kv_len <= 0 || kv_len > c.max_seq_len || kv_pos < 0 ||
        kv_pos >= kv_len || num_sink < 0 || num_sink > kv_len ||
        num_sink >= c.max_seq_len || kv_pos < num_sink) {
        throw std::out_of_range("Invalid position or KV-cache range");
    }

    const int q_dim = c.n_heads * c.head_dim;
    const int kv_dim = c.n_kv_heads * c.head_dim;

    RmsNorm(x, weights.attn_norm.data(), norm_buffer.data(), c.norm_eps, c.dim);
    MatMul(q.data(), weights.wq.data(), norm_buffer.data(), q_dim, c.dim);
    MatMul(k.data(), weights.wk.data(), norm_buffer.data(), kv_dim, c.dim);
    MatMul(v.data(), weights.wv.data(), norm_buffer.data(), kv_dim, c.dim);

    for (int head = 0; head < c.n_heads; ++head) {
        float* q_head = q.data() + head * c.head_dim;
        RmsNorm(q_head, weights.q_norm.data(), q_head, c.norm_eps, c.head_dim);
    }
    for (int head = 0; head < c.n_kv_heads; ++head) {
        float* k_head = k.data() + head * c.head_dim;
        RmsNorm(k_head, weights.k_norm.data(), k_head, c.norm_eps, c.head_dim);
    }

    ApplyRotaryEmb(q.data(), q_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim);
    ApplyRotaryEmb(k.data(), kv_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim);

    std::copy(k.begin(), k.end(), cache.k_.begin() + static_cast<std::size_t>(kv_pos) * kv_dim);
    std::copy(v.begin(), v.end(), cache.v_.begin() + static_cast<std::size_t>(kv_pos) * kv_dim);

    // Keep sink tokens at a constant relative distance after the ring buffer fills.
    for (int sink = 0; sink < num_sink; ++sink) {
        float* sink_key = cache.k_.data() + static_cast<std::size_t>(sink) * kv_dim;
        ApplyRotaryEmb(sink_key, kv_dim, c.head_dim, 1, c.rope_theta, c.rotary_dim);
    }

    const int queries_per_kv_head = c.n_heads / c.n_kv_heads;
    for (int head = 0; head < c.n_heads; ++head) {
        const int kv_head = head / queries_per_kv_head;
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

    MatMul(projected.data(), weights.wo.data(), attn_output.data(), c.dim, q_dim);
    for (int i = 0; i < c.dim; ++i) {
        x[i] += projected[i];
    }

    RmsNorm(x, weights.mlp_norm.data(), norm_buffer.data(), c.norm_eps, c.dim);
    FeedForwardNetwork(
        projected.data(),
        norm_buffer.data(),
        weights.w1.data(),
        weights.w2.data(),
        weights.w3.data(),
        c.hidden_dim,
        c.dim
    );
    for (int i = 0; i < c.dim; ++i) {
        x[i] += projected[i];
    }
}

// ----- Layers -----

void RmsNorm(
    const float *x, 
    const float *weights, 
    float *out,
    float  eps,
    int    n
) {
    
    float rms = 0.0f;
    for (int i = 0; i < n; i++) {
        rms += x[i] * x[i];
    }
    rms = sqrtf(rms / n + eps);
    
    const float inverse_rms = 1.0 / rms;

    for (int i = 0; i < n; i++) {
        out[i] = x[i] * inverse_rms * weights[i];
    }

}

void Softmax(
    const float *x, 
    float *out, 
    int    n
) {

    float mx_score = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; i++) {
        mx_score = std::max(mx_score, x[i]);
    }
    float score = 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = expf(x[i] - mx_score);
        score += out[i];
    }

    for (int i = 0; i < n; i++) {
        out[i] /= score;
    }

}

float Gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

float Silu(float x) {
    return x / (1.0f + expf(-x));
}

void MatMul(
    float *out,
    const float* x, 
    const float* y,
    int n,
    int m
) {
    // (n, m) x (m, ) = (n, )

    for (int i = 0; i < n; i++) {
        float val = 0.0f;
        for (int j = 0; j < m; j++) {
            val += x[i * m + j] * y[j];
        }
        out[i] = val;
    }
}

void ApplyRotaryEmb(
    float *out,
    int    d,
    int    head_dim,
    int    pos,
    float  theta,
    int    rotary_dim
) {
    const int rotary_half = rotary_dim / 2;
    for (int head_start = 0; head_start < d; head_start += head_dim) {
        for (int i = 0; i < rotary_half; ++i) {
            const float freq = 1.0f / powf(theta, 2.0f * i / rotary_dim);
            const float angle = pos * freq;
            const float cosine = cosf(angle);
            const float sine = sinf(angle);
            const int first_index = head_start + i;
            const int second_index = first_index + rotary_half;
            const float first = out[first_index];
            const float second = out[second_index];
            out[first_index] = first * cosine - second * sine;
            out[second_index] = second * cosine + first * sine;
        }
    }
}

void FeedForwardNetwork(
    float *out,
    const float *x,
    const float *w1,
    const float *w2,
    const float *w3,
    int hidden_dim,
    int dim
) {
    float *lin1 = new float[hidden_dim];
    float *lin2 = new float[hidden_dim];
    MatMul(lin1, w1, x, hidden_dim, dim);
    MatMul(lin2, w3, x, hidden_dim, dim);
    
    // this is like siluAndMul (?) 
    for (int i = 0; i < hidden_dim; i++) {
        lin1[i] = Silu(lin1[i]) * lin2[i];
    }

    MatMul(out, w2, lin1, dim, hidden_dim);
    delete []lin1;
    delete []lin2;
}

void Attn(
    float *out, // (dim, )
    float *atth, // (kv_len, ) - to hold attn scores
    const float *q, // (head_dim, )
    const float *k, // (kv_len, n_kv_heads, head_dim)
    const float *v, // (kv_len, n_kv_heads, head_dim)
    int head_dim,
    int n_kv_heads,
    int kv_len
) {
    const auto stride = n_kv_heads * head_dim;
    const auto sqrt_head_dim = sqrtf(head_dim);
    for (int i = 0; i < kv_len; i++) {
        auto score = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            score += q[j] * k[i * stride + j];
        }
        score /= sqrt_head_dim;
        atth[i] = score;
    }

    Softmax(atth, atth, kv_len);

    for (int i = 0; i < head_dim; i++) {
        auto res = 0.0f;
        for (int j = 0; j < kv_len; j++) {
            res += atth[j] * v[j * stride + i];
        }
        out[i] = res;
    }
}
