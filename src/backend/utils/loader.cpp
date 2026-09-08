#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "config.h"
#include "model.h"


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
    std::unreachable();
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

// Load Model 

Model::Model(const std::string& path) {
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
    inference_config_ = std::make_shared<Config>(metadata_json);
    if (inference_config_->dtype != "bf16") {
        throw std::runtime_error("Only bf16 Qwen3.bin files are supported");
    }

    model_path_ = path;
    model_max_seq_len_ = inference_config_->max_seq_len;

    const auto tensor_count = ReadPod<std::uint64_t>(in);
    for (auto i{0uz}; i < tensor_count; ++i) {
        TensorInfo info = ReadTensorInfo(in);
        ValidateSupportedTensorType(info);
        if (!tensors_.emplace(info.name, info).second) {
            throw std::runtime_error("Duplicate tensor record: " + info.name);
        }
        SkipBytes(in, info.byte_size);
    }
}

// Load Config
Config::Config(const std::string& json) {
    arch                = GetJsonValue<std::string>(json, "arch");
    dtype               = GetJsonValue<std::string>(json, "dtype");
    act_type            = GetJsonValue<std::string>(json, "act_type");
    dim                 = GetJsonValue<std::int32_t>(json, "dim");
    hidden_dim          = GetJsonValue<std::int32_t>(json, "hidden_dim");
    head_dim            = GetJsonValue<std::int32_t>(json, "head_dim");
    n_layers            = GetJsonValue<std::int32_t>(json, "n_layers");
    n_heads             = GetJsonValue<std::int32_t>(json, "n_heads");
    n_kv_heads          = GetJsonValue<std::int32_t>(json, "n_kv_heads");
    vocab_size          = GetJsonValue<std::int32_t>(json, "vocab_size");
    max_seq_len         = GetJsonValue<std::int32_t>(json, "max_seq_len");
    bos_token_id        = GetJsonValue<std::int32_t>(json, "bos_token_id");
    eos_token_id        = GetJsonValue<std::int32_t>(json, "eos_token_id");
    rotary_dim          = GetJsonValue<std::int32_t>(json, "rotary_dim");
    rope_theta          = GetJsonValue<float>(json, "rope_theta");
    norm_eps            = GetJsonValue<float>(json, "norm_eps");
    tie_word_embeddings = GetJsonValue<bool>(json, "tie_word_embeddings");
    attention_bias      = GetJsonValue<bool>(json, "attention_bias");
    qk_norm             = GetJsonValue<bool>(json, "qk_norm");
    ValidateConfig();
}
