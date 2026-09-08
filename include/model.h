#ifndef MODEL_H
#define MODEL_H
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "config.h"
#include "state.h"
#include "utils.h"


class Model {
public:
    explicit Model(const std::string& path);
    virtual ~Model() = default;

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;

    const Config* GetConfig() const noexcept;
    const std::unordered_map<std::string, TensorInfo>& GetTensorIndex() const noexcept;

    void InitializeInference(int context_length = 512);
    void ResetInference();

    template <SupportedTensorElement T>
    Tensor<T> LoadTensor(const std::string& name) const {
        const auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }

        const TensorInfo& info = it->second;
        const std::vector<std::uint8_t> bytes = LoadTensorBytes(
            model_path_, info, ExpectedDType<T>(), sizeof(T));

        std::vector<T> data(bytes.size() / sizeof(T));
        std::memcpy(data.data(), bytes.data(), bytes.size());
        return Tensor<T>{info, std::move(data)};
    }

    virtual void Prefill(const std::span<const std::int32_t> tokens, int pos, State &state) = 0;
    virtual void ForwardToken(std::int32_t token, int pos, State &state) = 0;
    virtual GenerationResult Generate(
        const std::vector<std::int32_t>& prompt_tokens,
        size_t max_generated_tokens = 512,
        bool stop_on_eos = true,
        float temperature = 0.6f,
        std::optional<std::uint64_t> seed = std::nullopt
    ) = 0;

protected:
    template <SupportedTensorElement T>
    auto LoadTensorData(const std::string& name) const -> std::vector<T> {
        return LoadTensor<T>(name).data;
    }

    const Config& GetInferenceConfig() const noexcept {
        return *inference_config_;
    }

    Config& GetInferenceConfig() noexcept {
        return *inference_config_;
    }

private:
    virtual void InitializeBackend() = 0;
    virtual void ResetBackend() = 0;

    std::string model_path_;
    std::unordered_map<std::string, TensorInfo> tensors_;
    std::shared_ptr<Config> inference_config_;
    int model_max_seq_len_ = 0;
};

#endif
