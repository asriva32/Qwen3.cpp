#include "model.h"

auto Model::GetConfig() const noexcept -> const Config* {
    return inference_config_.get();
}

auto Model::GetTensorIndex() const noexcept
    -> const std::unordered_map<std::string, TensorInfo>& {
    return tensors_;
}

auto Model::InitializeInference(int context_length) -> void {
    if (context_length <= 0 || context_length > model_max_seq_len_) {
        throw std::invalid_argument(
            "Context length is outside the model's supported range");
    }

    inference_config_->max_seq_len = context_length;
    InitializeBackend();
}

auto Model::ResetInference() -> void {
    ResetBackend();
}
