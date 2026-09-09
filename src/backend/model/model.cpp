#include "model.h"
#include "sampler.h"
#include <algorithm>
#include <chrono>

auto Model::GetConfig() const noexcept -> const Config* {
    return inference_config_.get();
}

auto Model::GetTensorIndex() const noexcept
    -> const std::unordered_map<std::string, TensorInfo>& {
    return tensors_;
}

auto Model::InitializeInference(int context_length) -> void {
    auto& config = GetInferenceConfig();
    config.max_seq_len = context_length;
    using bf16 = std::bfloat16_t;

    embedding_  = LoadTensorData<bf16>("model.embed.weight");
    final_norm_ = LoadTensorData<bf16>("model.norm.weight");
    if (config.tie_word_embeddings) {
        output_.clear();
    } else {
        output_ = LoadTensorData<bf16>("model.output.weight");
    }

    const auto dim = static_cast<size_t>(config.dim);
    const auto vocab_size = static_cast<size_t>(config.vocab_size);
    if (embedding_.size() != vocab_size * dim || final_norm_.size() != dim ||
        (!config.tie_word_embeddings && output_.size() != vocab_size * dim)) {
        throw std::runtime_error("Invalid embedding, final norm, or output tensor shape");
    }

    blocks_.clear();
    blocks_.reserve(config.n_layers);
    for (auto layer{0}; layer < config.n_layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        BlockWeights weights;
        weights.attn_norm = LoadTensorData<bf16>(prefix + ".attn.norm.weight");
        weights.q_norm    = LoadTensorData<bf16>(prefix + ".attn.q_norm.weight");
        weights.k_norm    = LoadTensorData<bf16>(prefix + ".attn.k_norm.weight");
        weights.wq        = LoadTensorData<bf16>(prefix + ".attn.wq.weight");
        weights.wk        = LoadTensorData<bf16>(prefix + ".attn.wk.weight");
        weights.wv        = LoadTensorData<bf16>(prefix + ".attn.wv.weight");
        weights.wo        = LoadTensorData<bf16>(prefix + ".attn.wo.weight");
        weights.mlp_norm  = LoadTensorData<bf16>(prefix + ".mlp.norm.weight");
        weights.w1        = LoadTensorData<bf16>(prefix + ".mlp.w1.weight");
        weights.w2        = LoadTensorData<bf16>(prefix + ".mlp.w2.weight");
        weights.w3        = LoadTensorData<bf16>(prefix + ".mlp.w3.weight");
        blocks_.emplace_back(&GetInferenceConfig(), std::move(weights));
    }

    for (const auto& block : blocks_) {
        block.ValidateWeights();
    }

    const auto prefill_capacity = std::min(
        kPrefillBatchSize,
        static_cast<size_t>(config.max_seq_len)
    );
    hidden_state_.resize(prefill_capacity * dim);
    normalized_state_.resize(prefill_capacity * dim);
    logits_.resize(config.vocab_size);
}

auto Model::ResetInference() -> void {
    for (Block& block : blocks_) {
        block.ResetCache();
    }
    std::fill(hidden_state_.begin(), hidden_state_.end(), 0.0f);
    std::fill(normalized_state_.begin(), normalized_state_.end(), 0.0f);
    std::fill(logits_.begin(), logits_.end(), 0.0f);
}

auto Model::ForwardTokenGPU(std::int32_t token, int pos, State& state) -> void {
    const auto& config = GetInferenceConfig();
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

    for (Block& block : blocks_) {
        block.Forward(hidden_state_.data(), pos, num_sink, kv_pos, kv_len, state);
    }

    rmsnorm_cpu(
        normalized_state_.data(),
        hidden_state_.data(),
        final_norm_.data(),
        config.norm_eps,
        config.dim
    );
    const auto classifier = config.tie_word_embeddings
        ? embedding_.data()
        : output_.data();
    matmul_cpu(
        logits_.data(),
        normalized_state_.data(),
        classifier,
        config.vocab_size,
        config.dim
    );
}

auto Model::ForwardTokenCPU(std::int32_t token, int pos, State& state) -> void {
    const auto& config = GetInferenceConfig();
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

    for (Block& block : blocks_) {
        block.Forward(hidden_state_.data(), pos, num_sink, kv_pos, kv_len, state);
    }

    rmsnorm_cpu(
        normalized_state_.data(),
        hidden_state_.data(),
        final_norm_.data(),
        config.norm_eps,
        config.dim
    );
    const auto classifier = config.tie_word_embeddings
        ? embedding_.data()
        : output_.data();
    matmul_cpu(
        logits_.data(),
        normalized_state_.data(),
        classifier,
        config.vocab_size,
        config.dim
    );
}

auto Model::PrefillGPU(
    std::span<const std::int32_t> tokens,
    int pos,
    State& state
) -> void {
    const auto num_tokens = tokens.size();
    const auto& config = GetInferenceConfig();

    if (tokens.empty()) {
        throw std::invalid_argument("Prefill requires at least one token");
    }
    if (pos < 0) {
        throw std::out_of_range("Token position must not be negative");
    }

    const auto start = static_cast<size_t>(pos);
    const auto context_length = static_cast<size_t>(config.max_seq_len);
    if (start > context_length || num_tokens > context_length - start) {
        throw std::out_of_range("Prefill chunk exceeds maximum context length");
    }
    if (num_tokens > state.batch_capacity ||
        num_tokens > hidden_state_.size() / static_cast<size_t>(config.dim)) {
        throw std::length_error("Prefill chunk exceeds CPU batch capacity");
    }

    for (auto t{0uz}; t < num_tokens; ++t) {
        const auto token = tokens[t];
        if (token < 0 || token >= config.vocab_size) {
            throw std::out_of_range("Token id is outside the vocabulary");
        }
        const auto* embedding_row =
            embedding_.data() + static_cast<size_t>(token) * config.dim;
        auto* destination =
            hidden_state_.data() + static_cast<size_t>(t) * config.dim;
        std::copy_n(embedding_row, config.dim, destination);
    }

    for (Block& block : blocks_) {
        block.ForwardPrefill(hidden_state_.data(), num_tokens, pos, state);
    }

    rmsnorm_cpu(
        normalized_state_.data(),
        hidden_state_.data(),
        final_norm_.data(),
        config.norm_eps,
        config.dim,
        static_cast<int>(num_tokens)
    );
    const auto classifier = config.tie_word_embeddings
        ? embedding_.data()
        : output_.data();
    const auto* last_hidden =
        normalized_state_.data() + static_cast<size_t>(num_tokens - 1) * config.dim;
    matmul_cpu(
        logits_.data(),
        last_hidden,
        classifier,
        config.vocab_size,
        config.dim
    );
}

auto Model::PrefillCPU(
    std::span<const std::int32_t> tokens,
    int pos,
    State& state
) -> void {
    const auto num_tokens = tokens.size();
    const auto& config = GetInferenceConfig();

    if (tokens.empty()) {
        throw std::invalid_argument("Prefill requires at least one token");
    }
    if (pos < 0) {
        throw std::out_of_range("Token position must not be negative");
    }

    const auto start = static_cast<size_t>(pos);
    const auto context_length = static_cast<size_t>(config.max_seq_len);
    if (start > context_length || num_tokens > context_length - start) {
        throw std::out_of_range("Prefill chunk exceeds maximum context length");
    }
    if (num_tokens > state.batch_capacity ||
        num_tokens > hidden_state_.size() / static_cast<size_t>(config.dim)) {
        throw std::length_error("Prefill chunk exceeds CPU batch capacity");
    }

    for (auto t{0uz}; t < num_tokens; ++t) {
        const auto token = tokens[t];
        if (token < 0 || token >= config.vocab_size) {
            throw std::out_of_range("Token id is outside the vocabulary");
        }
        const auto* embedding_row =
            embedding_.data() + static_cast<size_t>(token) * config.dim;
        auto* destination =
            hidden_state_.data() + static_cast<size_t>(t) * config.dim;
        std::copy_n(embedding_row, config.dim, destination);
    }

    for (Block& block : blocks_) {
        block.ForwardPrefill(hidden_state_.data(), num_tokens, pos, state);
    }

    rmsnorm_cpu(
        normalized_state_.data(),
        hidden_state_.data(),
        final_norm_.data(),
        config.norm_eps,
        config.dim,
        static_cast<int>(num_tokens)
    );
    const auto classifier = config.tie_word_embeddings
        ? embedding_.data()
        : output_.data();
    const auto* last_hidden =
        normalized_state_.data() + static_cast<size_t>(num_tokens - 1) * config.dim;
    matmul_cpu(
        logits_.data(),
        last_hidden,
        classifier,
        config.vocab_size,
        config.dim
    );
}

auto Model::Generate(
    const std::vector<std::int32_t>& prompt_tokens,
    size_t max_generated_tokens,
    bool stop_on_eos,
    float temperature,
    std::optional<std::uint64_t> seed,
    const TokenCallback& on_token
) -> GenerationResult {
    GenerationResult result;
    result.stats.prompt_tokens = prompt_tokens.size();
    if (prompt_tokens.empty()) {
        throw std::invalid_argument("Prompt must contain at least one token");
    }
    const auto& config = GetInferenceConfig();
    if (prompt_tokens.size() > static_cast<size_t>(config.max_seq_len)) {
        throw std::out_of_range("Prompt exceeds maximum context length");
    }
    const size_t batch_size = std::min(prompt_tokens.size(), kPrefillBatchSize);

    const auto prefill_start = std::chrono::steady_clock::now();
    State state(&GetInferenceConfig(), device);
    const std::span<const std::int32_t> tokens{prompt_tokens};
    for (auto i{0uz}; i < prompt_tokens.size(); i += batch_size) {
        const auto batch_tokens = tokens.subspan(
            i, std::min(batch_size, prompt_tokens.size() - i));

        // branch predictor should deal with this
        if (device == Device::CPU) {
            PrefillCPU(batch_tokens, static_cast<int>(i), state); 
        } else {
            PrefillGPU(batch_tokens, static_cast<int>(i), state);
        }
    }
    const auto prefill_end = std::chrono::steady_clock::now();
    result.stats.prefill_seconds =
        std::chrono::duration<double>(prefill_end - prefill_start).count();

    const auto decode_start = std::chrono::steady_clock::now();
    Sampler sampler(temperature, seed);
    auto pos = static_cast<int>(prompt_tokens.size());
    while (result.tokens.size() < max_generated_tokens) {
        const auto token = sampler.Sample(logits_);
        if (stop_on_eos && token == config.eos_token_id) {
            result.stats.stopped_on_eos = true;
            break;
        }
        result.tokens.push_back(token);
        if (on_token) {
            on_token(token);
        }
        if (device == Device::CPU) {
            ForwardTokenCPU(token, pos++, state);    
        } else {
            ForwardTokenGPU(token, pos++, state);
        }
        
    }
    const auto decode_end = std::chrono::steady_clock::now();
    result.stats.generated_tokens = result.tokens.size();
    result.stats.reached_token_limit =
        !result.stats.stopped_on_eos &&
        result.tokens.size() == max_generated_tokens;
    result.stats.decode_seconds =
        std::chrono::duration<double>(decode_end - decode_start).count();
    return result;
}
