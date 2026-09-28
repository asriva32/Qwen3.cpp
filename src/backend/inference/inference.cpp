#include "layers.h"

auto Block::ForwardPrefillGPU(
    float *x,
    size_t num_tokens,
    int pos,
    State &state
) -> void 
{
    if (!x) {
        throw std::invalid_argument("Block input must not be null");
    }
    if (num_tokens == 0) {
        throw std::invalid_argument("Prefill requires at least one token");
    }
    if (pos < 0) {
        throw std::out_of_range("Token position must not be negative");
    }
    const auto start = static_cast<size_t>(pos);
    const auto context_length = static_cast<size_t>(config->max_seq_len);
    if (start > context_length || num_tokens > context_length - start) {
        throw std::out_of_range("Prefill chunk exceeds maximum context length");
    }
    if (num_tokens > state.batch_capacity) {
        throw std::length_error("Prefill chunk exceeds CPU batch capacity");
    }
    const auto batch_size = static_cast<int>(num_tokens);
    // x is [num_tokens, dim]
    // norm_buffer is [num_tokens, dim]
    auto& q = state.q;
    auto& k = state.k;
    auto& v = state.v;
    auto& norm_buffer = state.norm_buffer;
    auto& attn_scores = state.attn_scores;
    auto& attn_output = state.attn_output;
    auto& projected = state.projected;

    const auto q_dim = config->n_heads * config->head_dim;
    const auto kv_dim = config->n_kv_heads * config->head_dim;

    rmsnorm_gpu(
        norm_buffer,
        x,
        attn_norm_.data,
        config->norm_eps,
        config->dim,
        batch_size
    );

    // Batched Q/K/V projections over row-major token inputs.
    matmul_gpu(
        q,
        norm_buffer,
        wq_.data,
        q_dim,
        config->dim,
        batch_size
    );

    matmul_gpu(
        k,
        norm_buffer,
        wk_.data,
        kv_dim,
        config->dim,
        batch_size
    );

    matmul_gpu(
        v,
        norm_buffer,
        wv_.data,
        kv_dim,
        config->dim,
        batch_size
    );

    // Normalize Q/K, apply RoPE, and update the contiguous KV-cache chunk.
    qk_norm_rope_and_update_cache(
        q,
        k,
        v,
        k_,
        v_,
        q_norm_.data,
        k_norm_.data,
        config->n_heads,
        config->n_kv_heads,
        config->head_dim,
        pos,
        static_cast<int>(start),
        config->norm_eps,
        config->rope_theta,
        config->rotary_dim,
        batch_size
    );

    // Each query attends only through its own position, preserving causality even
    // though the complete chunk has already been written to the KV cache.

    attn_gpu(
        attn_output,
        attn_scores,
        q,
        k_,
        v_,
        config->head_dim,
        config->n_heads,
        config->n_kv_heads,
        pos + 1,
        config->max_seq_len,
        batch_size
    );

    // output projection

    matmul_gpu(
        projected,
        attn_output,
        wo_.data,
        config->dim,
        q_dim,
        batch_size
    );

    // Residual
    add_gpu(x, projected, num_tokens * static_cast<size_t>(config->dim));

    // MLP

    rmsnorm_gpu(
        norm_buffer,
        x,
        mlp_norm_.data,
        config->norm_eps,
        config->dim,
        batch_size
    );


    // batched ffn
    ffn_gpu(
        projected,
        state.lin1,
        state.lin2,
        norm_buffer,
        w1_.data,
        w2_.data,
        w3_.data,
        config->hidden_dim,
        config->dim,
        batch_size
    );

    // Final residual
    add_gpu(x, projected, num_tokens * static_cast<size_t>(config->dim));
}

// ForwardPrefill
auto Block::ForwardPrefillCPU(
    float *x,
    size_t num_tokens,
    int pos,
    State &state
) -> void 
{
    if (!x) {
        throw std::invalid_argument("Block input must not be null");
    }
    if (num_tokens == 0) {
        throw std::invalid_argument("Prefill requires at least one token");
    }
   
    const auto start = static_cast<size_t>(pos);
    const auto context_length = static_cast<size_t>(config->max_seq_len);
    if (start > context_length || num_tokens > context_length - start) {
        throw std::out_of_range("Prefill chunk exceeds maximum context length");
    }
    if (num_tokens > state.batch_capacity) {
        throw std::length_error("Prefill chunk exceeds CPU batch capacity");
    }
    const auto batch_size = static_cast<int>(num_tokens);
    // x is [num_tokens, dim]
    // norm_buffer is [num_tokens, dim]
    auto& q = state.q;
    auto& k = state.k;
    auto& v = state.v;
    auto& norm_buffer = state.norm_buffer;
    auto& attn_scores = state.attn_scores;
    auto& attn_output = state.attn_output;
    auto& projected = state.projected;

    const auto q_dim = config->n_heads * config->head_dim;
    const auto kv_dim = config->n_kv_heads * config->head_dim;

    rmsnorm_cpu(
        norm_buffer,
        x,
        attn_norm_.data,
        config->norm_eps,
        config->dim,
        batch_size
    );

    // Batched Q/K/V projections over row-major token inputs.
    matmul_cpu(
        q,
        norm_buffer,
        wq_.data,
        q_dim,
        config->dim,
        batch_size
    );
    matmul_cpu(
        k,
        norm_buffer,
        wk_.data,
        kv_dim,
        config->dim,
        batch_size
    );

    matmul_cpu(
        v,
        norm_buffer,
        wv_.data,
        kv_dim,
        config->dim,
        batch_size
    );

    // Q/K norm
    #pragma omp parallel for collapse(2)
    for (size_t t = 0; t < num_tokens; ++t) {
        for (int head = 0; head < config->n_heads; ++head) {
            auto* q_head = q + t * q_dim + head * config->head_dim;
            rmsnorm_cpu(q_head, q_head, q_norm_.data, config->norm_eps,
                    config->head_dim);
        }
    }

    #pragma omp parallel for collapse(2)
    for (size_t t = 0; t < num_tokens; ++t) {
        for (int head = 0; head < config->n_kv_heads; ++head) {
            auto* k_head = k + t * kv_dim + head * config->head_dim;
            rmsnorm_cpu(k_head, k_head, k_norm_.data, config->norm_eps,
                    config->head_dim);
        }
    }

    // Rope

    #pragma omp parallel for
    for (size_t t = 0; t < num_tokens; ++t) {
        const auto token_pos = pos + static_cast<int>(t);

        rope_cpu(
            q + t * q_dim,
            q_dim,
            config->head_dim,
            token_pos,
            config->rope_theta,
            config->rotary_dim
        );

        rope_cpu(
            k + t * kv_dim,
            kv_dim,
            config->head_dim,
            token_pos,
            config->rope_theta,
            config->rotary_dim
        );
    }

    // Prefill is bounded by the context length, so cache positions are contiguous.
    #pragma omp parallel for
    for (size_t t = 0; t < num_tokens; ++t) {
        const auto kv_pos = start + t;
        const float* k_token = k + t * kv_dim;
        const float* v_token = v + t * kv_dim;
        auto* cache_k = k_ + kv_pos * kv_dim;
        auto* cache_v = v_ + kv_pos * kv_dim;

        for (int i = 0; i < kv_dim; ++i) {
            cache_k[i] = static_cast<std::bfloat16_t>(k_token[i]);
            cache_v[i] = static_cast<std::bfloat16_t>(v_token[i]);
        }
    }

    // Each query attends only through its own position, preserving causality even
    // though the complete chunk has already been written to the KV cache.

    const int queries_per_kv_head = config->n_heads / config->n_kv_heads;

    #pragma omp parallel for collapse(2)
    for (size_t t = 0; t < num_tokens; ++t) {
        for (int head = 0; head < config->n_heads; ++head) {
            const int kv_head = head / queries_per_kv_head;

            const auto kv_len = pos + static_cast<int>(t) + 1;
            attn_cpu(
                attn_output + t * q_dim + head * config->head_dim,
                attn_scores + (t * config->n_heads + head) * config->max_seq_len,
                q + t * q_dim + head * config->head_dim,
                k_ + kv_head * config->head_dim,
                v_ + kv_head * config->head_dim,
                config->head_dim,
                config->n_kv_heads,
                kv_len
            );
        }
    }

    // output projection

    matmul_cpu(
        projected,
        attn_output,
        wo_.data,
        config->dim,
        q_dim,
        batch_size
    );

    // Residual
    #pragma omp parallel for collapse(2)
    for (size_t t = 0; t < num_tokens; ++t) {
        for (int i = 0; i < config->dim; ++i) {
            x[t * config->dim + i] += projected[t * config->dim + i];
        }
    }

    // MLP

    rmsnorm_cpu(
        norm_buffer,
        x,
        mlp_norm_.data,
        config->norm_eps,
        config->dim,
        batch_size
    );


    // batched ffn
    ffn_cpu(
        projected,
        state.lin1,
        state.lin2,
        norm_buffer,
        w1_.data,
        w2_.data,
        w3_.data,
        config->hidden_dim,
        config->dim,
        batch_size
    );

    // Final residual
    #pragma omp parallel for collapse(2)
    for (size_t t = 0; t < num_tokens; ++t) {
        for (int i = 0; i < config->dim; ++i) {
            x[t * config->dim + i] += projected[t * config->dim + i];
        }
    }
}

auto Block::ForwardGPU(
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


    const auto q_dim = config->n_heads * config->head_dim;
    const auto kv_dim = config->n_kv_heads * config->head_dim;

    rmsnorm_gpu(norm_buffer, x, attn_norm_.data , config->norm_eps, config->dim);
    matmul_gpu(q, norm_buffer, wq_.data, q_dim, config->dim);
    matmul_gpu(k, norm_buffer, wk_.data, kv_dim, config->dim);
    matmul_gpu(v, norm_buffer, wv_.data, kv_dim, config->dim);

    qk_norm_rope_and_update_cache(
        q,
        k,
        v,
        k_,
        v_,
        q_norm_.data,
        k_norm_.data,
        config->n_heads,
        config->n_kv_heads,
        config->head_dim,
        pos,
        kv_pos,
        config->norm_eps,
        config->rope_theta,
        config->rotary_dim
    );

    // Keep sink tokens at a constant relative distance after the ring buffer fills.
    rotate_sink_tokens(
        k_,
        static_cast<size_t>(num_sink),
        static_cast<size_t>(kv_dim),
        config->head_dim,
        config->rope_theta,
        config->rotary_dim
    );

    attn_gpu(
        attn_output,
        attn_scores,
        q,
        k_,
        v_,
        config->head_dim,
        config->n_heads,
        config->n_kv_heads,
        kv_len,
        config->max_seq_len
    );

    matmul_gpu(projected, attn_output, wo_.data, config->dim, q_dim);
    add_gpu(x, projected, config->dim);

    rmsnorm_gpu(norm_buffer, x, mlp_norm_.data, config->norm_eps, config->dim);
    ffn_gpu(
        projected,
        state.lin1,
        state.lin2,
        norm_buffer,
        w1_.data,
        w2_.data,
        w3_.data,
        config->hidden_dim,
        config->dim
    );
    add_gpu(x, projected, config->dim);
}

auto Block::ForwardCPU(
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


    const auto q_dim = config->n_heads * config->head_dim;
    const auto kv_dim = config->n_kv_heads * config->head_dim;

    rmsnorm_cpu(norm_buffer, x, attn_norm_.data , config->norm_eps, config->dim);
    matmul_cpu(q, norm_buffer, wq_.data, q_dim, config->dim);
    matmul_cpu(k, norm_buffer, wk_.data, kv_dim, config->dim);
    matmul_cpu(v, norm_buffer, wv_.data, kv_dim, config->dim);

    for (auto head = 0; head < config->n_heads; ++head) {
        auto* q_head = q + head * config->head_dim;
        rmsnorm_cpu(q_head, q_head, q_norm_.data, config->norm_eps, config->head_dim);
    }
    for (auto head = 0; head < config->n_kv_heads; ++head) {
        auto* k_head = k + head * config->head_dim;
        rmsnorm_cpu(k_head, k_head, k_norm_.data, config->norm_eps, config->head_dim);
    }

    rope_cpu(q, q_dim, config->head_dim, pos, config->rope_theta, config->rotary_dim);
    rope_cpu(k, kv_dim, config->head_dim, pos, config->rope_theta, config->rotary_dim);

    for (auto i = 0; i < kv_dim; i++) {
        k_[i + kv_pos * kv_dim] = static_cast<std::bfloat16_t>(k[i]);
        v_[i + kv_pos * kv_dim] = static_cast<std::bfloat16_t>(v[i]);
    }

    // Keep sink tokens at a constant relative distance after the ring buffer fills.
    for (auto sink = 0; sink < num_sink; ++sink) {
        for (auto i = 0; i < kv_dim; i++) {
            k[i] = static_cast<float>(k_[sink * kv_dim + i]);
        }
        rope_cpu(k, kv_dim, config->head_dim, 1, config->rope_theta, config->rotary_dim);

        for (auto i = 0; i < kv_dim; i++) {
            k_[sink * kv_dim + i] = static_cast<std::bfloat16_t>(k[i]);
        }
    }

    const auto queries_per_kv_head = config->n_heads / config->n_kv_heads;
    int head;
    #pragma omp parallel for private(head)
    for (head = 0; head < config->n_heads; ++head) {
        const auto kv_head = head / queries_per_kv_head;
        attn_cpu(
            attn_output + head * config->head_dim,
            attn_scores + head * config->max_seq_len,
            q + head * config->head_dim,
            k_ + kv_head * config->head_dim,
            v_ + kv_head * config->head_dim,
            config->head_dim,
            config->n_kv_heads,
            kv_len
        );
    }

    matmul_cpu(projected, attn_output, wo_.data, config->dim, q_dim);
    for (auto i = 0; i < config->dim; ++i) {
        x[i] += projected[i];
    }

    rmsnorm_cpu(norm_buffer, x, mlp_norm_.data, config->norm_eps, config->dim);
    ffn_cpu(
        projected,
        state.lin1,
        state.lin2,
        norm_buffer,
        w1_.data,
        w2_.data,
        w3_.data,
        config->hidden_dim,
        config->dim
    );
    for (auto i = 0; i < config->dim; ++i) {
        x[i] += projected[i];
    }
}