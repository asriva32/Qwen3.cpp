#include <immintrin.h>
#include <cmath>
#include "layers.h"

static auto RmsNormHelper(
    float *out,
    const float *x,
    const std::bfloat16_t *w,
    float eps,
    int n
) -> void;

// Transformer Block
Block::Block(Config* config) : config(config) {
    cache = KVCache(config->max_seq_len, config->n_kv_heads, config->head_dim);
}

Block::Block(Config* config, BlockWeights weights)
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

// ForwardPrefill
auto Block::ForwardPrefill(
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

    RmsNorm(
        norm_buffer.data(),
        x,
        weights.attn_norm.data(),
        config->norm_eps,
        config->dim,
        batch_size
    );

    // Batched Q/K/V projections over row-major token inputs.
    MatMul(
        q.data(),
        norm_buffer.data(),
        weights.wq.data(),
        q_dim,
        config->dim,
        batch_size
    );
    MatMul(
        k.data(),
        norm_buffer.data(),
        weights.wk.data(),
        kv_dim,
        config->dim,
        batch_size
    );

    MatMul(
        v.data(),
        norm_buffer.data(),
        weights.wv.data(),
        kv_dim,
        config->dim,
        batch_size
    );

    // Q/K norm
    #pragma omp parallel for collapse(2)
    for (size_t t = 0; t < num_tokens; ++t) {
        for (int head = 0; head < config->n_heads; ++head) {
            auto* q_head = q.data() + t * q_dim + head * config->head_dim;
            RmsNorm(q_head, q_head, weights.q_norm.data(), config->norm_eps,
                    config->head_dim);
        }
    }

    #pragma omp parallel for collapse(2)
    for (size_t t = 0; t < num_tokens; ++t) {
        for (int head = 0; head < config->n_kv_heads; ++head) {
            auto* k_head = k.data() + t * kv_dim + head * config->head_dim;
            RmsNorm(k_head, k_head, weights.k_norm.data(), config->norm_eps,
                    config->head_dim);
        }
    }

    // Rope

    #pragma omp parallel for
    for (size_t t = 0; t < num_tokens; ++t) {
        const auto token_pos = pos + static_cast<int>(t);

        ApplyRotaryEmb(
            q.data() + t * q_dim,
            q_dim,
            config->head_dim,
            token_pos,
            config->rope_theta,
            config->rotary_dim
        );

        ApplyRotaryEmb(
            k.data() + t * kv_dim,
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
        const float* k_token = k.data() + t * kv_dim;
        const float* v_token = v.data() + t * kv_dim;
        auto* cache_k = cache.k_.data() + kv_pos * kv_dim;
        auto* cache_v = cache.v_.data() + kv_pos * kv_dim;

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
            FastAttn(
                attn_output.data() + t * q_dim + head * config->head_dim,
                attn_scores.data() + (t * config->n_heads + head) * config->max_seq_len,
                q.data() + t * q_dim + head * config->head_dim,
                cache.k_.data() + kv_head * config->head_dim,
                cache.v_.data() + kv_head * config->head_dim,
                config->head_dim,
                config->n_kv_heads,
                kv_len
            );
        }
    }

    // output projection

    MatMul(
        projected.data(),
        attn_output.data(),
        weights.wo.data(),
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

    RmsNorm(
        norm_buffer.data(),
        x,
        weights.mlp_norm.data(),
        config->norm_eps,
        config->dim,
        batch_size
    );

    
    // batched ffn
    FeedForwardNetwork(
        projected.data(),
        state.lin1.data(),
        state.lin2.data(),
        norm_buffer.data(),
        weights.w1.data(),
        weights.w2.data(),
        weights.w3.data(),
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

auto Block::Forward(
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

    RmsNorm(norm_buffer.data(), x, weights.attn_norm.data() , config->norm_eps, config->dim);
    MatMul(q.data(), norm_buffer.data(), weights.wq.data(), q_dim, config->dim);
    MatMul(k.data(), norm_buffer.data(), weights.wk.data(), kv_dim, config->dim);
    MatMul(v.data(), norm_buffer.data(), weights.wv.data(), kv_dim, config->dim);
    
    for (auto head = 0; head < config->n_heads; ++head) {
        auto* q_head = q.data() + head * config->head_dim;
        RmsNorm(q_head, q_head, weights.q_norm.data(), config->norm_eps, config->head_dim);
    }
    for (auto head = 0; head < config->n_kv_heads; ++head) {
        auto* k_head = k.data() + head * config->head_dim;
        RmsNorm(k_head, k_head, weights.k_norm.data(), config->norm_eps, config->head_dim);
    }

    ApplyRotaryEmb(q.data(), q_dim, config->head_dim, pos, config->rope_theta, config->rotary_dim);
    ApplyRotaryEmb(k.data(), kv_dim, config->head_dim, pos, config->rope_theta, config->rotary_dim);

    for (auto i = 0; i < kv_dim; i++) {
        cache.k_[i + kv_pos * kv_dim] = static_cast<std::bfloat16_t>(k[i]);
        cache.v_[i + kv_pos * kv_dim] = static_cast<std::bfloat16_t>(v[i]);
    }

    // Keep sink tokens at a constant relative distance after the ring buffer fills.
    for (auto sink = 0; sink < num_sink; ++sink) {
        for (auto i = 0; i < kv_dim; i++) {
            k[i] = static_cast<float>(cache.k_[sink * kv_dim + i]);
        }
        ApplyRotaryEmb(k.data(), kv_dim, config->head_dim, 1, config->rope_theta, config->rotary_dim);

        for (auto i = 0; i < kv_dim; i++) {
            cache.k_[sink * kv_dim + i] = static_cast<std::bfloat16_t>(k[i]);
        }
    }

    const auto queries_per_kv_head = config->n_heads / config->n_kv_heads;
    int head;
    #pragma omp parallel for private(head)
    for (head = 0; head < config->n_heads; ++head) {
        const auto kv_head = head / queries_per_kv_head;
        FastAttn(
            attn_output.data() + head * config->head_dim,
            attn_scores.data() + head * config->max_seq_len,
            q.data() + head * config->head_dim,
            cache.k_.data() + kv_head * config->head_dim,
            cache.v_.data() + kv_head * config->head_dim,
            config->head_dim,
            config->n_kv_heads,
            kv_len
        );
    }

    MatMul(projected.data(), attn_output.data(), weights.wo.data(), config->dim, q_dim);
    for (auto i = 0; i < config->dim; ++i) {
        x[i] += projected[i];
    }

    RmsNorm(norm_buffer.data(), x, weights.mlp_norm.data(), config->norm_eps, config->dim);
    FeedForwardNetwork(
        projected.data(),
        state.lin1.data(),
        state.lin2.data(),
        norm_buffer.data(),
        weights.w1.data(),
        weights.w2.data(),
        weights.w3.data(),
        config->hidden_dim,
        config->dim
    );
    for (auto i = 0; i < config->dim; ++i) {
        x[i] += projected[i];
    }
}

// SIMD helpers
auto Reduce(__m256 value) -> float {
    const __m128 low  = _mm256_extractf128_ps(value, 0);
    const __m128 high = _mm256_extractf128_ps(value, 1);

    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    return _mm_cvtss_f32(sum);
}

auto LoadBF16(const std::bfloat16_t *p) -> __m256 {
    const __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    __m256i wide = _mm256_cvtepu16_epi32(tmp);
    wide = _mm256_slli_epi32(wide, 16);
    return _mm256_castsi256_ps(wide);
}

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
}

// Layers
// tentative implementation
auto RmsNorm(
    float* out,
    const float* x,
    const std::bfloat16_t* weight,
    float eps,
    int dim,
    int batch_size
) -> void 
{
    if (batch_size == 1) {
        RmsNormHelper(out, x, weight, eps, dim);
        return;
    }
    #pragma omp parallel for
    for (int t = 0; t < batch_size; ++t) {
        RmsNormHelper(
            out + static_cast<size_t>(t) * dim,
            x   + static_cast<size_t>(t) * dim,
            weight,
            eps,
            dim
        );
    }
}

static auto RmsNormHelper(
    float *out,
    const float *x, 
    const std::bfloat16_t *w, 
    float eps,
    int n
) -> void 
{
    
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
) -> void 
{

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
    int m,
    int batch_size
) -> void {
    // (n, m) x (m, ) = (n, )
    if (batch_size == 1) {
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < n; i++) {
            const auto *w_row = w + static_cast<size_t>(i) * m;
            out[i] = FastDotProduct(w_row, x, m);
        }
        return;
    }

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            const auto* x_row = x + static_cast<size_t>(b) * m;
            const auto *w_row = w + static_cast<size_t>(i) * m;
            out[b * n + i] = FastDotProduct(w_row, x_row, m);
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
    int dim,
    int batch_size
) -> void {
    
    MatMul(lin1, x, w1, hidden_dim, dim, batch_size);
    MatMul(lin2, x, w3, hidden_dim, dim, batch_size);
    
    const auto count = batch_size * hidden_dim;
    for (auto i = 0; i < count; i++) {
        lin1[i] = Silu(lin1[i]) * lin2[i];
    }

    MatMul(out, lin1, w2, dim, hidden_dim, batch_size);
    
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
