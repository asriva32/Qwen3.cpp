#ifndef LAYERS_H
#define LAYERS_H
#include <stdfloat>
#include <vector>
#include "config.h"
#include "state.h"

struct WeightTensor {
    size_t size = 0;
    std::bfloat16_t *data = nullptr;
};

class Block {
friend class Model;

public:
    void ForwardCPU(float* x, int pos, int num_sink, int kv_pos, int kv_len, State &state);
    void ForwardGPU(float* x, int pos, int num_sink, int kv_pos, int kv_len, State &state);
    void ForwardPrefillCPU(float* x, size_t num_tokens, int pos, State& state);
    void ForwardPrefillGPU(float* x, size_t num_tokens, int pos, State& state);
    
    explicit Block(Config* config, Device device);
    ~Block();

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;
    Block(Block&& other) noexcept;
    Block& operator=(Block&& other) noexcept;

    void ResetCache();

    Config* GetConfig() {
        return config;
    }

private:
    void Release() noexcept;
    void TakeOwnership(Block& other) noexcept;
    void ValidateWeights() const;

    std::bfloat16_t *k_ = nullptr; // (seq_len, n_kv_heads * head_dim)
    std::bfloat16_t *v_ = nullptr; // (seq_len, n_kv_heads * head_dim)
    WeightTensor attn_norm_;
    WeightTensor q_norm_;
    WeightTensor k_norm_;
    WeightTensor wq_;
    WeightTensor wk_;
    WeightTensor wv_;
    WeightTensor wo_;
    WeightTensor mlp_norm_;
    WeightTensor w1_;
    WeightTensor w2_;
    WeightTensor w3_;
    Config* config = nullptr;
    Device device_ = Device::CPU;
};
// cpu impl
void  rmsnorm_cpu(float *out, const float *x, const std::bfloat16_t *weights, float eps, int n, int batch_size = 1);
void  softmax_cpu(float *out, const float *x, int n);
float silu_cpu(float x);
void  matmul_cpu(float *out, const float *x, const std::bfloat16_t *y, int n, int m, int batch_size = 1);
void  rope_cpu(float *out, int d, int head_dim, int pos, float theta, int rotary_dim);
void  ffn_cpu(float *out, float *lin1, float *lin2, const float *x, const std::bfloat16_t *w1, const std::bfloat16_t *w2, const std::bfloat16_t *w3, int hidden_dim, int dim, int batch_size = 1);
void  attn_cpu(float *out, float *atth, const float *q, const std::bfloat16_t *k, const std::bfloat16_t *v, int head_dim, int n_kv_heads, int kv_len);
// gpu impl
void  rmsnorm_gpu(float *out, const float *x, const std::bfloat16_t *weights, float eps, int n, int batch_size = 1);
void  matmul_gpu(float *out, const float *x, const std::bfloat16_t *y, int n, int m, int batch_size = 1);
void  qk_norm_rope_and_update_cache(float *q, float *k, const float *v, std::bfloat16_t *cache_k, std::bfloat16_t *cache_v, const std::bfloat16_t *q_norm, const std::bfloat16_t *k_norm, int n_heads, int n_kv_heads, int head_dim, int pos, int kv_pos, float norm_eps, float theta, int rotary_dim, int batch_size = 1);
void  rotate_sink_tokens(std::bfloat16_t *cache_k, size_t num_sink, size_t kv_dim, int head_dim, float rope_theta, int rotary_dim);
void  ffn_gpu(float *out, float *lin1, float *lin2, const float *x, const std::bfloat16_t *w1, const std::bfloat16_t *w2, const std::bfloat16_t *w3, int hidden_dim, int dim, int batch_size = 1);
void  attn_gpu(float *out, float *atth, const float *q, const std::bfloat16_t *k, const std::bfloat16_t *v, int head_dim, int n_heads, int n_kv_heads, int kv_len_start, int max_seq_len, int batch_size = 1);
void  add_gpu(float *destination, const float *source, size_t count);
void  embedding_gpu(float *out, const std::bfloat16_t *embedding_row, int dim);
void  argmax_gpu(std::int32_t *out, const float *values, int count);

#endif
