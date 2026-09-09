#ifndef LAYERS_H
#define LAYERS_H
#include <cuda_bf16.h>
#include <stdfloat>
#include <vector>
#include <memory>
#include "config.h"
#include "state.h"

struct KVCache {
    std::vector<std::bfloat16_t> k_; // (seq_len, n_kv_heads * head_dim)
    std::vector<std::bfloat16_t> v_; // (seq_len, n_kv_heads * head_dim)
    int seq_len = 0;
    int n_kv_heads = 0;
    int head_dim = 0;

    KVCache() = default;
    KVCache(int seq_len, int n_kv_heads, int head_dim) : 
        seq_len(seq_len), n_kv_heads(n_kv_heads), head_dim(head_dim) {
            k_.resize(seq_len * n_kv_heads * head_dim);
            v_.resize(seq_len * n_kv_heads * head_dim);
        }
};

struct BlockWeights {
    using bf16 = std::bfloat16_t;
    std::vector<bf16> attn_norm;
    std::vector<bf16> q_norm;
    std::vector<bf16> k_norm;
    std::vector<bf16> wq;
    std::vector<bf16> wk;
    std::vector<bf16> wv;
    std::vector<bf16> wo;
    std::vector<bf16> mlp_norm;
    std::vector<bf16> w1;
    std::vector<bf16> w2;
    std::vector<bf16> w3;
};

class Block {
friend class Model;

public:
    void Forward(float* x, int pos, int num_sink, int kv_pos, int kv_len, State &state);
    void ForwardPrefill(float* x, size_t num_tokens, int pos, State& state);
    explicit Block(Config* config);
    Block(Config* config, BlockWeights weights);

    void ResetCache();

    Config* GetConfig() {
        return config;
    }

    const KVCache& GetCache() const {
        return cache;
    }
private:
    void ValidateWeights() const;

    Config* config;
    BlockWeights weights;
    KVCache cache;
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
void  softmax_gpu(float *out, const float *x, int n);
float silu_gpu(float x);
void  matmul_gpu(float *out, const float *x, const std::bfloat16_t *y, int n, int m, int batch_size = 1);
void  rope_gpu(float *out, int d, int head_dim, int pos, float theta, int rotary_dim);
void  ffn_gpu(float *out, float *lin1, float *lin2, const float *x, const std::bfloat16_t *w1, const std::bfloat16_t *w2, const std::bfloat16_t *w3, int hidden_dim, int dim, int batch_size = 1);
void  attn_gpu(float *out, float *atth, const float *q, const std::bfloat16_t *k, const std::bfloat16_t *v, int head_dim, int n_kv_heads, int kv_len);


#endif
