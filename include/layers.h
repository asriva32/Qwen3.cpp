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
    void Forward(float* x, int pos, int num_sink, int kv_pos, int kv_len, State &state);
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
// first implement rmsnorm (seems the easiest)
// also need to add 4bit quantization after implementing all these kernels
void  rmsnorm_gpu(float *out, const float *x, const std::bfloat16_t *weights, float eps, int n, int batch_size = 1);
void  matmul_gpu(float *out, const float *x, const std::bfloat16_t *y, int n, int m, int batch_size = 1);
void  rope_gpu(float *out, int d, int head_dim, int pos, float theta, int rotary_dim);
void  ffn_gpu(float *out, float *lin1, float *lin2, const float *x, const std::bfloat16_t *w1, const std::bfloat16_t *w2, const std::bfloat16_t *w3, int hidden_dim, int dim, int batch_size = 1);
void  attn_gpu(float *out, float *atth, const float *q, const std::bfloat16_t *k, const std::bfloat16_t *v, int head_dim, int n_kv_heads, int kv_len);

#endif
