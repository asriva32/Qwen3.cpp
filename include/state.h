#ifndef STATE_H
#define STATE_H
#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include "config.h"
#include "utils.h"

extern "C" void* upload_cuda(void* device, size_t size);
extern "C" void* allocate_cuda_zeroed(size_t size);
extern "C" void zero_cuda(void* device, size_t size);
extern "C" void free_cuda(void* device);
extern "C" void set_cuda_device(int device);
extern "C" void download_cuda(void* host, const void* device, size_t size);
extern "C" void synchronize_cuda();

inline constexpr size_t kPrefillBatchSize = 128;

struct State {
    // From ffn
    float* lin1 = nullptr;
    float* lin2 = nullptr;
    // From Block
    float* norm_buffer = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* attn_output = nullptr;
    float* projected = nullptr;
    float* attn_scores = nullptr;
    std::int32_t* sampled_token = nullptr;
    size_t batch_capacity = 0;
    Device device = Device::CPU;

    explicit State(Config *c, Device device)
        : batch_capacity(std::min(
            kPrefillBatchSize,
            static_cast<size_t>(c->max_seq_len)
        ))
        , device(device) {
        const auto q_dim = c->n_heads * c->head_dim;
        const auto kv_dim = c->n_kv_heads * c->head_dim;

        if (device == Device::CPU) {
            lin1        = new float[batch_capacity * static_cast<size_t>(c->hidden_dim)];
            lin2        = new float[batch_capacity * static_cast<size_t>(c->hidden_dim)];
            norm_buffer = new float[batch_capacity * static_cast<size_t>(c->dim)];
            q           = new float[batch_capacity * static_cast<size_t>(q_dim)];
            k           = new float[batch_capacity * static_cast<size_t>(kv_dim)];
            v           = new float[batch_capacity * static_cast<size_t>(kv_dim)];
            attn_output = new float[batch_capacity * static_cast<size_t>(q_dim)];
            projected   = new float[batch_capacity * static_cast<size_t>(c->dim)];
            attn_scores = new float[batch_capacity * static_cast<size_t>(c->n_heads) 
                * static_cast<size_t>(c->max_seq_len)];
            sampled_token = new std::int32_t[1];
        } else {
            const auto allocate = [](size_t count) {
                return static_cast<float*>(
                    allocate_cuda_zeroed(count * sizeof(float))
                );
            };
            lin1        = allocate(batch_capacity * static_cast<size_t>(c->hidden_dim));
            lin2        = allocate(batch_capacity * static_cast<size_t>(c->hidden_dim));
            norm_buffer = allocate(batch_capacity * static_cast<size_t>(c->dim));
            q           = allocate(batch_capacity * static_cast<size_t>(q_dim));
            k           = allocate(batch_capacity * static_cast<size_t>(kv_dim));
            v           = allocate(batch_capacity * static_cast<size_t>(kv_dim));
            attn_output = allocate(batch_capacity * static_cast<size_t>(q_dim));
            projected   = allocate(batch_capacity * static_cast<size_t>(c->dim));
            attn_scores = allocate(batch_capacity * static_cast<size_t>(c->n_heads) * static_cast<size_t>(c->max_seq_len));
            sampled_token = static_cast<std::int32_t*>(
                allocate_cuda_zeroed(sizeof(std::int32_t))
            );
        }
    }

    ~State() {
        Release();
    }

    State(const State&) = delete;
    State& operator=(const State&) = delete;

    State(State&& other) noexcept {
        TakeOwnership(other);
    }

    State& operator=(State&& other) noexcept {
        if (this != &other) {
            Release();
            TakeOwnership(other);
        }
        return *this;
    }

private:
    void Release() noexcept {
        if (device == Device::CPU) {
            delete[] lin1;
            delete[] lin2;
            delete[] norm_buffer;
            delete[] q;
            delete[] k;
            delete[] v;
            delete[] attn_output;
            delete[] projected;
            delete[] attn_scores;
            delete[] sampled_token;
        } else {
            free_cuda(lin1);
            free_cuda(lin2);
            free_cuda(norm_buffer);
            free_cuda(q);
            free_cuda(k);
            free_cuda(v);
            free_cuda(attn_output);
            free_cuda(projected);
            free_cuda(attn_scores);
            free_cuda(sampled_token);
        }
        lin1 = nullptr;
        lin2 = nullptr;
        norm_buffer = nullptr;
        q = nullptr;
        k = nullptr;
        v = nullptr;
        attn_output = nullptr;
        projected = nullptr;
        attn_scores = nullptr;
        sampled_token = nullptr;
        batch_capacity = 0;
    }

    void TakeOwnership(State& other) noexcept {
        lin1 = std::exchange(other.lin1, nullptr);
        lin2 = std::exchange(other.lin2, nullptr);
        norm_buffer = std::exchange(other.norm_buffer, nullptr);
        q = std::exchange(other.q, nullptr);
        k = std::exchange(other.k, nullptr);
        v = std::exchange(other.v, nullptr);
        attn_output = std::exchange(other.attn_output, nullptr);
        projected = std::exchange(other.projected, nullptr);
        attn_scores = std::exchange(other.attn_scores, nullptr);
        sampled_token = std::exchange(other.sampled_token, nullptr);
        batch_capacity = std::exchange(other.batch_capacity, 0);
        device = other.device;
    }
};

#endif
