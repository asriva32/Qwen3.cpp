#ifndef STATE_H
#define STATE_H
#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>
#include "config.h"
#include "utils.h"

inline constexpr size_t kPrefillBatchSize = 128;

// TODO: migrate to an arena
// Something like one arena per thread
// separate arena for q, k, v, projected, norm_buffer
struct State {
    // From ffn
    float* lin1;
    float* lin2;
    // From Block
    float* norm_buffer;
    float* q;
    float* k;
    float* v;
    float* attn_output;
    float* projected;
    float* attn_scores;
    size_t batch_capacity;
    Device device;

    explicit State(Config *c, Device device)
        : batch_capacity(std::min(
            kPrefillBatchSize,
            static_cast<size_t>(c->max_seq_len)
        ))
        , device(device) {
        const auto q_dim = c->n_heads * c->head_dim;
        const auto kv_dim = c->n_kv_heads * c->head_dim;

        if (device == Device::CPU) {
            lin1 = new float[batch_capacity * static_cast<size_t>(c->hidden_dim)];
            lin2 = new float[batch_capacity * static_cast<size_t>(c->hidden_dim)];
            norm_buffer = new float[batch_capacity * static_cast<size_t>(c->dim)];
            q = new float[batch_capacity * static_cast<size_t>(q_dim)];
            k = new float[batch_capacity * static_cast<size_t>(kv_dim)];
            v = new float[batch_capacity * static_cast<size_t>(kv_dim)];
            attn_output = new float[batch_capacity * static_cast<size_t>(q_dim)];
            projected = new float[batch_capacity * static_cast<size_t>(c->dim)];
            attn_scores = new float[batch_capacity * static_cast<size_t>(c->n_heads) * static_cast<size_t>(c->max_seq_len)];
        } else {
            
        }
    }

    ~State() {
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
        } else {

        }
    }
};

#endif
