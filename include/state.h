#ifndef STATE_H
#define STATE_H
#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>
#include "config.h"

inline constexpr size_t kPrefillBatchSize = 128;

// TODO: migrate to an arena
// Something like one arena per thread
// separate arena for q, k, v, projected, norm_buffer
struct State {
    // From RMSNorm
    std::vector<float> lin1;
    std::vector<float> lin2;
    // From Block
    std::vector<float> norm_buffer;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> attn_output;
    std::vector<float> projected;
    std::vector<float> attn_scores;
    size_t batch_capacity;

    explicit State(Config *c)
        : batch_capacity(std::min(
            kPrefillBatchSize,
            static_cast<size_t>(c->max_seq_len)
        )) {
        const auto q_dim = c->n_heads * c->head_dim;
        const auto kv_dim = c->n_kv_heads * c->head_dim;

        lin1.resize(batch_capacity * static_cast<size_t>(c->hidden_dim));
        lin2.resize(batch_capacity * static_cast<size_t>(c->hidden_dim));
        norm_buffer.resize(batch_capacity * static_cast<size_t>(c->dim));
        q.resize(batch_capacity * static_cast<size_t>(q_dim));
        k.resize(batch_capacity * static_cast<size_t>(kv_dim));
        v.resize(batch_capacity * static_cast<size_t>(kv_dim));
        attn_output.resize(batch_capacity * static_cast<size_t>(q_dim));
        projected.resize(batch_capacity * static_cast<size_t>(c->dim));
        attn_scores.resize(
            batch_capacity * static_cast<size_t>(c->n_heads) * static_cast<size_t>(c->max_seq_len)
        );
    }
};

#endif
