#include "inference.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void CheckNear(float actual, float expected, const std::string& message) {
    if (std::abs(actual - expected) > 1e-5f) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual));
    }
}

std::vector<float> Identity(int size) {
    std::vector<float> result(static_cast<std::size_t>(size) * size, 0.0f);
    for (int i = 0; i < size; ++i) {
        result[static_cast<std::size_t>(i) * size + i] = 1.0f;
    }
    return result;
}

BlockWeights MakeWeights() {
    BlockWeights weights;
    weights.attn_norm.assign(4, 1.0f);
    weights.q_norm.assign(2, 1.0f);
    weights.k_norm.assign(2, 1.0f);
    weights.wq.assign(16, 0.0f);
    weights.wk = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    weights.wv = {
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    weights.wo = Identity(4);
    weights.mlp_norm.assign(4, 1.0f);
    weights.w1.assign(8, 0.0f);
    weights.w2.assign(8, 0.0f);
    weights.w3.assign(8, 0.0f);
    return weights;
}

BlockWeights MakeExpandedQueryWeights() {
    BlockWeights weights;
    weights.attn_norm.assign(2, 1.0f);
    weights.q_norm.assign(2, 1.0f);
    weights.k_norm.assign(2, 1.0f);
    weights.wq.assign(8, 0.0f);
    weights.wk.assign(4, 0.0f);
    weights.wv.assign(4, 0.0f);
    weights.wo.assign(8, 0.0f);
    weights.mlp_norm.assign(2, 1.0f);
    weights.w1.assign(4, 0.0f);
    weights.w2.assign(4, 0.0f);
    weights.w3.assign(4, 0.0f);
    return weights;
}

}  // namespace

int main() {
    try {
        float rotary[] = {1.0f, 2.0f, 3.0f, 4.0f};
        ApplyRotaryEmb(rotary, 4, 4, 1, 10000.0f, 4);
        CheckNear(rotary[0], std::cos(1.0f) - 3.0f * std::sin(1.0f), "rotary first half 0");
        CheckNear(rotary[2], 3.0f * std::cos(1.0f) + std::sin(1.0f), "rotary second half 0");
        CheckNear(rotary[1], 2.0f * std::cos(0.01f) - 4.0f * std::sin(0.01f),
                  "rotary first half 1");
        CheckNear(rotary[3], 4.0f * std::cos(0.01f) + 2.0f * std::sin(0.01f),
                  "rotary second half 1");

        const std::vector<float> sample_logits = {-2.0f, 0.5f, 0.25f};
        Check(Sampler::Greedy(sample_logits) == 1, "Greedy sampler chose the wrong token");
        auto expanded_config = std::make_shared<Config>();
        expanded_config->dim = 2;
        expanded_config->hidden_dim = 2;
        expanded_config->head_dim = 2;
        expanded_config->n_heads = 2;
        expanded_config->n_kv_heads = 1;
        expanded_config->max_seq_len = 2;
        expanded_config->rotary_dim = 2;
        expanded_config->rope_theta = 10000.0f;
        expanded_config->norm_eps = 1e-5f;
        Block expanded_query_block(expanded_config, MakeExpandedQueryWeights());
        float expanded_x[] = {1.0f, -1.0f};
        State state(expanded_config);
        expanded_query_block.forward(expanded_x, 0, 0, 0, 1, state);
        CheckNear(expanded_x[0], 1.0f, "expanded-query residual 0");
        CheckNear(expanded_x[1], -1.0f, "expanded-query residual 1");

        auto config = std::make_shared<Config>();
        config->dim = 4;
        config->hidden_dim = 2;
        config->head_dim = 2;
        config->n_heads = 2;
        config->n_kv_heads = 1;
        config->max_seq_len = 4;
        config->rotary_dim = 2;
        config->rope_theta = 10000.0f;
        config->norm_eps = 1e-5f;

        Block block(config, MakeWeights());
        float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
        const float rms = std::sqrt(7.5f + config->norm_eps);
        state = State(config);
        block.forward(x, 0, 0, 0, 1, state);

        // Both query heads share the only KV head, so both receive the same value.
        CheckNear(x[0], 1.0f + 3.0f / rms, "first attention residual");
        CheckNear(x[1], 2.0f + 4.0f / rms, "second attention residual");
        CheckNear(x[2], 3.0f + 3.0f / rms, "GQA first value");
        CheckNear(x[3], 4.0f + 4.0f / rms, "GQA second value");

        const KVCache& cache = block.GetCache();
        const float key_rms = std::sqrt(
            ((1.0f / rms) * (1.0f / rms) + (2.0f / rms) * (2.0f / rms)) / 2.0f +
            config->norm_eps);
        CheckNear(cache.k_[0], (1.0f / rms) / key_rms, "cached normalized key 0");
        CheckNear(cache.k_[1], (2.0f / rms) / key_rms, "cached normalized key 1");
        CheckNear(cache.v_[0], 3.0f / rms, "cached value 0");
        CheckNear(cache.v_[1], 4.0f / rms, "cached value 1");

        block.ResetCache();
        Check(std::all_of(cache.k_.begin(), cache.k_.end(), [](float value) { return value == 0.0f; }),
              "ResetCache did not clear keys");
        Check(std::all_of(cache.v_.begin(), cache.v_.end(), [](float value) { return value == 0.0f; }),
              "ResetCache did not clear values");

        bool rejected = false;
        try {
            block.forward(x, 4, 1, 0, 4, state);
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        Check(rejected, "Block accepted a write into the sink region");

        std::cout << "block tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "block test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
