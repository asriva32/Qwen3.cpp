#ifndef CONFIG_H
#define CONFIG_H
#include <cstdint>
#include <stdexcept>
#include <string>

class Config {
public:
    std::string arch;
    std::string dtype;
    std::string act_type;

    std::int32_t dim = 0;
    std::int32_t hidden_dim = 0;
    std::int32_t head_dim = 0;
    std::int32_t n_layers = 0;
    std::int32_t n_heads = 0;
    std::int32_t n_kv_heads = 0;
    std::int32_t vocab_size = 0;
    std::int32_t max_seq_len = 0;
    std::int32_t bos_token_id = 0;
    std::int32_t eos_token_id = 0;
    std::int32_t rotary_dim = 0;

    float rope_theta = 0.0f;
    float norm_eps = 0.0f;

    bool tie_word_embeddings = false;
    bool attention_bias = false;
    bool qk_norm = false;

    auto ValidateConfig() -> void {
        if (dim <= 0 || hidden_dim <= 0 || head_dim <= 0 ||
            n_heads <= 0 || n_kv_heads <= 0 || max_seq_len <= 0) {
            throw std::invalid_argument("Block config dimensions must be positive");
        }
        if (n_heads % n_kv_heads != 0) {
            throw std::invalid_argument("Attention head count must be divisible by KV head count");
        }
        if (rotary_dim < 0 || rotary_dim > head_dim ||
            rotary_dim % 2 != 0) {
            throw std::invalid_argument("rotary_dim must be even and no larger than head_dim");
        }
    }

    Config(const std::string& json);
};

#endif
