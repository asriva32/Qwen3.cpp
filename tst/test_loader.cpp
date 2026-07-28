#include "inference.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

const TensorInfo& FindTensor(const Qwen3& model, const std::string& name) {
    const auto& tensors = model.GetTensorIndex();
    const auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error("missing tensor: " + name);
    }
    return it->second;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "Qwen3.bin";

    try {
        Qwen3 model(model_path);

        Check(model.GetConfig() != nullptr, "missing model config");
        const Config& config = *model.GetConfig();
        Check(config.arch == "Qwen3ForCausalLM", "unexpected architecture");
        Check(config.dtype == "bf16", "unexpected dtype");
        Check(config.dim == 1024, "unexpected dim");
        Check(config.hidden_dim == 3072, "unexpected hidden dim");
        Check(config.head_dim == 128, "unexpected head dim");
        Check(config.n_layers == 28, "unexpected layer count");
        Check(config.n_heads == 16, "unexpected attention head count");
        Check(config.n_kv_heads == 8, "unexpected kv head count");
        Check(config.vocab_size == 151936, "unexpected vocab size");
        Check(config.bos_token_id >= 0, "invalid bos token id");
        Check(config.eos_token_id >= 0, "invalid eos token id");
        Check(config.tie_word_embeddings, "expected tied embeddings");
        Check(config.qk_norm, "expected q/k norm");

        Check(model.GetTensorIndex().size() == 313, "unexpected tensor record count");
        Check(model.GetTokenizer().HasBackend(), "tokenizer backend was not initialized");
        Check(static_cast<std::int32_t>(model.GetTokenizer().VocabSize()) == config.vocab_size,
              "tokenizer offset count mismatch");
        Check(model.GetTokenizer().BosTokenId() == config.bos_token_id, "tokenizer bos id mismatch");
        Check(model.GetTokenizer().EosTokenId() == config.eos_token_id, "tokenizer eos id mismatch");
        Check(model.GetTokenizer().Token(model.GetTokenizer().BosTokenId()) == "<|endoftext|>", "bad bos token text");
        Check(model.GetTokenizer().Token(model.GetTokenizer().EosTokenId()) == "<|im_end|>", "bad eos token text");
        const std::vector<std::int32_t> encoded = model.GetTokenizer().Encode("Hello");
        Check(!encoded.empty(), "tokenizer encode returned no tokens");
        Check(!model.GetTokenizer().Decode(encoded).empty(), "tokenizer decode returned empty text");

        const std::string embed_name = "model.embed.weight";
        const TensorInfo& embed = FindTensor(model, embed_name);
        Check(embed.dtype == TensorDType::BFloat16, "embedding dtype should be bf16");
        Check(embed.shape.size() == 2, "embedding should be rank 2");
        Check(embed.shape[0] == config.vocab_size, "embedding vocab dimension mismatch");
        Check(embed.shape[1] == config.dim, "embedding model dimension mismatch");
        Check(embed.byte_size == 311164928, "embedding byte size mismatch");
        Check(embed.data_offset > 0, "embedding data offset should be positive");

        const std::string qnorm_name = "model.layers.0.attn.q_norm.weight";
        const TensorInfo& qnorm = FindTensor(model, qnorm_name);
        Check(qnorm.dtype == TensorDType::BFloat16, "q_norm dtype should be bf16");
        Check(qnorm.shape.size() == 1, "q_norm should be rank 1");
        Check(qnorm.shape[0] == config.head_dim, "q_norm shape mismatch");

        const Tensor<std::bfloat16_t> qnorm_tensor =
            model.LoadTensor<std::bfloat16_t>(qnorm_name);
        Check(
            static_cast<std::int64_t>(
                qnorm_tensor.data.size() * sizeof(std::bfloat16_t)
            ) == qnorm.byte_size,
              "q_norm payload size mismatch");
        Check(qnorm_tensor.ptr() != nullptr, "q_norm bf16 view should not be null");
        Check(qnorm_tensor.data.size() == config.head_dim, "q_norm bf16 count mismatch");

        std::cout << "loader tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "loader test failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
