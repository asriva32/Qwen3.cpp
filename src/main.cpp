#include "inference.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string DTypeName(TensorDType dtype) {
    switch (dtype) {
        case TensorDType::Float32:
            return "f32";
        case TensorDType::UInt8:
            return "u8";
        case TensorDType::Int32:
            return "i32";
    }
    throw std::runtime_error("Unknown tensor dtype");
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "Qwen3.bin";
    const std::string prompt = argc > 2 ? argv[2] : "Hello";
    const int context_length = argc > 3 ? std::stoi(argv[3]) : 512;
    const bool apply_chat_template = argc <= 4 || std::string(argv[4]) != "--raw";

    try {
        Qwen3 model;
        model.Load(model_path);

        const Config* config = model.GetConfig();
        std::cout << "Loaded " << model_path << '\n';
        std::cout << "arch=" << config->arch
                  << " dtype=" << config->dtype
                  << " dim=" << config->dim
                  << " layers=" << config->n_layers
                  << " heads=" << config->n_heads
                  << " kv_heads=" << config->n_kv_heads
                  << " vocab=" << config->vocab_size << '\n';
        std::cout << "indexed_tensors=" << model.GetTensorIndex().size() << '\n';
        std::cout << "bos_token=" << model.GetTokenizer().Token(model.GetTokenizer().BosTokenId()) << '\n';
        std::cout << "eos_token=" << model.GetTokenizer().Token(model.GetTokenizer().EosTokenId()) << '\n';

        const auto embed_it = model.GetTensorIndex().find("model.embed.weight");
        if (embed_it != model.GetTensorIndex().end()) {
            const TensorInfo& info = embed_it->second;
            std::cout << "model.embed.weight "
                      << DTypeName(info.dtype)
                      << " [" << info.shape[0] << ", " << info.shape[1] << "] "
                      << info.byte_size << " bytes\n";
        }

        std::cout << "Initializing inference with context=" << context_length << "\n";
        model.InitializeInference(context_length);
        std::cout << "Prompt: " << prompt << "\n";
        const GenerationResult result = model.Generate(prompt, apply_chat_template);
        std::cout << result.text << '\n';
        std::cout << std::fixed << std::setprecision(2)
                  << "prefill: " << result.stats.prompt_tokens << " tokens, "
                  << result.stats.prefill_seconds << " s, "
                  << result.stats.PrefillTokensPerSecond() << " tok/s\n"
                  << "decode: " << result.stats.generated_tokens << " tokens, "
                  << result.stats.decode_seconds << " s, "
                  << result.stats.DecodeTokensPerSecond() << " tok/s\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
