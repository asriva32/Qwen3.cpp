#include "inference.h"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

std::string DTypeName(TensorDType dtype) {
    switch (dtype) {
        case TensorDType::Float32:
            return "f32";
        case TensorDType::BFloat16:
            return "bf16";
        case TensorDType::UInt8:
            return "u8";
        case TensorDType::Int32:
            return "i32";
    }
    throw std::runtime_error("Unknown tensor dtype");
}

std::size_t ParseTokenCount(std::string_view value, std::string_view option) {
    std::size_t count = 0;
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), count);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument(
            std::string(option) + " requires a non-negative integer");
    }
    return count;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string model_path = argc > 1 ? argv[1] : "Qwen3.bin";
        const std::string prompt = argc > 2 ? argv[2] : "Hello";
        const int context_length = argc > 3 ? std::stoi(argv[3]) : 512;
        auto apply_chat_template = true;
        auto max_generated_tokens = std::size_t{512};
        auto stop_on_eos = true;
        auto token_limit_was_set = false;

        for (auto i = 4; i < argc; ++i) {
            const auto option = std::string_view(argv[i]);
            if (option == "--raw") {
                apply_chat_template = false;
            } else if (option == "--max-tokens" || option == "--benchmark") {
                if (token_limit_was_set) {
                    throw std::invalid_argument(
                        "--max-tokens and --benchmark cannot be combined");
                }
                if (++i >= argc) {
                    throw std::invalid_argument(
                        std::string(option) + " requires a token count");
                }
                max_generated_tokens = ParseTokenCount(argv[i], option);
                stop_on_eos = option != "--benchmark";
                token_limit_was_set = true;
            } else {
                throw std::invalid_argument("Unknown option: " + std::string(option));
            }
        }

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
        if (!stop_on_eos) {
            std::cout << "Fixed-token benchmark: " << max_generated_tokens
                      << " decode steps\n";
        }
        const GenerationResult result = model.Generate(
            prompt,
            apply_chat_template,
            max_generated_tokens,
            stop_on_eos
        );
        std::cout << result.text << '\n';
        std::cout << std::fixed << std::setprecision(2)
                  << "prefill: " << result.stats.prompt_tokens << " tokens, "
                  << result.stats.prefill_seconds << " s, "
                  << result.stats.PrefillTokensPerSecond() << " tok/s\n"
                  << "decode: " << result.stats.generated_tokens << " tokens, "
                  << result.stats.decode_seconds << " s, "
                  << result.stats.DecodeTokensPerSecond() << " tok/s\n"
                  << "stop_reason: "
                  << (result.stats.stopped_on_eos ? "eos" : "token_limit")
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
