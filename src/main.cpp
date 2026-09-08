#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <omp.h>

#include "cpu_impl.h"
#include "gpu_impl.h"
#include "sampler.h"
#include "tokenizer.h"

namespace {

enum class Device {
    Cpu,
    Gpu,
};

struct Options {
    std::string model_path;
    std::vector<std::int32_t> tokens;
    std::optional<std::string> prompt;
    Device device = Device::Cpu;
    int context_length = 512;
    std::size_t max_tokens = 128;
    float temperature = Sampler::kDefaultTemperature;
    std::optional<std::uint64_t> seed;
    int threads = 0;
    bool stop_on_eos = true;
    bool raw_prompt = false;
};

void PrintUsage(std::ostream& out) {
    out << "Usage: qwen3 --model PATH (--prompt TEXT | --tokens ID[,ID...]) [options]\n"
           "\n"
           "Options:\n"
           "  --prompt TEXT       Text prompt to tokenize\n"
           "  --tokens IDS       Pre-tokenized comma-separated token IDs\n"
           "  --raw               Do not apply the Qwen chat template to --prompt\n"
           "  --device DEVICE     Inference device: cpu or gpu (default: cpu)\n"
           "  --context-length N  Runtime context length (default: 512)\n"
           "  --max-tokens N      Maximum generated tokens (default: 128)\n"
           "  --temperature N     Sampling temperature (default: 0.6)\n"
           "  --seed N            Random seed for reproducible sampling\n"
           "  --greedy            Use deterministic argmax sampling\n"
           "  --threads N         OpenMP thread count\n"
           "  --no-eos            Ignore EOS and generate exactly --max-tokens\n"
           "  -h, --help          Show this help\n";
}

template <typename T>
T ParseInteger(std::string_view text, std::string_view option) {
    T value{};
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument(std::string(option) + " requires an integer");
    }
    return value;
}

std::vector<std::int32_t> ParseTokens(std::string_view text) {
    if (text.empty()) {
        throw std::invalid_argument("--tokens requires at least one token ID");
    }

    std::vector<std::int32_t> result;
    while (!text.empty()) {
        const auto separator = text.find(',');
        const auto item = text.substr(0, separator);
        if (item.empty()) {
            throw std::invalid_argument("--tokens contains an empty token ID");
        }
        result.push_back(ParseInteger<std::int32_t>(item, "--tokens"));
        if (separator == std::string_view::npos) {
            break;
        }
        text.remove_prefix(separator + 1);
    }
    return result;
}

Device ParseDevice(std::string_view value) {
    if (value == "cpu") {
        return Device::Cpu;
    }
    if (value == "gpu") {
        return Device::Gpu;
    }
    throw std::invalid_argument("--device must be either cpu or gpu");
}

std::unique_ptr<Model> CreateModel(const Options& options) {
    switch (options.device) {
        case Device::Cpu:
            return std::make_unique<CPUImpl>(
                options.model_path, options.context_length);
        case Device::Gpu:
            return std::make_unique<GPUImpl>(
                options.model_path, options.context_length);
    }
    throw std::invalid_argument("Unsupported inference device");
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option = argv[i];
        const auto value = [&]() -> std::string_view {
            if (++i >= argc) {
                throw std::invalid_argument(std::string(option) + " requires a value");
            }
            return argv[i];
        };

        if (option == "--model") {
            options.model_path = value();
        } else if (option == "--device") {
            options.device = ParseDevice(value());
        } else if (option == "--prompt") {
            options.prompt = value();
        } else if (option == "--tokens") {
            options.tokens = ParseTokens(value());
        } else if (option == "--context-length") {
            options.context_length = ParseInteger<int>(value(), option);
        } else if (option == "--max-tokens") {
            options.max_tokens = ParseInteger<std::size_t>(value(), option);
        } else if (option == "--temperature") {
            options.temperature = ParseInteger<float>(value(), option);
        } else if (option == "--seed") {
            options.seed = ParseInteger<std::uint64_t>(value(), option);
        } else if (option == "--greedy") {
            options.temperature = 0.0f;
        } else if (option == "--threads") {
            options.threads = ParseInteger<int>(value(), option);
        } else if (option == "--no-eos") {
            options.stop_on_eos = false;
        } else if (option == "--raw") {
            options.raw_prompt = true;
        } else if (option == "-h" || option == "--help") {
            PrintUsage(std::cout);
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown option: " + std::string(option));
        }
    }

    if (options.model_path.empty()) {
        throw std::invalid_argument("--model is required");
    }
    if (options.prompt.has_value() == !options.tokens.empty()) {
        throw std::invalid_argument("Provide exactly one of --prompt or --tokens");
    }
    if (options.context_length <= 0) {
        throw std::invalid_argument("--context-length must be greater than zero");
    }
    if (options.threads < 0) {
        throw std::invalid_argument("--threads must not be negative");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        if (options.threads > 0) {
            omp_set_num_threads(options.threads);
        }

        auto model = CreateModel(options);
        const Tokenizer tokenizer(*model);
        const auto prompt_tokens = options.prompt
            ? tokenizer.Encode(
                  options.raw_prompt ? *options.prompt : FormatChatPrompt(*options.prompt)
              )
            : options.tokens;

        const GenerationResult result = model->Generate(
            prompt_tokens,
            options.max_tokens,
            options.stop_on_eos,
            options.temperature,
            options.seed
        );

        std::cout << tokenizer.Decode(result.tokens) << '\n';
        std::cout << '\n' << std::fixed << std::setprecision(2)
                  << "prefill: " << result.stats.prompt_tokens << " tokens, "
                  << result.stats.PrefillTokensPerSecond() << " tok/s\n"
                  << "decode: " << result.stats.generated_tokens << " tokens, "
                  << result.stats.DecodeTokensPerSecond() << " tok/s\n"
                  << "stop_reason: "
                  << (result.stats.stopped_on_eos ? "eos" : "token_limit") << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        PrintUsage(std::cerr);
        return 1;
    }
    return 0;
}
