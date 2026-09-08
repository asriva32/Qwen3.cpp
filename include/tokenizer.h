#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

class Model;

// Dependency-free byte-level BPE tokenizer. The model constructor loads the
// vocabulary and ranked merge table embedded in a converted Qwen model.
class Tokenizer {
public:
    using Merge = std::pair<std::string, std::string>;

    explicit Tokenizer(const Model& model);
    Tokenizer(std::vector<std::string> vocabulary, std::vector<Merge> merges);

    [[nodiscard]] std::vector<std::int32_t> Encode(std::string_view text) const;
    [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
    [[nodiscard]] std::string_view Token(std::int32_t id) const;
    [[nodiscard]] std::size_t VocabSize() const noexcept;

private:
    struct PairHash {
        std::size_t operator()(const Merge& pair) const noexcept;
    };

    void Initialize(std::vector<Merge> merges);
    void EncodeOrdinary(std::string_view text, std::vector<std::int32_t>& out) const;
    void EncodePiece(std::string_view piece, std::vector<std::int32_t>& out) const;

    std::vector<std::string> vocabulary_;
    std::unordered_map<std::string, std::int32_t> token_ids_;
    std::unordered_map<Merge, std::size_t, PairHash> merge_ranks_;
    std::vector<std::pair<std::string, std::int32_t>> special_tokens_;
};

[[nodiscard]] std::string FormatChatPrompt(std::string_view prompt);

#endif
