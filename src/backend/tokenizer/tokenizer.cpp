#include "tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <stdexcept>

#include "model.h"

namespace {

class JsonCursor {
public:
    explicit JsonCursor(std::string_view json) : json_(json) {}

    void SeekToKey(std::string_view key) {
        const auto marker = '"' + std::string(key) + '"';
        position_ = json_.find(marker, position_);
        if (position_ == std::string_view::npos) {
            throw std::runtime_error("Missing tokenizer JSON key: " + std::string(key));
        }
        position_ += marker.size();
        SkipWhitespace();
        Expect(':');
    }

    void SkipWhitespace() {
        while (position_ < json_.size() &&
               std::isspace(static_cast<unsigned char>(json_[position_]))) {
            ++position_;
        }
    }

    bool Consume(char value) {
        SkipWhitespace();
        if (position_ < json_.size() && json_[position_] == value) {
            ++position_;
            return true;
        }
        return false;
    }

    void Expect(char value) {
        if (!Consume(value)) {
            throw std::runtime_error("Malformed tokenizer JSON");
        }
    }

    std::string String() {
        SkipWhitespace();
        Expect('"');
        std::string result;
        while (position_ < json_.size()) {
            const char value = json_[position_++];
            if (value == '"') return result;
            if (value != '\\') {
                result.push_back(value);
                continue;
            }
            if (position_ >= json_.size()) break;
            const char escaped = json_[position_++];
            switch (escaped) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': AppendUnicodeEscape(result); break;
                default: throw std::runtime_error("Invalid tokenizer JSON escape");
            }
        }
        throw std::runtime_error("Unterminated tokenizer JSON string");
    }

private:
    static int Hex(char value) {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        throw std::runtime_error("Invalid tokenizer Unicode escape");
    }

    std::uint32_t UnicodeEscape() {
        if (position_ + 4 > json_.size()) {
            throw std::runtime_error("Truncated tokenizer Unicode escape");
        }
        std::uint32_t codepoint = 0;
        for (int i = 0; i < 4; ++i) codepoint = codepoint * 16 + Hex(json_[position_++]);
        return codepoint;
    }

    static void AppendUtf8(std::string& output, std::uint32_t codepoint) {
        if (codepoint <= 0x7f) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ff) {
            output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0xffff) {
            output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    void AppendUnicodeEscape(std::string& output) {
        auto codepoint = UnicodeEscape();
        if (codepoint >= 0xd800 && codepoint <= 0xdbff &&
            position_ + 6 <= json_.size() && json_[position_] == '\\' &&
            json_[position_ + 1] == 'u') {
            position_ += 2;
            const auto low = UnicodeEscape();
            if (low < 0xdc00 || low > 0xdfff) {
                throw std::runtime_error("Invalid tokenizer surrogate pair");
            }
            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
        }
        AppendUtf8(output, codepoint);
    }

    std::string_view json_;
    std::size_t position_ = 0;
};

std::vector<std::string> LoadVocabulary(const Model& model) {
    const auto bytes = model.LoadTensor<std::uint8_t>("tokenizer.tokens").data;
    const auto offsets = model.LoadTensor<std::int32_t>("tokenizer.offsets").data;
    std::vector<std::string> vocabulary;
    vocabulary.reserve(offsets.size());

    for (const auto offset : offsets) {
        if (offset < 0 || static_cast<std::size_t>(offset) >= bytes.size()) {
            throw std::runtime_error("Tokenizer offset is outside the token table");
        }
        const auto begin = bytes.begin() + offset;
        const auto end = std::find(begin, bytes.end(), std::uint8_t{0});
        if (end == bytes.end()) {
            throw std::runtime_error("Tokenizer token is missing its terminator");
        }
        std::string token;
        token.reserve(static_cast<std::size_t>(end - begin));
        for (auto it = begin; it != end; ++it) {
            token.push_back(static_cast<char>(*it == 7 ? 0 : *it));
        }
        vocabulary.push_back(std::move(token));
    }
    return vocabulary;
}

std::uint32_t NextCodepoint(std::string_view text, std::size_t& offset) {
    const auto first = static_cast<std::uint8_t>(text[offset++]);
    if (first < 0x80) return first;
    int continuation_count = 0;
    std::uint32_t codepoint = 0;
    if ((first & 0xe0) == 0xc0) {
        continuation_count = 1;
        codepoint = first & 0x1f;
    } else if ((first & 0xf0) == 0xe0) {
        continuation_count = 2;
        codepoint = first & 0x0f;
    } else if ((first & 0xf8) == 0xf0) {
        continuation_count = 3;
        codepoint = first & 0x07;
    } else {
        throw std::runtime_error("Invalid UTF-8 in tokenizer merge table");
    }
    for (int i = 0; i < continuation_count; ++i) {
        if (offset >= text.size()) {
            throw std::runtime_error("Truncated UTF-8 in tokenizer merge table");
        }
        const auto byte = static_cast<std::uint8_t>(text[offset++]);
        if ((byte & 0xc0) != 0x80) {
            throw std::runtime_error("Invalid UTF-8 continuation byte");
        }
        codepoint = (codepoint << 6) | (byte & 0x3f);
    }
    return codepoint;
}

std::array<int, 512> ByteDecoder() {
    std::array<int, 512> decoder;
    decoder.fill(-1);
    std::array<bool, 256> direct{};
    for (int byte = 33; byte <= 126; ++byte) direct[byte] = true;
    for (int byte = 161; byte <= 172; ++byte) direct[byte] = true;
    for (int byte = 174; byte <= 255; ++byte) direct[byte] = true;
    for (int byte = 0; byte < 256; ++byte) {
        if (direct[byte]) decoder[byte] = byte;
    }
    int extra = 0;
    for (int byte = 0; byte < 256; ++byte) {
        if (!direct[byte]) decoder[256 + extra++] = byte;
    }
    return decoder;
}

std::string DecodeByteAlphabet(std::string_view encoded) {
    static const auto decoder = ByteDecoder();
    std::string result;
    for (std::size_t offset = 0; offset < encoded.size();) {
        const auto codepoint = NextCodepoint(encoded, offset);
        if (codepoint >= decoder.size() || decoder[codepoint] < 0) {
            throw std::runtime_error("Unexpected symbol in tokenizer merge table");
        }
        result.push_back(static_cast<char>(decoder[codepoint]));
    }
    return result;
}

std::vector<Tokenizer::Merge> LoadMerges(const Model& model) {
    const auto json_bytes = model.LoadTensor<std::uint8_t>("tokenizer.json").data;
    const std::string_view json(
        reinterpret_cast<const char*>(json_bytes.data()), json_bytes.size());
    JsonCursor cursor(json);
    cursor.SeekToKey("merges");
    cursor.Expect('[');

    std::vector<Tokenizer::Merge> merges;
    while (true) {
        if (cursor.Consume(']')) break;
        cursor.Expect('[');
        auto left = DecodeByteAlphabet(cursor.String());
        cursor.Expect(',');
        auto right = DecodeByteAlphabet(cursor.String());
        cursor.Expect(']');
        merges.emplace_back(std::move(left), std::move(right));
        if (cursor.Consume(']')) break;
        cursor.Expect(',');
    }
    return merges;
}

bool IsAsciiLetter(unsigned char value) {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z');
}

bool IsLetter(unsigned char value) {
    return IsAsciiLetter(value) || value >= 0x80;
}

bool IsNumber(unsigned char value) {
    return value >= '0' && value <= '9';
}

bool IsNewline(unsigned char value) {
    return value == '\r' || value == '\n';
}

bool IsSpace(unsigned char value) {
    return std::isspace(value) != 0;
}

bool IsPunctuation(unsigned char value) {
    return !IsSpace(value) && !IsLetter(value) && !IsNumber(value);
}

std::size_t ContractionLength(std::string_view text, std::size_t offset) {
    static constexpr std::array contractions{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    for (const std::string_view contraction : contractions) {
        if (offset + contraction.size() > text.size()) continue;
        auto matches = true;
        for (std::size_t i = 0; i < contraction.size(); ++i) {
            const auto actual = static_cast<unsigned char>(text[offset + i]);
            const auto expected = static_cast<unsigned char>(contraction[i]);
            if (std::tolower(actual) != std::tolower(expected)) {
                matches = false;
                break;
            }
        }
        if (matches) return contraction.size();
    }
    return 0;
}

std::vector<std::string_view> PreTokenize(std::string_view text) {
    std::vector<std::string_view> pieces;
    for (std::size_t start = 0; start < text.size();) {
        auto end = start;
        if (const auto contraction = ContractionLength(text, start); contraction != 0) {
            end += contraction;
        } else {
            const auto current = static_cast<unsigned char>(text[end]);
            const auto next_is_letter = end + 1 < text.size() &&
                IsLetter(static_cast<unsigned char>(text[end + 1]));
            if (IsLetter(current) || (!IsNewline(current) && !IsLetter(current) &&
                                      !IsNumber(current) && next_is_letter)) {
                if (!IsLetter(current)) ++end;
                while (end < text.size() &&
                       IsLetter(static_cast<unsigned char>(text[end]))) ++end;
            } else if (IsNumber(current)) {
                ++end;
            } else {
                if (current == ' ' && end + 1 < text.size() &&
                    IsPunctuation(static_cast<unsigned char>(text[end + 1]))) ++end;
                if (end < text.size() &&
                    IsPunctuation(static_cast<unsigned char>(text[end]))) {
                    while (end < text.size() &&
                           IsPunctuation(static_cast<unsigned char>(text[end]))) ++end;
                    while (end < text.size() &&
                           IsNewline(static_cast<unsigned char>(text[end]))) ++end;
                } else {
                    auto whitespace_end = end;
                    auto last_newline = std::string_view::npos;
                    while (whitespace_end < text.size() &&
                           IsSpace(static_cast<unsigned char>(text[whitespace_end]))) {
                        if (IsNewline(static_cast<unsigned char>(text[whitespace_end]))) {
                            last_newline = whitespace_end;
                        }
                        ++whitespace_end;
                    }
                    if (last_newline != std::string_view::npos) {
                        end = last_newline + 1;
                    } else if (whitespace_end < text.size() && whitespace_end - end > 1) {
                        end = whitespace_end - 1;
                    } else {
                        end = whitespace_end;
                    }
                }
            }
        }
        if (end == start) ++end;
        pieces.push_back(text.substr(start, end - start));
        start = end;
    }
    return pieces;
}

}  // namespace

Tokenizer::Tokenizer(const Model& model)
    : Tokenizer(LoadVocabulary(model), LoadMerges(model)) {
    if (VocabSize() != static_cast<std::size_t>(model.GetConfig()->vocab_size)) {
        throw std::runtime_error("Tokenizer vocabulary size does not match model config");
    }
}

Tokenizer::Tokenizer(std::vector<std::string> vocabulary, std::vector<Merge> merges)
    : vocabulary_(std::move(vocabulary)) {
    Initialize(std::move(merges));
}

std::size_t Tokenizer::PairHash::operator()(const Merge& pair) const noexcept {
    const auto left = std::hash<std::string>{}(pair.first);
    const auto right = std::hash<std::string>{}(pair.second);
    return left ^ (right + 0x9e3779b97f4a7c15ULL + (left << 6) + (left >> 2));
}

void Tokenizer::Initialize(std::vector<Merge> merges) {
    for (std::size_t id = 0; id < vocabulary_.size(); ++id) {
        if (!vocabulary_[id].empty()) {
            token_ids_.try_emplace(vocabulary_[id], static_cast<std::int32_t>(id));
        }
        if (vocabulary_[id].starts_with("<|") && vocabulary_[id].ends_with("|>")) {
            special_tokens_.emplace_back(vocabulary_[id], static_cast<std::int32_t>(id));
        }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(), [](const auto& a, const auto& b) {
        return a.first.size() > b.first.size();
    });
    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        merge_ranks_.try_emplace(std::move(merges[rank]), rank);
    }
}

void Tokenizer::EncodePiece(std::string_view piece, std::vector<std::int32_t>& out) const {
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (const char byte : piece) symbols.emplace_back(1, byte);

    while (symbols.size() > 1) {
        auto best_rank = std::numeric_limits<std::size_t>::max();
        Merge best_pair;
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto rank = merge_ranks_.find({symbols[i], symbols[i + 1]});
            if (rank != merge_ranks_.end() && rank->second < best_rank) {
                best_rank = rank->second;
                best_pair = rank->first;
            }
        }
        if (best_rank == std::numeric_limits<std::size_t>::max()) break;

        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (std::size_t i = 0; i < symbols.size();) {
            if (i + 1 < symbols.size() && symbols[i] == best_pair.first &&
                symbols[i + 1] == best_pair.second) {
                merged.push_back(symbols[i] + symbols[i + 1]);
                i += 2;
            } else {
                merged.push_back(std::move(symbols[i++]));
            }
        }
        symbols = std::move(merged);
    }

    for (const auto& symbol : symbols) {
        const auto token = token_ids_.find(symbol);
        if (token == token_ids_.end()) {
            throw std::runtime_error("BPE produced a symbol absent from the vocabulary");
        }
        out.push_back(token->second);
    }
}

void Tokenizer::EncodeOrdinary(std::string_view text, std::vector<std::int32_t>& out) const {
    for (const auto piece : PreTokenize(text)) EncodePiece(piece, out);
}

std::vector<std::int32_t> Tokenizer::Encode(std::string_view text) const {
    std::vector<std::int32_t> result;
    for (std::size_t offset = 0; offset < text.size();) {
        auto special_offset = std::string_view::npos;
        const std::pair<std::string, std::int32_t>* special = nullptr;
        for (const auto& candidate : special_tokens_) {
            const auto found = text.find(candidate.first, offset);
            if (found < special_offset) {
                special_offset = found;
                special = &candidate;
            }
        }
        if (special == nullptr) {
            EncodeOrdinary(text.substr(offset), result);
            break;
        }
        EncodeOrdinary(text.substr(offset, special_offset - offset), result);
        result.push_back(special->second);
        offset = special_offset + special->first.size();
    }
    return result;
}

std::string Tokenizer::Decode(std::span<const std::int32_t> tokens) const {
    std::string result;
    for (const auto id : tokens) result.append(Token(id));
    return result;
}

std::string_view Tokenizer::Token(std::int32_t id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= vocabulary_.size()) {
        throw std::out_of_range("Token ID is outside the vocabulary");
    }
    return vocabulary_[id];
}

std::size_t Tokenizer::VocabSize() const noexcept {
    return vocabulary_.size();
}

std::string FormatChatPrompt(std::string_view prompt) {
    return "<|im_start|>user\n" + std::string(prompt) +
           "<|im_end|>\n<|im_start|>assistant\n";
}
