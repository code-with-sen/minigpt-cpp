#include "tokenizer.hpp"

#include <climits>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "json.hpp"

namespace minigpt {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string utf8(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return out;
}

// One decoded code point and where its bytes sit in the original string.
struct CodePoint {
    uint32_t cp;
    size_t pos, len;
};

// Decodes UTF-8 leniently: an invalid byte becomes its own "code point" so no input is lost.
std::vector<CodePoint> decode_utf8(const std::string& s) {
    std::vector<CodePoint> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        uint32_t cp = c;
        bool ok = len > 0 && i + len <= s.size();
        if (ok && len > 1) {
            cp = c & (0xFF >> (len + 1));
            for (size_t k = 1; k < len; ++k) {
                unsigned char cc = static_cast<unsigned char>(s[i + k]);
                if ((cc >> 6) != 0x2) { ok = false; break; }
                cp = (cp << 6) | (cc & 0x3F);
            }
        }
        if (!ok) { cp = 0xFFFD; len = 1; }
        out.push_back({cp, i, len});
        i += len;
    }
    return out;
}

// Character classes used by GPT-2's pattern. ASCII is exact; beyond ASCII we use the Unicode
// blocks that matter in practice (spaces, punctuation and symbol blocks, digits), and treat
// the rest as letters, which covers the scripts of nearly all real text.
bool is_space(uint32_t c) {
    return c == ' ' || (c >= 0x09 && c <= 0x0D) || c == 0x1C || c == 0x1D || c == 0x1E || c == 0x1F ||
           c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
           c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

bool is_number(uint32_t c) {
    return (c >= '0' && c <= '9') || c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE) ||
           (c >= 0x0660 && c <= 0x0669) || (c >= 0x06F0 && c <= 0x06F9) || (c >= 0x0966 && c <= 0x096F) ||
           (c >= 0x2070 && c <= 0x2079) || (c >= 0x2080 && c <= 0x2089) || (c >= 0x2150 && c <= 0x218B) ||
           (c >= 0x2460 && c <= 0x249B) || (c >= 0xFF10 && c <= 0xFF19);
}

bool is_letter(uint32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    if (is_space(c) || is_number(c)) return false;
    if (c == 0xAA || c == 0xB5 || c == 0xBA) return true;   // ª µ º
    if (c < 0xC0) return false;                             // Latin-1 punctuation and symbols
    if (c == 0xD7 || c == 0xF7) return false;               // × ÷
    if (c >= 0x2000 && c <= 0x2BFF) return false;           // punctuation, arrows, math, shapes
    if (c >= 0x3000 && c <= 0x303F) return false;           // CJK punctuation
    if (c >= 0xFE30 && c <= 0xFE6F) return false;           // CJK compatibility forms
    if (c >= 0xFF00 && c <= 0xFF20) return false;           // fullwidth punctuation
    if (c >= 0x1F000 && c <= 0x1FAFF) return false;         // emoji and pictographs
    if (c == 0xFFFD) return false;
    return true;
}

}  // namespace

Tokenizer::Tokenizer(const std::string& vocab_path, const std::string& merges_path) {
    // GPT-2's byte <-> unicode table: printable bytes map to themselves, the rest are shifted
    // to code points 256+ so that every byte has a visible stand-in.
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        uint32_t cp = printable ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
        byte_encoder_[b] = utf8(cp);
        byte_decoder_[byte_encoder_[b]] = static_cast<unsigned char>(b);
    }

    std::string vocab_text = read_file(vocab_path);
    json::Parser parser(vocab_text);
    parser.for_each_number_in_object([&](const std::string& token, double id) {
        int i = static_cast<int>(id);
        token_to_id_[token] = i;
        if (static_cast<int>(id_to_token_.size()) <= i) id_to_token_.resize(i + 1);
        id_to_token_[i] = token;
    });
    auto eot = token_to_id_.find("<|endoftext|>");
    if (eot != token_to_id_.end()) eot_ = eot->second;

    std::istringstream merges(read_file(merges_path));
    std::string line;
    int rank = 0;
    while (std::getline(merges, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.rfind("#version", 0) == 0) continue;
        merge_rank_.emplace(line, rank++);
    }
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) {
    // Hand-written equivalent of GPT-2's pattern:
    //   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    std::vector<CodePoint> cps = decode_utf8(text);
    const size_t n = cps.size();
    std::vector<std::string> out;
    auto emit = [&](size_t a, size_t b) {  // code points [a, b)
        size_t start = cps[a].pos, end = cps[b - 1].pos + cps[b - 1].len;
        out.push_back(text.substr(start, end - start));
    };
    auto is_other = [](uint32_t c) { return !is_space(c) && !is_letter(c) && !is_number(c); };

    size_t i = 0;
    while (i < n) {
        // Contractions: 's 't 're 've 'm 'll 'd (case-sensitive, as in GPT-2).
        if (cps[i].cp == '\'' && i + 1 < n) {
            uint32_t a = cps[i + 1].cp, b = i + 2 < n ? cps[i + 2].cp : 0;
            size_t len = 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') len = 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) len = 3;
            if (len) { emit(i, i + len); i += len; continue; }
        }
        // Optional single space, then a run of letters / numbers / other symbols.
        size_t j = i + (cps[i].cp == ' ' ? 1 : 0);
        if (j < n) {
            bool (*cls)(uint32_t) = nullptr;
            if (is_letter(cps[j].cp)) cls = is_letter;
            else if (is_number(cps[j].cp)) cls = is_number;
            if (cls) {
                size_t k = j;
                while (k < n && cls(cps[k].cp)) ++k;
                emit(i, k); i = k; continue;
            }
            if (is_other(cps[j].cp)) {
                size_t k = j;
                while (k < n && is_other(cps[k].cp)) ++k;
                emit(i, k); i = k; continue;
            }
        }
        // Whitespace: \s+(?!\S) leaves the last space of a run for the next word; \s+ takes the rest.
        size_t k = i;
        while (k < n && is_space(cps[k].cp)) ++k;
        if (k == i) { emit(i, i + 1); ++i; continue; }  // unreachable in practice; keeps progress
        if (k < n && k - i >= 2) k -= 1;
        emit(i, k);
        i = k;
    }
    return out;
}

std::vector<int> Tokenizer::bpe(const std::string& chunk) const {
    auto cached = cache_.find(chunk);
    if (cached != cache_.end()) return cached->second;

    std::vector<std::string> parts;
    for (unsigned char b : chunk) parts.push_back(byte_encoder_[b]);

    while (parts.size() > 1) {
        int best = INT_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            auto it = merge_rank_.find(parts[i] + " " + parts[i + 1]);
            if (it != merge_rank_.end() && it->second < best) { best = it->second; best_i = i; }
        }
        if (best == INT_MAX) break;
        // Merge every occurrence of the winning pair, left to right.
        const std::string left = parts[best_i], right = parts[best_i + 1];
        std::vector<std::string> merged;
        for (size_t i = 0; i < parts.size();) {
            if (i + 1 < parts.size() && parts[i] == left && parts[i + 1] == right) {
                merged.push_back(left + right);
                i += 2;
            } else {
                merged.push_back(parts[i]);
                ++i;
            }
        }
        parts.swap(merged);
    }

    std::vector<int> ids;
    for (const auto& p : parts) {
        auto it = token_to_id_.find(p);
        if (it == token_to_id_.end()) throw std::runtime_error("token missing from vocab: " + p);
        ids.push_back(it->second);
    }
    cache_.emplace(chunk, ids);
    return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    for (const auto& chunk : pretokenize(text)) {
        auto part = bpe(chunk);
        ids.insert(ids.end(), part.begin(), part.end());
    }
    return ids;
}

std::string Tokenizer::decode_token(int id) const {
    if (id < 0 || id >= vocab_size()) throw std::runtime_error("token id out of range");
    const std::string& tok = id_to_token_[id];
    std::string bytes;
    for (const auto& cp : decode_utf8(tok)) {
        auto it = byte_decoder_.find(tok.substr(cp.pos, cp.len));
        if (it != byte_decoder_.end()) bytes += static_cast<char>(it->second);
        else bytes += tok.substr(cp.pos, cp.len);  // special tokens like <|endoftext|>
    }
    return bytes;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) out += decode_token(id);
    return out;
}

}  // namespace minigpt
