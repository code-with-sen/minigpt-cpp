// GPT-2's byte-level BPE tokenizer, reimplemented from the original encoder.py.
//
// Encoding has three steps:
//  1. Split text into chunks with GPT-2's pre-tokenizer regex (words, numbers, punctuation,
//     whitespace), implemented by hand below since std::regex has no Unicode classes.
//  2. Map every byte of a chunk to a printable Unicode character (so any byte sequence,
//     including invalid UTF-8, has a token representation).
//  3. Repeatedly merge the adjacent pair with the lowest rank in merges.txt until no pair
//     can be merged, then look each piece up in vocab.json.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace minigpt {

class Tokenizer {
public:
    // vocab_path: vocab.json (a.k.a. encoder.json); merges_path: merges.txt (a.k.a. vocab.bpe)
    Tokenizer(const std::string& vocab_path, const std::string& merges_path);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    std::string decode_token(int id) const;

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    int eot_token() const { return eot_; }

    // Exposed for tests: the pre-tokenizer's split of a string.
    static std::vector<std::string> pretokenize(const std::string& text);

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> merge_rank_;  // "left right" -> rank
    std::string byte_encoder_[256];                     // byte -> UTF-8 of its stand-in character
    std::unordered_map<std::string, unsigned char> byte_decoder_;
    int eot_ = -1;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;

    std::vector<int> bpe(const std::string& chunk) const;
};

}  // namespace minigpt
