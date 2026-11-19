// Minimal JSON reader: just enough for safetensors headers and GPT-2's vocab.json.
// Supports objects, arrays, strings (with \uXXXX escapes and surrogate pairs), numbers,
// true/false/null. Throws std::runtime_error on malformed input.
#pragma once

#include <cctype>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace minigpt::json {

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;

    const Value& at(const std::string& key) const {
        auto it = object.find(key);
        if (type != Type::Object || it == object.end()) throw std::runtime_error("json: missing key '" + key + "'");
        return it->second;
    }
    bool has(const std::string& key) const { return type == Type::Object && object.count(key) > 0; }
};

class Parser {
public:
    explicit Parser(std::string_view text) : s_(text) {}

    Value parse() {
        Value v = value();
        skip_ws();
        if (i_ != s_.size()) fail("trailing characters");
        return v;
    }

    // Streaming helper for large flat objects such as vocab.json ({"token": id, ...}):
    // calls fn(key, number) for each entry without building a tree.
    template <typename Fn>
    void for_each_number_in_object(Fn&& fn) {
        skip_ws();
        expect('{');
        skip_ws();
        if (peek() == '}') { ++i_; return; }
        while (true) {
            skip_ws();
            std::string key = string_literal();
            skip_ws();
            expect(':');
            skip_ws();
            double n = number();
            fn(key, n);
            skip_ws();
            if (peek() == ',') { ++i_; continue; }
            expect('}');
            break;
        }
    }

private:
    std::string_view s_;
    size_t i_ = 0;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("json: " + msg + " at offset " + std::to_string(i_));
    }
    char peek() const { return i_ < s_.size() ? s_[i_] : '\0'; }
    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++i_;
    }
    void skip_ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
    }

    Value value() {
        skip_ws();
        Value v;
        char c = peek();
        if (c == '{') {
            v.type = Value::Type::Object;
            ++i_;
            skip_ws();
            if (peek() == '}') { ++i_; return v; }
            while (true) {
                skip_ws();
                std::string key = string_literal();
                skip_ws();
                expect(':');
                v.object.emplace(std::move(key), value());
                skip_ws();
                if (peek() == ',') { ++i_; continue; }
                expect('}');
                break;
            }
        } else if (c == '[') {
            v.type = Value::Type::Array;
            ++i_;
            skip_ws();
            if (peek() == ']') { ++i_; return v; }
            while (true) {
                v.array.push_back(value());
                skip_ws();
                if (peek() == ',') { ++i_; continue; }
                expect(']');
                break;
            }
        } else if (c == '"') {
            v.type = Value::Type::String;
            v.string = string_literal();
        } else if (s_.substr(i_, 4) == "true") {
            v.type = Value::Type::Bool; v.boolean = true; i_ += 4;
        } else if (s_.substr(i_, 5) == "false") {
            v.type = Value::Type::Bool; i_ += 5;
        } else if (s_.substr(i_, 4) == "null") {
            i_ += 4;
        } else {
            v.type = Value::Type::Number;
            v.number = number();
        }
        return v;
    }

    double number() {
        size_t start = i_;
        if (peek() == '-') ++i_;
        while (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) || s_[i_] == '.' ||
                                  s_[i_] == 'e' || s_[i_] == 'E' || s_[i_] == '+' || s_[i_] == '-'))
            ++i_;
        if (start == i_) fail("expected a value");
        return std::stod(std::string(s_.substr(start, i_ - start)));
    }

    static void append_utf8(std::string& out, uint32_t cp) {
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
    }

    uint32_t hex4() {
        if (i_ + 4 > s_.size()) fail("truncated \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s_[i_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad hex digit");
        }
        return v;
    }

    std::string string_literal() {
        expect('"');
        std::string out;
        while (true) {
            if (i_ >= s_.size()) fail("unterminated string");
            char c = s_[i_++];
            if (c == '"') break;
            if (c != '\\') { out += c; continue; }
            char e = s_[i_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF && s_.substr(i_, 2) == "\\u") {
                        i_ += 2;
                        uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
        return out;
    }
};

inline Value parse(std::string_view text) { return Parser(text).parse(); }

}  // namespace minigpt::json
