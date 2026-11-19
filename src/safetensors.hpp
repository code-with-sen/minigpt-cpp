// Reader for the .safetensors format used by HuggingFace:
//   [8 bytes: little-endian u64 header size N][N bytes: JSON header][raw tensor data]
// The header maps each tensor name to {dtype, shape, data_offsets} within the data section.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

namespace minigpt {

struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;  // byte offsets into the data section

    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

class SafeTensors {
public:
    explicit SafeTensors(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        uint64_t header_len = 0;
        unsigned char len_bytes[8];
        if (!f.read(reinterpret_cast<char*>(len_bytes), 8)) throw std::runtime_error("truncated file: " + path);
        for (int i = 7; i >= 0; --i) header_len = (header_len << 8) | len_bytes[i];
        if (header_len > (100u << 20)) throw std::runtime_error("implausible safetensors header in " + path);

        std::string header(header_len, '\0');
        if (!f.read(header.data(), static_cast<std::streamsize>(header_len)))
            throw std::runtime_error("truncated header: " + path);

        json::Value root = json::parse(header);
        for (const auto& [name, v] : root.object) {
            if (name == "__metadata__") continue;
            TensorInfo t;
            t.dtype = v.at("dtype").string;
            for (const auto& d : v.at("shape").array) t.shape.push_back(static_cast<int64_t>(d.number));
            t.begin = static_cast<uint64_t>(v.at("data_offsets").array.at(0).number);
            t.end = static_cast<uint64_t>(v.at("data_offsets").array.at(1).number);
            tensors_.emplace(name, std::move(t));
        }

        f.seekg(0, std::ios::end);
        uint64_t data_len = static_cast<uint64_t>(f.tellg()) - 8 - header_len;
        data_.resize(data_len);
        f.seekg(static_cast<std::streamoff>(8 + header_len));
        if (!f.read(reinterpret_cast<char*>(data_.data()), static_cast<std::streamsize>(data_len)))
            throw std::runtime_error("truncated tensor data: " + path);
    }

    bool has(const std::string& name) const { return tensors_.count(name) > 0; }

    const TensorInfo& info(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
        return it->second;
    }

    // Copies a tensor out as fp32 (converting from F16/BF16 if needed).
    std::vector<float> read_f32(const std::string& name) const {
        const TensorInfo& t = info(name);
        const uint8_t* src = data_.data() + t.begin;
        size_t n = static_cast<size_t>(t.numel());
        std::vector<float> out(n);
        if (t.end > data_.size() || t.end < t.begin) throw std::runtime_error("bad offsets for " + name);
        if (t.dtype == "F32") {
            if (t.end - t.begin != n * 4) throw std::runtime_error("size mismatch for " + name);
            std::memcpy(out.data(), src, n * 4);
        } else if (t.dtype == "F16" || t.dtype == "BF16") {
            if (t.end - t.begin != n * 2) throw std::runtime_error("size mismatch for " + name);
            for (size_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, src + 2 * i, 2);
                out[i] = t.dtype == "F16" ? half_to_float(h) : bf16_to_float(h);
            }
        } else {
            throw std::runtime_error("unsupported dtype " + t.dtype + " for " + name);
        }
        return out;
    }

    const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }

private:
    std::map<std::string, TensorInfo> tensors_;
    std::vector<uint8_t> data_;

    static float bf16_to_float(uint16_t h) {
        uint32_t bits = static_cast<uint32_t>(h) << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    }

    static float half_to_float(uint16_t h) {
        uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign;
            } else {  // subnormal: normalize it
                exp = 127 - 15 + 1;
                while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
                bits = sign | (exp << 23) | ((mant & 0x3FF) << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7F800000u | (mant << 13);
        } else {
            bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    }
};

}  // namespace minigpt
