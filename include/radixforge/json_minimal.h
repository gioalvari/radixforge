#pragma once

#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace radixforge {
namespace json {

enum class Type { Null, Bool, Number, String, Array, Object };

class JsonValue {
  public:
    using Object = std::map<std::string, JsonValue>;
    using Array = std::vector<JsonValue>;

    JsonValue() : type_(Type::Null) {}
    explicit JsonValue(bool v) : type_(Type::Bool), bool_(v) {}
    explicit JsonValue(double v) : type_(Type::Number), number_(v) {}
    explicit JsonValue(const std::string& v) : type_(Type::String), string_(v) {}
    explicit JsonValue(std::string&& v) : type_(Type::String), string_(std::move(v)) {}
    explicit JsonValue(const Array& v) : type_(Type::Array), array_(v) {}
    explicit JsonValue(Array&& v) : type_(Type::Array), array_(std::move(v)) {}
    explicit JsonValue(const Object& v) : type_(Type::Object), object_(v) {}
    explicit JsonValue(Object&& v) : type_(Type::Object), object_(std::move(v)) {}

    Type type() const {
        return type_;
    }
    bool is_null() const {
        return type_ == Type::Null;
    }

    bool as_bool() const {
        return bool_;
    }
    double as_number() const {
        return number_;
    }
    const std::string& as_string() const {
        return string_;
    }
    const Array& as_array() const {
        return array_;
    }
    const Object& as_object() const {
        return object_;
    }

    // Convenience accessors
    const JsonValue& operator[](const std::string& key) const {
        static const JsonValue null_val;
        auto it = object_.find(key);
        return it != object_.end() ? it->second : null_val;
    }

    const JsonValue& operator[](size_t idx) const {
        return array_.at(idx);
    }

    size_t size() const {
        if (type_ == Type::Array)
            return array_.size();
        if (type_ == Type::Object)
            return object_.size();
        return 0;
    }

    bool contains(const std::string& key) const {
        return type_ == Type::Object && object_.count(key) > 0;
    }

  private:
    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    Array array_;
    Object object_;
};

// Recursive descent parser
class Parser {
  public:
    static JsonValue parse(const std::string& input) {
        Parser p(input);
        auto val = p.parse_value(0);
        p.skip_ws();
        if (p.pos_ != input.size())
            throw std::runtime_error("unexpected trailing JSON data");
        return val;
    }

  private:
    explicit Parser(const std::string& input) : src_(input), pos_(0) {}

    static constexpr size_t kMaxDepth = 64;

    JsonValue parse_value(size_t depth) {
        if (depth > kMaxDepth)
            throw std::runtime_error("JSON nesting limit exceeded");
        skip_ws();
        if (pos_ >= src_.size())
            throw std::runtime_error("unexpected end of JSON input");
        char c = src_[pos_];
        if (c == '"')
            return parse_string_value();
        if (c == '{')
            return parse_object(depth + 1);
        if (c == '[')
            return parse_array(depth + 1);
        if (c == 't' || c == 'f')
            return parse_bool();
        if (c == 'n')
            return parse_null();
        if (c == '-' || (c >= '0' && c <= '9'))
            return parse_number();
        throw std::runtime_error("invalid JSON value");
    }

    JsonValue parse_string_value() {
        return JsonValue(parse_string());
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        while (pos_ < src_.size()) {
            const unsigned char c = static_cast<unsigned char>(src_[pos_++]);
            if (c == '"')
                return result;
            if (c < 0x20)
                throw std::runtime_error("unescaped control character in JSON string");
            if (c != '\\') {
                result += static_cast<char>(c);
                continue;
            }

            if (pos_ >= src_.size())
                throw std::runtime_error("unterminated JSON escape sequence");
            switch (src_[pos_++]) {
            case '"':
                result += '"';
                break;
            case '\\':
                result += '\\';
                break;
            case '/':
                result += '/';
                break;
            case 'b':
                result += '\b';
                break;
            case 'f':
                result += '\f';
                break;
            case 'n':
                result += '\n';
                break;
            case 'r':
                result += '\r';
                break;
            case 't':
                result += '\t';
                break;
            case 'u':
                append_unicode_escape(result);
                break;
            default:
                throw std::runtime_error("invalid JSON escape sequence");
            }
        }
        throw std::runtime_error("unterminated JSON string");
    }

    JsonValue parse_object(size_t depth) {
        expect('{');
        JsonValue::Object obj;
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == '}') {
            pos_++;
            return JsonValue(std::move(obj));
        }
        while (true) {
            skip_ws();
            if (pos_ >= src_.size() || src_[pos_] != '"')
                throw std::runtime_error("expected JSON object key");
            std::string key = parse_string();
            expect(':');
            JsonValue val = parse_value(depth);
            obj.emplace(std::move(key), std::move(val));
            skip_ws();
            if (pos_ >= src_.size())
                throw std::runtime_error("unterminated JSON object");
            if (src_[pos_] == '}') {
                ++pos_;
                return JsonValue(std::move(obj));
            }
            if (src_[pos_] == ',') {
                pos_++;
                continue;
            }
            throw std::runtime_error("expected ',' or '}' in JSON object");
        }
    }

    JsonValue parse_array(size_t depth) {
        expect('[');
        JsonValue::Array arr;
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == ']') {
            pos_++;
            return JsonValue(std::move(arr));
        }
        while (true) {
            arr.push_back(parse_value(depth));
            skip_ws();
            if (pos_ >= src_.size())
                throw std::runtime_error("unterminated JSON array");
            if (src_[pos_] == ']') {
                ++pos_;
                return JsonValue(std::move(arr));
            }
            if (src_[pos_] == ',') {
                pos_++;
                continue;
            }
            throw std::runtime_error("expected ',' or ']' in JSON array");
        }
    }

    JsonValue parse_number() {
        size_t start = pos_;
        if (src_[pos_] == '-')
            pos_++;
        if (pos_ >= src_.size())
            throw std::runtime_error("invalid JSON number");
        if (src_[pos_] == '0') {
            pos_++;
            if (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9')
                throw std::runtime_error("leading zero in JSON number");
        } else if (src_[pos_] >= '1' && src_[pos_] <= '9') {
            do {
                pos_++;
            } while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9');
        } else {
            throw std::runtime_error("invalid JSON number");
        }
        if (pos_ < src_.size() && src_[pos_] == '.') {
            pos_++;
            const size_t fraction_start = pos_;
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9')
                pos_++;
            if (pos_ == fraction_start)
                throw std::runtime_error("invalid JSON number fraction");
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            pos_++;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-'))
                pos_++;
            const size_t exponent_start = pos_;
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9')
                pos_++;
            if (pos_ == exponent_start)
                throw std::runtime_error("invalid JSON number exponent");
        }
        const double value = std::stod(src_.substr(start, pos_ - start));
        if (!std::isfinite(value))
            throw std::runtime_error("non-finite JSON number");
        return JsonValue(value);
    }

    JsonValue parse_bool() {
        if (src_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            return JsonValue(true);
        }
        if (src_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            return JsonValue(false);
        }
        throw std::runtime_error("invalid JSON literal");
    }

    JsonValue parse_null() {
        if (src_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            return JsonValue();
        }
        throw std::runtime_error("invalid JSON literal");
    }

    void skip_ws() {
        while (pos_ < src_.size() && (src_[pos_] == ' ' || src_[pos_] == '\t' ||
                                      src_[pos_] == '\n' || src_[pos_] == '\r'))
            pos_++;
    }

    void expect(char c) {
        skip_ws();
        if (pos_ >= src_.size() || src_[pos_] != c)
            throw std::runtime_error("unexpected JSON token");
        ++pos_;
    }

    static int hex_value(char c) {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        if (c >= 'A' && c <= 'F')
            return c - 'A' + 10;
        return -1;
    }

    uint16_t parse_unicode_code_unit() {
        if (pos_ + 4 > src_.size())
            throw std::runtime_error("truncated JSON Unicode escape");
        uint16_t code_unit = 0;
        for (int index = 0; index < 4; ++index) {
            const int value = hex_value(src_[pos_++]);
            if (value < 0)
                throw std::runtime_error("invalid JSON Unicode escape");
            code_unit = static_cast<uint16_t>((code_unit << 4) | value);
        }
        return code_unit;
    }

    void append_unicode_escape(std::string& result) {
        uint32_t code_point = parse_unicode_code_unit();
        if (code_point >= 0xD800 && code_point <= 0xDBFF) {
            if (pos_ + 2 > src_.size() || src_[pos_] != '\\' || src_[pos_ + 1] != 'u')
                throw std::runtime_error("unpaired high surrogate in JSON string");
            pos_ += 2;
            const uint32_t low_surrogate = parse_unicode_code_unit();
            if (low_surrogate < 0xDC00 || low_surrogate > 0xDFFF)
                throw std::runtime_error("invalid low surrogate in JSON string");
            code_point = 0x10000 + ((code_point - 0xD800) << 10) + (low_surrogate - 0xDC00);
        } else if (code_point >= 0xDC00 && code_point <= 0xDFFF) {
            throw std::runtime_error("unpaired low surrogate in JSON string");
        }
        append_utf8(result, code_point);
    }

    static void append_utf8(std::string& result, uint32_t code_point) {
        if (code_point <= 0x7F) {
            result += static_cast<char>(code_point);
        } else if (code_point <= 0x7FF) {
            result += static_cast<char>(0xC0 | (code_point >> 6));
            result += static_cast<char>(0x80 | (code_point & 0x3F));
        } else if (code_point <= 0xFFFF) {
            result += static_cast<char>(0xE0 | (code_point >> 12));
            result += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (code_point & 0x3F));
        } else {
            result += static_cast<char>(0xF0 | (code_point >> 18));
            result += static_cast<char>(0x80 | ((code_point >> 12) & 0x3F));
            result += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (code_point & 0x3F));
        }
    }

    const std::string& src_;
    size_t pos_;
};

inline JsonValue parse(const std::string& input) {
    return Parser::parse(input);
}

} // namespace json
} // namespace radixforge
