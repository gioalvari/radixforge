#pragma once

#include <map>
#include <memory>
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

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }

    bool as_bool() const { return bool_; }
    double as_number() const { return number_; }
    const std::string& as_string() const { return string_; }
    const Array& as_array() const { return array_; }
    const Object& as_object() const { return object_; }

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
        if (type_ == Type::Array) return array_.size();
        if (type_ == Type::Object) return object_.size();
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
        auto val = p.parse_value();
        return val;
    }

private:
    explicit Parser(const std::string& input) : src_(input), pos_(0) {}

    JsonValue parse_value() {
        skip_ws();
        if (pos_ >= src_.size()) return JsonValue();
        char c = src_[pos_];
        if (c == '"') return parse_string_value();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_number();
    }

    JsonValue parse_string_value() {
        return JsonValue(parse_string());
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        while (pos_ < src_.size() && src_[pos_] != '"') {
            if (src_[pos_] == '\\') {
                pos_++;
                if (pos_ >= src_.size()) break;
                char esc = src_[pos_++];
                switch (esc) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case 'u': {
                        // Parse 4 hex digits
                        if (pos_ + 4 > src_.size()) break;
                        uint32_t cp = 0;
                        for (int i = 0; i < 4; i++) {
                            cp <<= 4;
                            char h = src_[pos_++];
                            if (h >= '0' && h <= '9') cp |= (h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                        }
                        // Simple UTF-8 encode for BMP
                        if (cp < 0x80) {
                            result += (char)cp;
                        } else if (cp < 0x800) {
                            result += (char)(0xC0 | (cp >> 6));
                            result += (char)(0x80 | (cp & 0x3F));
                        } else {
                            result += (char)(0xE0 | (cp >> 12));
                            result += (char)(0x80 | ((cp >> 6) & 0x3F));
                            result += (char)(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: result += esc; break;
                }
            } else {
                result += src_[pos_++];
            }
        }
        if (pos_ < src_.size()) pos_++; // skip closing quote
        return result;
    }

    JsonValue parse_object() {
        expect('{');
        JsonValue::Object obj;
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == '}') { pos_++; return JsonValue(std::move(obj)); }
        while (pos_ < src_.size()) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            JsonValue val = parse_value();
            obj.emplace(std::move(key), std::move(val));
            skip_ws();
            if (pos_ < src_.size() && src_[pos_] == ',') { pos_++; continue; }
            break;
        }
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == '}') pos_++;
        return JsonValue(std::move(obj));
    }

    JsonValue parse_array() {
        expect('[');
        JsonValue::Array arr;
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == ']') { pos_++; return JsonValue(std::move(arr)); }
        while (pos_ < src_.size()) {
            arr.push_back(parse_value());
            skip_ws();
            if (pos_ < src_.size() && src_[pos_] == ',') { pos_++; continue; }
            break;
        }
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == ']') pos_++;
        return JsonValue(std::move(arr));
    }

    JsonValue parse_number() {
        size_t start = pos_;
        if (src_[pos_] == '-') pos_++;
        while (pos_ < src_.size() && (src_[pos_] >= '0' && src_[pos_] <= '9')) pos_++;
        if (pos_ < src_.size() && src_[pos_] == '.') {
            pos_++;
            while (pos_ < src_.size() && (src_[pos_] >= '0' && src_[pos_] <= '9')) pos_++;
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            pos_++;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) pos_++;
            while (pos_ < src_.size() && (src_[pos_] >= '0' && src_[pos_] <= '9')) pos_++;
        }
        return JsonValue(std::stod(src_.substr(start, pos_ - start)));
    }

    JsonValue parse_bool() {
        if (src_.compare(pos_, 4, "true") == 0) { pos_ += 4; return JsonValue(true); }
        if (src_.compare(pos_, 5, "false") == 0) { pos_ += 5; return JsonValue(false); }
        return JsonValue();
    }

    JsonValue parse_null() {
        if (src_.compare(pos_, 4, "null") == 0) { pos_ += 4; }
        return JsonValue();
    }

    void skip_ws() {
        while (pos_ < src_.size() && (src_[pos_] == ' ' || src_[pos_] == '\t' ||
               src_[pos_] == '\n' || src_[pos_] == '\r')) pos_++;
    }

    void expect(char c) {
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == c) pos_++;
    }

    const std::string& src_;
    size_t pos_;
};

inline JsonValue parse(const std::string& input) {
    return Parser::parse(input);
}

} // namespace json
} // namespace radixforge
