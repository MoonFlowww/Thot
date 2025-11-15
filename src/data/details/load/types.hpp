#ifndef THOT_TYPES_HPP
#define THOT_TYPES_HPP
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace Thot::Data::Type {
    struct CSVParameters {
        bool has_header = true;
        char delimiter = ',';
        std::vector<std::string> columns; // empty = all columns
    };

    struct CSV {
        std::string file;
        CSVParameters parameters{};
    };

    struct TextParameters {
        bool keep_empty_lines = false;
        bool lowercase = false;
        bool trim_whitespace = true;
    };

    struct Text {
        std::string file;
        TextParameters parameters{};
    };

    enum class BinaryDataType {
        Int8,
        UInt8,
        Int16,
        UInt16,
        Int32,
        UInt32,
        Float32,
        Float64
    };

    struct BinaryParameters {
        BinaryDataType type = BinaryDataType::UInt8;
        std::size_t record_size = 0; // elements per sample; 0 => inferred from stream
        bool little_endian = true;
    };

    struct Binary {
        std::string file;
        BinaryParameters parameters{};
    };

    struct GlobalParameters {
        float train_fraction = 0.8f;
        float test_fraction = 0.2f;
        bool shuffle = true;
    };
}
#endif //THOT_TYPES_HPP