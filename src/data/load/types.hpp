#ifndef THOT_TYPES_HPP
#define THOT_TYPES_HPP
#include <cstddef>
#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <optional>

namespace Thot::Data::Type {
    enum class ImageRescaleMode {
        None,
        Downscale,
        Upscale
    };

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

    struct ImageParameters {
        bool recursive = false;
        bool grayscale = false;
        bool normalize = true;
        bool channels_first = true;
        bool size_to_max_tile = false;
        std::optional<std::array<int64_t, 2>> size{};
        std::string color_order = "RGB";
        std::optional<std::array<int64_t, 2>> rescale_to{};
        ImageRescaleMode rescale_mode = ImageRescaleMode::None;
    };

    struct PNG {
        std::string directory;
        ImageParameters parameters{};
    };

    struct JPEG {
        std::string directory;
        ImageParameters parameters{};
    };

    struct JPG {
        std::string directory;
        ImageParameters parameters{};
    };

    struct BMP {
        std::string directory;
        ImageParameters parameters{};
    };

    struct TIFF {
        std::string directory;
        ImageParameters parameters{};
    };

    struct PPM {
        std::string directory;
        ImageParameters parameters{};
    };

    struct PGM {
        std::string directory;
        ImageParameters parameters{};
    };

    struct PBM {
        std::string directory;
        ImageParameters parameters{};
    };

    struct GlobalParameters {
        float train_fraction = 0.8f;
        float test_fraction = 0.2f;
        bool shuffle = true;

    };
}
#endif //THOT_TYPES_HPP