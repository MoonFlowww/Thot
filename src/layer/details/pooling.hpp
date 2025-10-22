#ifndef THOT_POOLING_HPP
#define THOT_POOLING_HPP
#include <cstdint>
#include <variant>
#include <vector>

#include "../../activation/activation.hpp"

namespace Thot::Layer::Details {

    struct MaxPool2dOptions {
        std::vector<std::int64_t> kernel_size{2, 2};
        std::vector<std::int64_t> stride{};
        std::vector<std::int64_t> padding{0, 0};
        std::vector<std::int64_t> dilation{1, 1};
        bool ceil_mode{false};
    };

    struct AvgPool2dOptions {
        std::vector<std::int64_t> kernel_size{2, 2};
        std::vector<std::int64_t> stride{};
        std::vector<std::int64_t> padding{0, 0};
        bool ceil_mode{false};
        bool count_include_pad{false};
    };

    struct AdaptiveAvgPool2dOptions {
        std::vector<std::int64_t> output_size{1, 1};
    };

    struct AdaptiveMaxPool2dOptions {
        std::vector<std::int64_t> output_size{1, 1};
    };

    using PoolingOptions = std::variant<MaxPool2dOptions, AvgPool2dOptions, AdaptiveAvgPool2dOptions, AdaptiveMaxPool2dOptions>;

    struct PoolingDescriptor {
        PoolingOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
    };

}

#endif //THOT_POOLING_HPP