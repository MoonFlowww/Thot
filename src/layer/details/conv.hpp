#ifndef THOT_CONV_HPP
#define THOT_CONV_HPP
#include <cstdint>
#include <string>
#include <vector>

#include "../../activation/activation.hpp"
#include "../../initialization/initialization.hpp"

namespace Thot::Layer::Details {

    struct Conv2dOptions {
        std::int64_t in_channels{};
        std::int64_t out_channels{};
        std::vector<std::int64_t> kernel_size{3, 3};
        std::vector<std::int64_t> stride{1, 1};
        std::vector<std::int64_t> padding{0, 0};
        std::vector<std::int64_t> dilation{1, 1};
        std::int64_t groups{1};
        bool bias{true};
        std::string padding_mode{"zeros"};
    };

    struct Conv2dDescriptor {
        Conv2dOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
        ::Thot::Initialization::Descriptor initialization{::Thot::Initialization::Default};
    };

}

#endif //THOT_CONV_HPP