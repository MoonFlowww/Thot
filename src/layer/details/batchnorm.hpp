#ifndef THOT_BATCHNORM_HPP
#define THOT_BATCHNORM_HPP
#include <cstdint>

#include "../../common/local.hpp"
#include "../../activation/activation.hpp"
#include "../../initialization/initialization.hpp"
#include "../../common/local.hpp"

namespace Thot::Layer::Details {

    struct BatchNorm2dOptions {
        std::int64_t num_features{};
        double eps{1e-5};
        double momentum{0.1};
        bool affine{true};
        bool track_running_stats{true};
    };

    struct BatchNorm2dDescriptor {
        BatchNorm2dOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
        ::Thot::Initialization::Descriptor initialization{::Thot::Initialization::Default};
        ::Thot::LocalConfig local{};
    };

}

#endif //THOT_BATCHNORM_HPP