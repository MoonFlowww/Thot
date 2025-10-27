#ifndef THOT_DROPOUT_HPP
#define THOT_DROPOUT_HPP
#include "../../activation/activation.hpp"
#include "../../common/local.hpp"

namespace Thot::Layer::Details {

    struct HardDropoutOptions {
        double probability{0.5};
        bool inplace{false};
    };

    struct HardDropoutDescriptor {
        HardDropoutOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
        ::Thot::LocalConfig local{};
    };

}

#endif //THOT_DROPOUT_HPP