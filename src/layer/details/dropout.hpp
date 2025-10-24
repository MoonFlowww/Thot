#ifndef THOT_DROPOUT_HPP
#define THOT_DROPOUT_HPP
#include "../../activation/activation.hpp"
#include "../../common/local.hpp"

namespace Thot::Layer::Details {

    struct DropoutOptions {
        double probability{0.5};
        bool inplace{false};
    };

    struct DropoutDescriptor {
        DropoutOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
        ::Thot::LocalConfig local{};
    };

}

#endif //THOT_DROPOUT_HPP