#ifndef THOT_ACTIVATION_HPP
#define THOT_ACTIVATION_HPP
// This file is a factory, must exempt it from any logical-code. For functions look into "/details"

namespace Thot::Activation {
    enum class Type {
        Identity,
        ReLU,
    };

    struct Descriptor {
        Type type{Type::Identity};
    };

    inline constexpr Descriptor Identity{Type::Identity};
    inline constexpr Descriptor ReLU{Type::ReLU};
}

#endif //THOT_ACTIVATION_HPP
