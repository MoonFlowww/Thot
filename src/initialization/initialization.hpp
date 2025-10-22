#ifndef THOT_INITIALIZATION_HPP
#define THOT_INITIALIZATION_HPP
// This file is a factory, must exempt it from any logical-code. For functions look into "/details"

namespace Thot::Initialization {
    enum class Type {
        Default,
        XavierNormal,
    };

    struct Descriptor {
        Type type{Type::Default};
    };

    inline constexpr Descriptor Default{Type::Default};
    inline constexpr Descriptor XavierNormal{Type::XavierNormal};
}

#endif //THOT_INITIALIZATION_HPP
