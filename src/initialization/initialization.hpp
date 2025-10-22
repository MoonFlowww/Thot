#ifndef THOT_INITIALIZATION_HPP
#define THOT_INITIALIZATION_HPP
// This file is a factory, must exempt it from any logical-code. For functions look into "/details"

namespace Thot::Initialization {
    enum class Type {
        Default,
        XavierNormal,
        XavierUniform,
        KaimingNormal,
        KaimingUniform,
        ZeroBias,
        Dirac,
        Lyapunov,
    };

    struct Descriptor {
        Type type{Type::Default};
    };

    inline constexpr Descriptor Default{Type::Default};
    inline constexpr Descriptor XavierNormal{Type::XavierNormal};
    inline constexpr Descriptor XavierUniform{Type::XavierUniform};
    inline constexpr Descriptor KaimingNormal{Type::KaimingNormal};
    inline constexpr Descriptor KaimingUniform{Type::KaimingUniform};
    inline constexpr Descriptor ZeroBias{Type::ZeroBias};
    inline constexpr Descriptor Dirac{Type::Dirac};
    inline constexpr Descriptor Lyapunov{Type::Lyapunov};
}

#endif //THOT_INITIALIZATION_HPP
