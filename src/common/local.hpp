#ifndef THOT_COMMON_LOCAL_HPP
#define THOT_COMMON_LOCAL_HPP

#include <optional>

#include "../optimizer/optimizer.hpp"

namespace Thot {
    struct LocalConfig {
        std::optional<::Thot::Optimizer::Descriptor> optimizer{};
    };
}

#endif // THOT_COMMON_LOCAL_HPP
