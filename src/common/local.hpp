#ifndef THOT_COMMON_LOCAL_HPP
#define THOT_COMMON_LOCAL_HPP

#include <optional>
#include <vector>

#include "../loss/loss.hpp"
#include "../optimizer/optimizer.hpp"
#include "../regularization/regularization.hpp"

namespace Thot {
    struct LocalConfig {
        std::optional<::Thot::Optimizer::Descriptor> optimizer{};
        std::optional<::Thot::Loss::Descriptor> loss{};
        std::vector<::Thot::Regularization::Descriptor> regularization{};
    };
}

#endif // THOT_COMMON_LOCAL_HPP
