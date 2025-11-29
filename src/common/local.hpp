#ifndef Nott_COMMON_LOCAL_HPP
#define Nott_COMMON_LOCAL_HPP

#include <optional>
#include <vector>

#include "../loss/loss.hpp"
#include "../optimizer/optimizer.hpp"
#include "../regularization/regularization.hpp"

namespace Nott {
    struct LocalConfig {
        std::optional<::Nott::Optimizer::Descriptor> optimizer{};
        std::optional<::Nott::Loss::Descriptor> loss{};
        std::vector<::Nott::Regularization::Descriptor> regularization{};
    };
}

#endif // Nott_COMMON_LOCAL_HPP
