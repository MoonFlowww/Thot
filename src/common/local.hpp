#ifndef OMNI_COMMON_LOCAL_HPP
#define OMNI_COMMON_LOCAL_HPP

#include <optional>
#include <vector>

#include "../loss/loss.hpp"
#include "../optimizer/optimizer.hpp"
#include "../regularization/regularization.hpp"

namespace Omni {
    struct LocalConfig {
        std::optional<::Omni::Optimizer::Descriptor> optimizer{};
        std::optional<::Omni::Loss::Descriptor> loss{};
        std::vector<::Omni::Regularization::Descriptor> regularization{};
    };
}

#endif // OMNI_COMMON_LOCAL_HPP
