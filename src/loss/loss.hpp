#ifndef THOT_LOSS_HPP
#define THOT_LOSS_HPP
// This file is an factory, must exempt it from any logical-code. For functions look into "/details"
#include <cstddef>
#include <type_traits>
#include "loss/details/mse.hpp"

namespace thot::loss {
    template <details::Reduction ReductionMode,
              bool WithWeight = false,
              std::size_t PredictionRank = 0,
              std::size_t TargetRank = 0>
    struct MSE {
        using descriptor_type = details::MSEDescriptor<ReductionMode, WithWeight, PredictionRank, TargetRank>;

        static constexpr details::Reduction reduction = descriptor_type::reduction;
        static constexpr bool uses_weight = descriptor_type::uses_weight;
        static constexpr std::size_t prediction_rank = descriptor_type::prediction_rank;
        static constexpr std::size_t target_rank = descriptor_type::target_rank;

        static constexpr descriptor_type descriptor() { return descriptor_type{}; }
    };
}

#endif //THOT_LOSS_HPP