#ifndef THOT_LOSS_HPP
#define THOT_LOSS_HPP
// This file is a factory, must exempt it from any logical-code. For functions look into "/details"
#include <variant>



#include "details/reduction.hpp"
#include "details/ce.hpp"
#include "details/mse.hpp"
#include "details/bce.hpp"
#include "details/nll.hpp"
#include "details/mae.hpp"
#include "details/smooth_l1.hpp"
#include "details/kl.hpp"
#include "details/cosine_embedding.hpp"
#include "details/margin_ranking.hpp"

namespace Thot::Loss {
    using Reduction = Details::Reduction;


    using MSEOptions = Details::MSEOptions;
    using MSEDescriptor = Details::MSEDescriptor;

    using CrossEntropyOptions = Details::CrossEntropyOptions;
    using CrossEntropyDescriptor = Details::CrossEntropyDescriptor;

    using BCEWithLogitsOptions = Details::BCEWithLogitsOptions;
    using BCEWithLogitsDescriptor = Details::BCEWithLogitsDescriptor;

    using NegativeLogLikelihoodOptions = Details::NegativeLogLikelihoodOptions;
    using NegativeLogLikelihoodDescriptor = Details::NegativeLogLikelihoodDescriptor;

    using MAEOptions = Details::MAEOptions;
    using MAEDescriptor = Details::MAEDescriptor;

    using SmoothL1Options = Details::SmoothL1Options;
    using SmoothL1Descriptor = Details::SmoothL1Descriptor;

    using KLDivOptions = Details::KLDivOptions;
    using KLDivDescriptor = Details::KLDivDescriptor;

    using CosineEmbeddingOptions = Details::CosineEmbeddingOptions;
    using CosineEmbeddingDescriptor = Details::CosineEmbeddingDescriptor;

    using MarginRankingOptions = Details::MarginRankingOptions;
    using MarginRankingDescriptor = Details::MarginRankingDescriptor;



    using Descriptor = std::variant<
        MSEDescriptor,
        CrossEntropyDescriptor,
        BCEWithLogitsDescriptor,
        //CosineEmbeddingDescriptor,
        //KLDivDescriptor,
        MAEDescriptor,
        //MarginRankingDescriptor,
        NegativeLogLikelihoodDescriptor,
        SmoothL1Descriptor>;


    [[nodiscard]] constexpr auto MSE(const MSEOptions& options = {}) noexcept -> MSEDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto CrossEntropy(const CrossEntropyOptions& options = {}) noexcept -> CrossEntropyDescriptor {
        return {options};
    }


    [[nodiscard]] constexpr auto BCEWithLogits(const BCEWithLogitsOptions& options = {}) noexcept -> BCEWithLogitsDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto NegativeLogLikelihood(const NegativeLogLikelihoodOptions& options = {}) noexcept -> NegativeLogLikelihoodDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto MAE(const MAEOptions& options = {}) noexcept -> MAEDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto SmoothL1(const SmoothL1Options& options = {}) noexcept -> SmoothL1Descriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto KLDiv(const KLDivOptions& options = {}) noexcept -> KLDivDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto CosineEmbedding(const CosineEmbeddingOptions& options = {}) noexcept -> CosineEmbeddingDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto MarginRanking(const MarginRankingOptions& options = {}) noexcept -> MarginRankingDescriptor {
        return {options};
    }
}

#endif //THOT_LOSS_HPP
