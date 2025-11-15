#ifndef THOT_DATA_TRANSFORMS_DIMENSIONALITY_REDUCTION_HPP
#define THOT_DATA_TRANSFORMS_DIMENSIONALITY_REDUCTION_HPP

#include "dimensionality_reduction/common.hpp"
#include "dimensionality_reduction/pca.hpp"
#include "dimensionality_reduction/rpca.hpp"

namespace Thot::Data::Transforms {
    using DimensionalityReduction::PCA;
    using DimensionalityReduction::PCAResult;
    using DimensionalityReduction::ProjectPCA;
    using DimensionalityReduction::RPCA;
    namespace DimensionalityReductionDetails = DimensionalityReduction::Details;
}

#endif // THOT_DATA_TRANSFORMS_DIMENSIONALITY_REDUCTION_HPP