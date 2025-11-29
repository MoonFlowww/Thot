#ifndef Nott_DATA_TRANSFORM_DIMENSIONALITY_REDUCTION_HPP
#define Nott_DATA_TRANSFORM_DIMENSIONALITY_REDUCTION_HPP

#include "dimensionality_reduction/common.hpp"
#include "dimensionality_reduction/pca.hpp"
#include "dimensionality_reduction/rpca.hpp"

namespace Nott::Data::Transform {
    using DimensionalityReduction::PCA;
    using DimensionalityReduction::PCAResult;
    using DimensionalityReduction::ProjectPCA;
    using DimensionalityReduction::RPCA;
    namespace DimensionalityReductionDetails = DimensionalityReduction::Details;
}

#endif // Nott_DATA_TRANSFORM_DIMENSIONALITY_REDUCTION_HPP