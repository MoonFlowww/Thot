#ifndef THOT_COSINE_EMBEDDING_HPP
#define THOT_COSINE_EMBEDDING_HPP

#include <torch/torch.h>

#include "reduction.hpp"

namespace Thot::Loss::Details {
    struct CosineEmbeddingOptions {
        Reduction reduction{Reduction::Mean};
        double margin{0.0};
    };

    struct CosineEmbeddingDescriptor {
        CosineEmbeddingOptions options{};
    };

    inline torch::Tensor compute(const CosineEmbeddingDescriptor& descriptor, const torch::Tensor& input1, const torch::Tensor& input2, const torch::Tensor& target) {
        auto opts = torch::nn::functional::CosineEmbeddingLossFuncOptions{};
        opts = opts.margin(descriptor.options.margin);
        opts = opts.reduction(to_torch_reduction<torch::nn::CosineEmbeddingLossOptions>(descriptor.options.reduction));

        return torch::nn::functional::cosine_embedding_loss(input1, input2, target, opts);
    }

}

#endif // THOT_COSINE_EMBEDDING_HPP
