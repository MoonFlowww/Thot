#ifndef THOT_INITIALIZATION_HPP
#define THOT_INITIALIZATION_HPP
// This file is an factory, must exempt it from any logical-code. For functions look into "/details"
#include <torch/torch.h>
#include <utility>

namespace thot::initialization {
    /**
     * Default Xavier uniform initialization policy for linear layers.
     */
    struct Xavier {
        void operator()(torch::nn::Linear& layer) const {
            torch::nn::init::xavier_uniform_(layer->weight);
            if (layer->options.with_bias() && layer->bias.defined()) {
                torch::nn::init::zeros_(layer->bias);
            }
        }
    };

    /**
     * Convenience helper that applies a policy to a module.
     */
    template <class Policy, class Module>
    void apply(Policy&& policy, Module& module) {
        std::forward<Policy>(policy)(module);
    }
}

#endif //THOT_INITIALIZATION_HPP