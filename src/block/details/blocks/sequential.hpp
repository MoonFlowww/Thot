#ifndef OMNI_BLOCK_DETAILS_SEQUENTIAL_HPP
#define OMNI_BLOCK_DETAILS_SEQUENTIAL_HPP

#include <vector>
#include <torch/torch.h>

#include <utility>
#include "../../../activation/apply.hpp"
#include "../../../common/local.hpp"
#include "../../../layer/layer.hpp"

namespace Omni::Block::Details {

    struct SequentialDescriptor {
        std::vector<::Omni::Layer::Descriptor> layers{};
        ::Omni::LocalConfig local{};
    };

    class SequentialBlockModuleImpl : public torch::nn::Module {
    public:
        explicit SequentialBlockModuleImpl(std::vector<::Omni::Layer::Descriptor> layers)
        {
            std::size_t index{0};
            block_layers_.reserve(layers.size());
            for (auto& descriptor : layers) {
                auto registered_layer = ::Omni::Layer::Details::build_registered_layer(*this, descriptor, index++);
                block_layers_.push_back(std::move(registered_layer));
            }
        }

        torch::Tensor forward(torch::Tensor input)
        {
            auto output = std::move(input);
            for (auto& layer : block_layers_) {
                output = layer.forward(std::move(output));
                output = ::Omni::Activation::Details::apply(layer.activation, std::move(output));
            }
            return output;
        }

    private:
        std::vector<::Omni::Layer::Details::RegisteredLayer> block_layers_{};
    };

    TORCH_MODULE(SequentialBlockModule);

}

#endif // OMNI_BLOCK_DETAILS_SEQUENTIAL_HPP