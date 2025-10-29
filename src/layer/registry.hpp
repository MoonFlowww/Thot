#ifndef THOT_LAYER_REGISTRY_HPP
#define THOT_LAYER_REGISTRY_HPP

#include <functional>
#include <memory>
#include <string>
#include <variant>



namespace Thot::Layer::Details {
    template <class Impl>
    [[nodiscard]] inline std::shared_ptr<torch::nn::Module>
    to_shared_module_ptr(const torch::nn::ModuleHolder<Impl>& holder)
    {
        static_assert(std::is_base_of_v<torch::nn::Module, Impl>, "ModuleHolder implementation must derive from torch::nn::Module.");
        return std::static_pointer_cast<torch::nn::Module>(holder.ptr());
    }

    template <class Impl>
    [[nodiscard]] inline std::shared_ptr<torch::nn::Module>
    to_shared_module_ptr(const std::shared_ptr<Impl>& pointer)
    {
        static_assert(std::is_base_of_v<torch::nn::Module, Impl>,
                      "Shared pointer implementation must derive from torch::nn::Module.");
        return std::static_pointer_cast<torch::nn::Module>(pointer);
    }

    struct RegisteredLayer {
        std::function<torch::Tensor(torch::Tensor)> forward{};
        ::Thot::Activation::Type activation{::Thot::Activation::Type::Identity};
        std::shared_ptr<torch::nn::Module> module{};
        ::Thot::LocalConfig local{};
        std::string name{};
    };

    template <class Owner, class Descriptor>
    RegisteredLayer build_registered_layer(Owner&, const Descriptor&, std::size_t) {
        static_assert(sizeof(Descriptor) == 0, "Unsupported layer descriptor provided to build_registered_layer.");
        return {};
    }


    template <class Owner, class... DescriptorTypes>
    RegisteredLayer build_registered_layer(Owner& owner, const std::variant<DescriptorTypes...>& descriptor, std::size_t index) {
        return std::visit(
            [&](const auto& concrete_descriptor) {
                return build_registered_layer(owner, concrete_descriptor, index);
            },
            descriptor);
    }



}
#endif // THOT_LAYER_REGISTRY_HPP