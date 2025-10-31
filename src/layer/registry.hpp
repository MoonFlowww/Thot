#ifndef THOT_LAYER_REGISTRY_HPP
#define THOT_LAYER_REGISTRY_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <variant>
#include <type_traits>
#include <utility>
#include <stdexcept>


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
        struct ForwardBinding {
            using Invoker = torch::Tensor (*)(void*, torch::Tensor);

            Invoker invoke{nullptr};
            void* context{nullptr};

            [[nodiscard]] explicit operator bool() const noexcept { return invoke != nullptr; }

            torch::Tensor operator()(torch::Tensor input) const
            {
                if (!invoke) {
                    throw std::logic_error("Attempted to invoke an empty forward binding.");
                }
                return invoke(context, std::move(input));
            }
        };

        template <class Module>
        void bind_module_forward(Module* module)
        {
            forward = ForwardBinding{&dispatch_module<Module>, module};
            forward_context_inline_ = false;
            forward_context_size_ = 0;
        }

        template <class Functor>
        void bind_inline_forward(Functor functor)
        {
            static_assert(std::is_trivially_copyable_v<Functor>,
                          "Forward functor contexts must be trivially copyable.");
            static_assert(sizeof(Functor) <= kForwardContextInlineSize,
                          "Forward functor context exceeds inline storage capacity.");

            forward_context_inline_ = true;
            forward_context_size_ = sizeof(Functor);
            std::memcpy(forward_context_storage_, std::addressof(functor), sizeof(Functor));
            forward = ForwardBinding{&dispatch_functor<Functor>, forward_context_storage_};
        }

        ForwardBinding forward{};
        ::Thot::Activation::Type activation{::Thot::Activation::Type::Identity};
        std::shared_ptr<torch::nn::Module> module{};
        ::Thot::LocalConfig local{};
        std::string name{};
        RegisteredLayer() = default;

        RegisteredLayer(const RegisteredLayer& other)
            : forward(other.forward)
            , activation(other.activation)
            , module(other.module)
            , local(other.local)
            , name(other.name)
            , forward_context_size_(other.forward_context_size_)
            , forward_context_inline_(other.forward_context_inline_)
        {
            copy_inline_context(other);
        }

        RegisteredLayer(RegisteredLayer&& other) noexcept
            : forward(other.forward)
            , activation(other.activation)
            , module(std::move(other.module))
            , local(std::move(other.local))
            , name(std::move(other.name))
            , forward_context_size_(other.forward_context_size_)
            , forward_context_inline_(other.forward_context_inline_)
        {
            copy_inline_context(other);
            other.reset_move_source();
        }

        RegisteredLayer& operator=(const RegisteredLayer& other)
        {
            if (this == &other) {
                return *this;
            }

            forward = other.forward;
            activation = other.activation;
            module = other.module;
            local = other.local;
            name = other.name;
            forward_context_size_ = other.forward_context_size_;
            forward_context_inline_ = other.forward_context_inline_;
            copy_inline_context(other);
            return *this;
        }

        RegisteredLayer& operator=(RegisteredLayer&& other) noexcept
        {
            if (this == &other) {
                return *this;
            }

            forward = other.forward;
            activation = other.activation;
            module = std::move(other.module);
            local = std::move(other.local);
            name = std::move(other.name);
            forward_context_size_ = other.forward_context_size_;
            forward_context_inline_ = other.forward_context_inline_;
            copy_inline_context(other);
            other.reset_move_source();
            return *this;
        }

    private:
        static constexpr std::size_t kForwardContextInlineSize = sizeof(void*) * 3;

        template <class Module>
        static torch::Tensor dispatch_module(void* context, torch::Tensor input)
        {
            auto* module = static_cast<Module*>(context);
            return module->forward(std::move(input));
        }

        template <class Functor>
        static torch::Tensor dispatch_functor(void* context, torch::Tensor input)
        {
            auto* functor = static_cast<Functor*>(context);
            return (*functor)(std::move(input));
        }

        void copy_inline_context(const RegisteredLayer& other)
        {
            if (!forward_context_inline_) {
                return;
            }

            std::memcpy(
                forward_context_storage_,
                other.forward_context_storage_,
                kForwardContextInlineSize);
            forward.context = forward_context_storage_;
        }

        void reset_move_source() noexcept
        {
            if (forward_context_inline_) {
                std::memset(forward_context_storage_, 0, kForwardContextInlineSize);
            }
            forward_context_size_ = 0;
            forward_context_inline_ = false;
            forward.invoke = nullptr;
            forward.context = nullptr;
        }

        alignas(std::max_align_t) std::byte forward_context_storage_[kForwardContextInlineSize]{};
        std::uint8_t forward_context_size_{0};
        bool forward_context_inline_{false};
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