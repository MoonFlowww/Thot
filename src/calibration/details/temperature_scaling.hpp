#ifndef Nott_TEMPERATURE_SCALING_HPP
#define Nott_TEMPERATURE_SCALING_HPP
#include <atomic>
#include <memory>
#include <stdexcept>
#include <utility>

namespace Details {
    class TemperatureScalingModuleImpl : public torch::nn::Module {
    public:
        TemperatureScalingModuleImpl()
        {
            log_temperature_ = register_parameter("log_temperature", torch::zeros({1}));
        }

        torch::Tensor forward(torch::Tensor logits) const
        {
            const auto temperature = torch::exp(log_temperature_);
            return logits / temperature;
        }

        [[nodiscard]] torch::Tensor temperature() const
        {
            return torch::exp(log_temperature_);
        }

    private:
        torch::Tensor log_temperature_;
    };

    TORCH_MODULE(TemperatureScalingModule);

    class TemperatureScalingMethod final : public Method {
    public:
        explicit TemperatureScalingMethod(TemperatureScalingDescriptor descriptor)
            : descriptor_(std::move(descriptor))
        {
            module_ = TemperatureScalingModule();
        }

        void attach(torch::nn::Module& model, const torch::Device& device) override
        {
            if (!module_) {
                module_ = TemperatureScalingModule();
            }

            module_->to(device);

            const auto identifier = next_identifier();
            const auto name = "temperature_scaling_" + std::to_string(identifier);
            model.register_module(name, module_);
            attached_ = true;
        }

        void fit(const torch::Tensor& logits, const torch::Tensor& targets) override
        {
            if (!attached_) {
                throw std::logic_error("Temperature scaling method must be attached before fitting.");
            }

            auto calibration_logits = logits.detach();
            auto calibration_targets = targets.detach();

            if (!calibration_logits.defined() || !calibration_targets.defined()) {
                throw std::invalid_argument("Temperature scaling requires defined logits and targets tensors.");
            }

            auto parameters = module_->parameters();
            if (parameters.empty()) {
                throw std::logic_error("Temperature scaling module does not expose optimisable parameters.");
            }

            calibration_logits = calibration_logits.to(parameters.front().device());
            if (calibration_targets.scalar_type() != torch::kLong) {
                calibration_targets = calibration_targets.to(torch::kLong);
            }
            calibration_targets = calibration_targets.to(calibration_logits.device());

            auto options = torch::optim::LBFGSOptions(descriptor_.learning_rate);
            options.max_iter(descriptor_.max_iterations);
            options.tolerance_grad(1e-7);
            options.tolerance_change(1e-9);

            torch::optim::LBFGS optimizer(parameters, options);
            auto loss_fn = torch::nn::CrossEntropyLoss();

            auto closure = [&]() -> torch::Tensor {
                optimizer.zero_grad();
                auto scaled_logits = module_->forward(calibration_logits);
                auto loss = loss_fn(scaled_logits, calibration_targets);
                loss.backward();
                return loss;
            };

            optimizer.step(closure);
            module_->eval();
        }

        void plot(std::ostream& stream) const override
        {
            if (!module_) {
                stream << "Temperature scaling (uninitialised)";
                return;
            }
            stream << "Temperature scaling (T="
                   << module_->temperature().item<double>()
                   << ")";
        }

        [[nodiscard]] torch::Tensor transform(torch::Tensor logits) const override
        {
            if (!module_) {
                throw std::logic_error("Temperature scaling module is not available for transformation.");
            }
            return module_->forward(std::move(logits));
        }

    private:
        static std::size_t next_identifier()
        {
            static std::atomic<std::size_t> counter{0};
            return counter.fetch_add(1, std::memory_order_relaxed);
        }

        TemperatureScalingDescriptor descriptor_{};
        TemperatureScalingModule module_{};
        bool attached_{false};
    };
    inline MethodPtr make_temperature_scaling_method(TemperatureScalingDescriptor descriptor)
    {
        return std::make_shared<TemperatureScalingMethod>(std::move(descriptor));
    }

}
#endif //Nott_TEMPERATURE_SCALING_HPP