#ifndef Nott_SOLVER_HPP
#define Nott_SOLVER_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>

#include <torch/torch.h>

namespace Nott::Solver {

    enum class MethodFamily {
        Euler,
        RungeKutta,
        BDF,
        EulerMaruyama,
        Milstein,
        StochasticRungeKutta,
    };

    enum class NoiseModel {
        None,
        Additive,
        Diagonal,
        Full
    };

    struct StepControl {
        double initial_step{1e-3};
        std::optional<double> max_step{};
        bool adaptive{false};
        double safety_factor{0.9};

        [[nodiscard]] double effective_step() const {
            double step = initial_step;
            if (max_step) {
                step = std::min(step, *max_step);
            }
            if (adaptive) {
                step *= safety_factor;
            }
            if (step <= 0.0) {
                throw std::invalid_argument("Step size must be positive.");
            }
            return step;
        }
    };

    struct Descriptor {
        MethodFamily method{MethodFamily::Euler};
        StepControl step_control{};
        NoiseModel noise{NoiseModel::None};

        [[nodiscard]] bool is_sde() const noexcept {
            switch (method) {
                case MethodFamily::EulerMaruyama:
                case MethodFamily::Milstein:
                case MethodFamily::StochasticRungeKutta:
                    return true;
                default:
                    return false;
            }
        }
    };

    using BaseStep = std::function<torch::Tensor(torch::Tensor)>;

    struct Runtime {
        Descriptor descriptor{};

        Runtime() = default;
        explicit Runtime(Descriptor descriptor_) : descriptor(std::move(descriptor_)) {}

        [[nodiscard]] inline torch::Tensor integrate(torch::Tensor state, const BaseStep& base_step) const;
    };

}

#include "details/ode.hpp"
#include "details/sde.hpp"

namespace Nott::Solver {
    inline torch::Tensor Runtime::integrate(torch::Tensor state, const BaseStep& base_step) const {
        if (descriptor.is_sde()) {
            return Details::integrate_sde(descriptor, std::move(state), base_step);
        }
        return Details::integrate_ode(descriptor, std::move(state), base_step);
    }
}


#endif // Nott_SOLVER_HPP