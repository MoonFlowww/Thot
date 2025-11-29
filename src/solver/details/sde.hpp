#ifndef OMNI_SOLVER_DETAILS_SDE_HPP
#define OMNI_SOLVER_DETAILS_SDE_HPP


#include <cmath>
#include <stdexcept>

namespace Omni::Solver::Details {
    inline torch::Tensor integrate_sde(const Descriptor& descriptor,
                                       torch::Tensor state,
                                       const BaseStep& base_step) {
        if (!base_step) {
            throw std::invalid_argument("SDE solver requested without a base step function.");
        }

        const auto step = descriptor.step_control.effective_step();
        const auto sqrt_step = std::sqrt(step);

        auto compute_diffusion = [&](const torch::Tensor& drift) {
            switch (descriptor.noise) {
                case NoiseModel::None:
                    return torch::zeros_like(drift);
                case NoiseModel::Additive:
                    return torch::ones_like(drift);
                case NoiseModel::Diagonal:
                    return drift.abs();
                case NoiseModel::Full:
                    return drift;
                default:
                    return torch::zeros_like(drift);
            }
        };

        auto drift = base_step(state);
        auto diffusion = compute_diffusion(drift);
        auto dW = torch::randn_like(state) * sqrt_step;

        switch (descriptor.method) {
            case MethodFamily::EulerMaruyama: {
                return state + drift * step + diffusion * dW;
            }
            case MethodFamily::Milstein: {
                auto grad = torch::where(state.abs() > 1e-6, diffusion / (state.abs() + 1e-6), torch::zeros_like(state));
                auto noise_term = diffusion * dW + 0.5 * diffusion * grad * (dW * dW - step);
                return state + drift * step + noise_term;
            }
            case MethodFamily::StochasticRungeKutta: {
                auto y_tilde = state + drift * step + diffusion * dW;
                auto drift_tilde = base_step(y_tilde);
                return state + 0.5 * (drift + drift_tilde) * step + diffusion * dW;
            }
            default:
                throw std::invalid_argument("Requested SDE integration with an ODE method.");
        }
    }
}
#endif // OMNI_SOLVER_DETAILS_SDE_HPP