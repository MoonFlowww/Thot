#ifndef OMNI_SOLVER_DETAILS_ODE_HPP
#define OMNI_SOLVER_DETAILS_ODE_HPP

#include <cmath>
#include <stdexcept>

namespace Omni::Solver::Details {

    inline torch::Tensor integrate_ode(const Descriptor& descriptor,
                                       torch::Tensor state,
                                       const BaseStep& base_step) {
        if (!base_step) {
            throw std::invalid_argument("ODE solver requested without a base step function.");
        }

        const auto step = descriptor.step_control.effective_step();

        switch (descriptor.method) {
            case MethodFamily::Euler: {
                auto derivative = base_step(state);
                return state + derivative * step;
            }
            case MethodFamily::RungeKutta: {
                auto k1 = base_step(state);
                auto k2 = base_step(state + 0.5 * step * k1);
                auto k3 = base_step(state + 0.5 * step * k2);
                auto k4 = base_step(state + step * k3);
                return state + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
            }
            case MethodFamily::BDF: {
                // Use a simple fixed-point refinement to approximate an implicit BDF step.
                auto predictor = state + step * base_step(state);
                auto current = predictor.clone();
                for (int iteration = 0; iteration < 3; ++iteration) {
                    auto f = base_step(current);
                    current = state + step * f;
                }
                return current;
            }
            default:
                throw std::invalid_argument("Requested ODE integration with an SDE method.");
        }
    }

}

#endif // OMNI_SOLVER_DETAILS_ODE_HPP