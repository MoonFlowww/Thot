#ifndef THOT_SOLVER_REGISTRY_HPP
#define THOT_SOLVER_REGISTRY_HPP

#include "solver.hpp"

#include "details/ode.hpp"
#include "details/sde.hpp"

namespace Thot::Solver {
    inline torch::Tensor Runtime::integrate(torch::Tensor state, const BaseStep& base_step) const {
        if (descriptor.is_sde()) {
            return Details::integrate_sde(descriptor, std::move(state), base_step);
        }
        return Details::integrate_ode(descriptor, std::move(state), base_step);
    }

}

#endif //THOT_SOLVER_REGISTRY_HPP