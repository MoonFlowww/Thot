#ifndef THOT_MUON_HPP
#define THOT_MUON_HPP
// Muon, AdaMuon, MuonManifold
// Muon: https://arxiv.org/pdf/2502.16982
// AdaMuon: https://arxiv.org/pdf/2507.11005
// MuonManifold: https://thinkingmachines.ai/blog/modular-manifolds/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>

#include <torch/torch.h>

namespace Thot::Optimizer::Details {

    namespace detail {
        inline auto make_scalar_like(const torch::Tensor& reference, double value) -> torch::Tensor {
            if (reference.defined()) {
                return torch::full({}, value, reference.options());
            }
            return torch::full({}, value, torch::TensorOptions().dtype(torch::kFloat));
        }

        inline auto safe_norm(const torch::Tensor& tensor, double eps) -> torch::Tensor {
            auto norm = torch::linalg_vector_norm(tensor);
            return norm.clamp_min(eps);
        }

        inline auto clip_by_max_norm(const torch::Tensor& update, double max_norm, double eps) -> torch::Tensor {
            if (max_norm <= 0.0) {
                return update;
            }

            auto current_norm = torch::linalg_vector_norm(update);
            auto threshold = detail::make_scalar_like(current_norm, max_norm);
            auto eps_scalar = detail::make_scalar_like(current_norm, eps);
            auto scale = (threshold / (current_norm + eps_scalar)).clamp_max(1.0);
            return update * scale;
        }

        inline void ensure_tensor_like(torch::Tensor& storage, const torch::Tensor& reference) {
            if (!storage.defined()) {
                storage = torch::zeros_like(reference);
            }
        }

        inline void apply_weight_decay(torch::Tensor& update, const torch::Tensor& param, double weight_decay) {
            if (weight_decay == 0.0) {
                return;
            }

            update = update + weight_decay * param;
        }
    }

    struct MuonOptions {
        double learning_rate{1e-3};
        double beta{0.95};
        double weight_decay{0.0};
        double epsilon{1e-8};
        double max_update_norm{0.0};

        [[nodiscard]] inline auto validated() const -> MuonOptions {
            auto copy = *this;
            copy.beta = std::clamp(copy.beta, 0.0, 1.0);
            copy.learning_rate = std::max(copy.learning_rate, 0.0);
            copy.epsilon = std::max(copy.epsilon, 0.0);
            copy.max_update_norm = std::max(copy.max_update_norm, 0.0);
            return copy;
        }
    };

    struct MuonState {
        torch::Tensor exp_avg{};
        int64_t step{0};

        void reset() {
            exp_avg = torch::Tensor();
            step = 0;
        }
    };

    inline void muon_step_(torch::Tensor& param,
                           const torch::Tensor& grad,
                           MuonState& state,
                           const MuonOptions& raw_options) {
        auto options = raw_options.validated();

        detail::ensure_tensor_like(state.exp_avg, grad);

        state.exp_avg.mul_(options.beta).add_(grad, 1.0 - options.beta);

        auto parameter_norm = detail::safe_norm(param, options.epsilon);
        auto direction_norm = detail::safe_norm(state.exp_avg, options.epsilon);
        auto direction = state.exp_avg * (parameter_norm / direction_norm);

        detail::apply_weight_decay(direction, param, options.weight_decay);
        direction = detail::clip_by_max_norm(direction, options.max_update_norm, options.epsilon);

        param.add_(direction, -options.learning_rate);
        state.step += 1;
    }

    struct MuonDescriptor {
        MuonOptions options{};
    };

    struct AdaMuonOptions : MuonOptions {
        double beta2{0.999};

        [[nodiscard]] inline auto validated() const -> AdaMuonOptions {
            auto copy = *this;
            copy.beta = std::clamp(copy.beta, 0.0, 1.0);
            copy.beta2 = std::clamp(copy.beta2, 0.0, 1.0);
            copy.learning_rate = std::max(copy.learning_rate, 0.0);
            copy.epsilon = std::max(copy.epsilon, 0.0);
            copy.max_update_norm = std::max(copy.max_update_norm, 0.0);
            return copy;
        }
    };

    struct AdaMuonState : MuonState {
        torch::Tensor exp_avg_sq{};

        void reset() {
            MuonState::reset();
            exp_avg_sq = torch::Tensor();
        }
    };

    inline void ada_muon_step_(torch::Tensor& param,
                               const torch::Tensor& grad,
                               AdaMuonState& state,
                               const AdaMuonOptions& raw_options) {
        auto options = raw_options.validated();

        detail::ensure_tensor_like(state.exp_avg, grad);
        detail::ensure_tensor_like(state.exp_avg_sq, grad);

        state.step += 1;

        state.exp_avg.mul_(options.beta).add_(grad, 1.0 - options.beta);
        state.exp_avg_sq.mul_(options.beta2).addcmul_(grad, grad, 1.0 - options.beta2);

        auto bias_correction1 = 1.0 - std::pow(options.beta, static_cast<double>(state.step));
        auto bias_correction2 = 1.0 - std::pow(options.beta2, static_cast<double>(state.step));

        auto corrected_exp_avg = state.exp_avg / bias_correction1;
        auto corrected_exp_avg_sq = state.exp_avg_sq / bias_correction2;
        auto denom = torch::sqrt(corrected_exp_avg_sq).add_(options.epsilon);
        auto direction = corrected_exp_avg / denom;

        auto parameter_norm = detail::safe_norm(param, options.epsilon);
        auto direction_norm = detail::safe_norm(direction, options.epsilon);
        direction.mul_(parameter_norm / direction_norm);

        detail::apply_weight_decay(direction, param, options.weight_decay);
        direction = detail::clip_by_max_norm(direction, options.max_update_norm, options.epsilon);

        param.add_(direction, -options.learning_rate);
    }

    struct AdaMuonDescriptor {
        AdaMuonOptions options{};
    };

    enum class ManifoldKind {
        Euclidean,
        UnitSphere,
        Stiefel
    };

    struct MuonManifoldOptions : AdaMuonOptions {
        ManifoldKind manifold{ManifoldKind::UnitSphere};
        double retraction_epsilon{1e-12};
        bool renormalize{true};

        [[nodiscard]] inline auto validated() const -> MuonManifoldOptions {
            auto base = AdaMuonOptions::validated();
            MuonManifoldOptions copy;
            static_cast<AdaMuonOptions&>(copy) = base;
            copy.manifold = manifold;
            copy.retraction_epsilon = std::max(retraction_epsilon, 0.0);
            copy.renormalize = renormalize;
            return copy;
        }
    };

    struct MuonManifoldState : AdaMuonState {
        void reset() {
            AdaMuonState::reset();
        }
    };

    inline void retract_to_manifold(torch::Tensor& param,
                                    const MuonManifoldOptions& options) {
        switch (options.manifold) {
            case ManifoldKind::Euclidean:
                return;
            case ManifoldKind::UnitSphere: {
                auto norm = detail::safe_norm(param, options.retraction_epsilon);
                param = param / norm;
                return;
            }
            case ManifoldKind::Stiefel: {
                auto qr = std::get<0>(torch::linalg_qr(param));
                param = qr;
                return;
            }
        }
    }

    inline void muon_manifold_step_(torch::Tensor& param,
                                    const torch::Tensor& grad,
                                    MuonManifoldState& state,
                                    const MuonManifoldOptions& raw_options) {
        auto options = raw_options.validated();
        ada_muon_step_(param, grad, state, options);
        if (options.renormalize) {
            retract_to_manifold(param, options);
        }
    }

    struct MuonManifoldDescriptor {
        MuonManifoldOptions options{};
    };
}

#endif //THOT_MUON_HPP