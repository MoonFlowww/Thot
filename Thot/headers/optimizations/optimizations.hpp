#pragma once
#include "../tensor.hpp"
#include <memory>
#include <unordered_map>
#include <string>
#include <sstream>

#include "learning_rate.hpp"

namespace Thot {
    class Optimizer {
    protected:
        float learning_rate_;
        LrFn lr_fn_;

    public:
        Optimizer(float learning_rate, LrFn lr_fn = nullptr) : learning_rate_(learning_rate), lr_fn_(lr_fn) {}

        virtual ~Optimizer() = default;

        virtual void update(Utils::Tensor& weights, const Utils::Tensor& gradients) = 0;
        virtual std::string get_name() const = 0;
        virtual std::string get_params() const = 0;

        float get_learning_rate() const { return learning_rate_; }
        void set_learning_rate(float lr) { learning_rate_ = lr; }

        void step_lr(int epoch, int fold) {
            if (lr_fn_) {
                learning_rate_ = lr_fn_(epoch, fold);
            }
        }

        static std::shared_ptr<Optimizer> SGD(float learning_rate = 0.01f, LrFn lr_fn = nullptr);
        static std::shared_ptr<Optimizer> SGD(float learning_rate, LrSchedule lr_sched);

        static std::shared_ptr<Optimizer> SGDM(float learning_rate = 0.01f, float momentum = 0.9f, LrFn lr_fn = nullptr);
        static std::shared_ptr<Optimizer> SGDM(float learning_rate, float momentum, LrSchedule lr_sched);

        static std::shared_ptr<Optimizer> Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, LrFn lr_fn = nullptr);
        static std::shared_ptr<Optimizer> Adam(float learning_rate, float beta1, float beta2, float epsilon, LrSchedule lr_sched);

        static std::shared_ptr<Optimizer> Muon(float lr = 0.01f, float beta = 0.9f, float weight_decay = 0.0f, LrFn lr_fn = nullptr);
        static std::shared_ptr<Optimizer> Muon(float lr, float beta, float weight_decay, LrSchedule lr_sched);

        static std::shared_ptr<Optimizer> AdaMuon(float lr = 0.01f, float beta1 = 0.9f, float beta2 =0.999f, float weight_decay = 0.0f, LrFn lr_fn = nullptr);
        static std::shared_ptr<Optimizer> AdaMuon(float lr, float beta1, float beta2, float weight_decay, LrSchedule lr_sched);

    };

    class SGD;
    class SGDM;
    class Adam;
    class Muon;
    class AdaMuon;
}

#include "details/sgd.hpp"
#include "details/sgdm.hpp"
#include "details/adam.hpp"
#include "details/muon.hpp"
#include "details/adamuon.hpp"


namespace Thot {
    inline std::shared_ptr<Optimizer> Optimizer::SGD(float learning_rate, LrFn lr_fn) {
        return std::make_shared<Thot::SGD>(learning_rate, lr_fn);
    }

    inline std::shared_ptr<Optimizer> Optimizer::SGD(float learning_rate, LrSchedule lr_sched) {
        LrFn fn = lr_sched ? lr_sched(learning_rate) : nullptr;
        return std::make_shared<Thot::SGD>(learning_rate, fn);
    }

    inline std::shared_ptr<Optimizer> Optimizer::SGDM(float learning_rate, float momentum, LrFn lr_fn) {
        return std::make_shared<Thot::SGDM>(learning_rate, momentum, lr_fn);
    }

    inline std::shared_ptr<Optimizer> Optimizer::SGDM(float learning_rate, float momentum, LrSchedule lr_sched) {
        LrFn fn = lr_sched ? lr_sched(learning_rate) : nullptr;
        return std::make_shared<Thot::SGDM>(learning_rate, momentum, fn);
    }

    inline std::shared_ptr<Optimizer> Optimizer::Adam(float learning_rate, float beta1, float beta2, float epsilon, LrFn lr_fn) {
        return std::make_shared<Thot::Adam>(learning_rate, beta1, beta2, epsilon, lr_fn);
    }
    inline std::shared_ptr<Optimizer> Optimizer::Adam(float learning_rate, float beta1, float beta2, float epsilon, LrSchedule lr_sched) {
        LrFn fn = lr_sched ? lr_sched(learning_rate) : nullptr;
        return std::make_shared<Thot::Adam>(learning_rate, beta1, beta2, epsilon, fn);
    }

    inline std::shared_ptr<Optimizer> Optimizer::Muon(float lr, float beta, float weight_decay, LrFn lr_fn) {
        return std::make_shared<Thot::Muon>(lr, beta, weight_decay, lr_fn);
    }

    inline std::shared_ptr<Optimizer> Optimizer::Muon(float lr, float beta, float weight_decay, LrSchedule lr_sched) {
        LrFn fn = lr_sched ? lr_sched(lr) : nullptr;
        return std::make_shared<Thot::Muon>(lr, beta, weight_decay, fn);
    }
    inline std::shared_ptr<Optimizer> Optimizer::AdaMuon(float lr, float beta1, float beta2, float weight_decay, LrFn lr_fn) {
        return std::make_shared<Thot::AdaMuon>(lr, beta1, beta2, weight_decay, lr_fn);
    }
    inline std::shared_ptr<Optimizer> Optimizer::AdaMuon(float lr, float beta1, float beta2, float weight_decay, LrSchedule lr_sched) {
        LrFn fn = lr_sched ? lr_sched(lr) : nullptr;
        return std::make_shared<Thot::AdaMuon>(lr, beta1, beta2, weight_decay, fn);
    }



}