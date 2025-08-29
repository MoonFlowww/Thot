#pragma once

#include "../tensor.hpp"
#include "../../cuda/cuh/losses/loss.cuh"
#include <sstream>

namespace Thot {
    enum class Loss {
        MSE,
        MAE,
        BinaryCrossEntropy,
        CategoricalCrossEntropy,
        SparseCategoricalCrossEntropy,
        Hinge,
        Huber,
        KLDivergence
    };
    class Losses {
    public:


    private:
        Loss Loss_;
        float epsilon_;
        float delta_;

    public:
        Losses(Loss Loss = Loss::MSE, float epsilon = 1e-8f, float delta = 1.0f)
            : Loss_(Loss), epsilon_(epsilon), delta_(delta) {
        }

        Loss get_type() const {
            return Loss_;
        }

        std::string get_params() const {
            std::ostringstream oss;
            oss << "Eps=" << epsilon_;
            if (Loss_ == Loss::Huber) {
                oss << "\nDelta=" << delta_;
            }
            return oss.str();
        }

        static std::string to_string(Loss type) {
            switch (type) {
            case Loss::MSE: return "MSE";
            case Loss::MAE: return "MAE";
            case Loss::BinaryCrossEntropy: return "BCE";
            case Loss::CategoricalCrossEntropy: return "CCE";
            case Loss::SparseCategoricalCrossEntropy: return "SCCE";
            case Loss::Hinge: return "Hinge";
            case Loss::Huber: return "Huber";
            case Loss::KLDivergence: return "KL div";
            default: return "Unknown";
            }
        }

        float compute(const Utils::Tensor& predictions, const Utils::Tensor& targets) {
            float* loss = nullptr;
            cudaMalloc(&loss, sizeof(float));

            switch (Loss_) {
            case Loss::MSE:
                cuda::losses::launchMSE(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    loss,
                    predictions.size()
                );
                break;

            case Loss::MAE:
                cuda::losses::launchMAE(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    loss,
                    predictions.size()
                );
                break;

            case Loss::BinaryCrossEntropy:
                cuda::losses::launchBinaryCrossEntropy(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    loss,
                    predictions.size(),
                    epsilon_
                );
                break;

            case Loss::CategoricalCrossEntropy:
                cuda::losses::launchCategoricalCrossEntropy(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    loss,
                    predictions.shape()[0],
                    predictions.shape()[1],
                    epsilon_
                );
                break;

            case Loss::SparseCategoricalCrossEntropy:
                cuda::losses::launchSparseCategoricalCrossEntropy(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    loss,
                    predictions.shape()[0],
                    predictions.shape()[1],
                    epsilon_
                );
                break;

            case Loss::Hinge:
                cuda::losses::launchHinge(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    loss,
                    predictions.size()
                );
                break;

            case Loss::Huber:
                cuda::losses::launchHuber(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    loss,
                    predictions.size(),
                    delta_
                );
                break;

            case Loss::KLDivergence:
                cuda::losses::launchKLDivergence(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    loss,
                    predictions.size(),
                    epsilon_
                );
                break;
            }

            float result;
            cudaMemcpy(&result, loss, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(loss);
            return result;
        }

        Utils::Tensor compute_gradients(const Utils::Tensor& predictions, const Utils::Tensor& targets) {
            Utils::Tensor gradients(predictions.shape());

            switch (Loss_) {
            case Loss::MSE:
                cuda::losses::launchMSEGradient(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    static_cast<float*>(gradients.data()),
                    predictions.size()
                );
                break;

            case Loss::MAE:
                cuda::losses::launchMAEGradient(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    static_cast<float*>(gradients.data()),
                    predictions.size()
                );
                break;

            case Loss::BinaryCrossEntropy:
                cuda::losses::launchBinaryCrossEntropyGradient(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    static_cast<float*>(gradients.data()),
                    predictions.size(),
                    epsilon_
                );
                break;

            case Loss::CategoricalCrossEntropy:
                cuda::losses::launchCategoricalCrossEntropyGradient(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    static_cast<float*>(gradients.data()),
                    predictions.shape()[0],
                    predictions.shape()[1],
                    epsilon_
                );
                break;

            case Loss::SparseCategoricalCrossEntropy:
                cuda::losses::launchSparseCategoricalCrossEntropyGradient(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    static_cast<float*>(gradients.data()),
                    predictions.shape()[0],
                    predictions.shape()[1],
                    epsilon_
                );
                break;

            case Loss::Hinge:
                cuda::losses::launchHingeGradient(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    static_cast<float*>(gradients.data()),
                    predictions.size()
                );
                break;

            case Loss::Huber:
                cuda::losses::launchHuberGradient(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    static_cast<float*>(gradients.data()),
                    predictions.size(),
                    delta_
                );
                break;

            case Loss::KLDivergence:
                cuda::losses::launchKLDivergenceGradient(
                    static_cast<float*>(predictions.data()),
                    static_cast<float*>(targets.data()),
                    static_cast<float*>(gradients.data()),
                    predictions.size(),
                    epsilon_
                );
                break;
            }

            return gradients;
        }
    };
}