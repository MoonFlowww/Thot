#pragma once


#include <sstream>



#include "../tensor.hpp"
#include "../../cuda/cuh/losses/loss.cuh"


namespace Thot {
    enum class Loss {
        MSE,
        MAE,
        BinaryCrossEntropy,
        CrossEntropy,
        CategoricalCrossEntropy,
        SparseCategoricalCrossEntropy,
        Hinge,
        Huber,
        KLDivergence,
        VAE
    };

    inline Loss loss_from_string(const std::string &name) { // for model.save() & model.load()
        if (name == "MSE") return Loss::MSE;
        if (name == "MAE") return Loss::MAE;
        if (name == "BCE") return Loss::BinaryCrossEntropy;
        if (name == "CE") return Loss::CrossEntropy;
        if (name == "CCE") return Loss::CategoricalCrossEntropy;
        if (name == "SCCE") return Loss::SparseCategoricalCrossEntropy;
        if (name == "Hinge") return Loss::Hinge;
        if (name == "Huber") return Loss::Huber;
        if (name == "KL div") return Loss::KLDivergence;
        if (name == "VAE") return Loss::VAE;
        throw std::runtime_error("Unknown loss: " + name);
    }
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
            case Loss::CrossEntropy: return "CE";
            case Loss::CategoricalCrossEntropy: return "CCE";
            case Loss::SparseCategoricalCrossEntropy: return "SCCE";
            case Loss::Hinge: return "Hinge";
            case Loss::Huber: return "Huber";
            case Loss::KLDivergence: return "KL div";
                case Loss::VAE: return "VAE";
            default: return "Unknown";
            }
        }

        float compute(const Utils::Tensor& predictions, const Utils::Tensor& targets) {
            size_t buffer_size = predictions.size();
            float* loss = nullptr;
            cudaMalloc(&loss, buffer_size * sizeof(float));

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

            case Loss::CrossEntropy:
                cuda::losses::launchCrossEntropy(
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

                case Loss::VAE:
                    cuda::losses::launchMSE(
                        static_cast<float*>(predictions.data()),
                        static_cast<float*>(targets.data()),
                        loss,
                        predictions.size()
                    );
                    break;
            }

            size_t reduce_size = predictions.size();
            if (Loss_ == Loss::CategoricalCrossEntropy || Loss_ == Loss::SparseCategoricalCrossEntropy) {
                reduce_size = predictions.shape()[0];
            }


            //old version
            //float result;
            //cudaMemcpy(&result, loss, sizeof(float), cudaMemcpyDeviceToHost);

            //new version
            auto result = cuda::losses::reduceLoss(
                loss,
                static_cast<int>(reduce_size)
            );
            //end

            cudaFree(loss);
            int normalizer = 0;
            switch (Loss_) {
                case Loss::CategoricalCrossEntropy:
                case Loss::SparseCategoricalCrossEntropy:
                    normalizer = predictions.shape()[0];
                    break;
                default:
                    normalizer = predictions.size();
                    break;
            }
            return result / static_cast<float>(normalizer);
        }

        float compute_with_kl(const Utils::Tensor& reconstruction, const Utils::Tensor& targets,
                              const Utils::Tensor& mean, const Utils::Tensor& logvar) {
            float recon = compute(reconstruction, targets);

            size_t buffer_size = mean.size();
            float* kl = nullptr;
            cudaMalloc(&kl, buffer_size * sizeof(float));
            cuda::losses::launchKLDivergence(
                static_cast<float*>(mean.data()),
                static_cast<float*>(logvar.data()),
                kl,
                buffer_size,
                epsilon_
            );
            auto kl_val = cuda::losses::reduceLoss(kl, static_cast<int>(buffer_size));
            cudaFree(kl);
            return recon + kl_val / buffer_size;
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

            case Loss::CrossEntropy:
                cuda::losses::launchCrossEntropyGradient(
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