#ifndef THOT_DROPOUT_HPP
#define THOT_DROPOUT_HPP
#include "../../activation/activation.hpp"
#include "../../common/local.hpp"
#include <torch/torch.h>

namespace Thot::Layer::Details {

    struct HardDropoutOptions {
        double probability{0.5};
        bool inplace{false};
    };

    struct HardDropoutDescriptor {
        HardDropoutOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
        ::Thot::LocalConfig local{};
    };

    class HardDropoutImpl : public torch::nn::Module {
    public:
        explicit HardDropoutImpl(HardDropoutOptions options = {})
            : options_(options)
        {
            TORCH_CHECK(options_.probability >= 0.0 && options_.probability < 1.0,
                        "HardDropout probability must be in the range [0, 1).");
        }

        torch::Tensor forward(torch::Tensor input)
        {
            if (!input.defined()) {
                return input;
            }

            TORCH_CHECK(input.is_floating_point(),
                        "HardDropout expects floating point tensors.");

            if (!is_training() || options_.probability == 0.0) {
                return input;
            }

            const double keep_prob = 1.0 - options_.probability;
            TORCH_CHECK(keep_prob > 0.0,
                        "HardDropout probability must be strictly less than 1.");

            auto mask = torch::bernoulli(torch::full_like(input, keep_prob));
            auto output = mask.mul(input).div_(keep_prob);

            if (options_.inplace) {
                input.copy_(output);
                return input;
            }

            return output;
        }

        [[nodiscard]] const HardDropoutOptions& options() const noexcept { return options_; }

    private:
        HardDropoutOptions options_{};
    };

    TORCH_MODULE(HardDropout);









    struct SoftDropoutOptions {
        double probability{0.5};
        double noise_mean{0.0};
        double noise_std{1.0};
        bool inplace{false};
    };

    struct SoftDropoutDescriptor {
        SoftDropoutOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
        ::Thot::LocalConfig local{};
    };

    class SoftDropoutImpl : public torch::nn::Module {
    public:
        explicit SoftDropoutImpl(SoftDropoutOptions options = {})
            : options_(options)
        {
            TORCH_CHECK(options_.probability >= 0.0 && options_.probability < 1.0,
                        "SoftDropout probability must be in the range [0, 1).");
            TORCH_CHECK(options_.noise_std >= 0.0,
                        "SoftDropout noise standard deviation must be non-negative.");
        }

        torch::Tensor forward(torch::Tensor input)
        {
            if (!input.defined()) {
                return input;
            }

            TORCH_CHECK(input.is_floating_point(),
                        "SoftDropout expects floating point tensors.");

            if (!is_training() || options_.probability == 0.0) {
                return input;
            }

            const double keep_prob = 1.0 - options_.probability;
            TORCH_CHECK(keep_prob > 0.0,
                        "SoftDropout probability must be strictly less than 1.");

            auto mask = torch::bernoulli(torch::full_like(input, keep_prob)).to(torch::kBool);
            auto scaled_input = input / keep_prob;
            auto noise = torch::normal(options_.noise_mean,
                                       options_.noise_std,
                                       input.sizes(),
                                       input.options());
            auto noisy_input = scaled_input + noise;
            auto output = torch::where(mask, scaled_input, noisy_input);

            if (options_.inplace) {
                input.copy_(output);
                return input;
            }

            return output;
        }


        [[nodiscard]] const SoftDropoutOptions& options() const noexcept { return options_; }

    private:
        SoftDropoutOptions options_{};
    };

    TORCH_MODULE(SoftDropout);

}

#endif //THOT_DROPOUT_HPP