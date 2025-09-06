#pragma once

#include "../../tensor.hpp"
#include "../../initializations/initializations.hpp"
#include "../../activations/activations.hpp"
#include "conv2d.hpp"
#include "../../../cuda/cuh/layers/rcnn.cuh"

#include <vector>
#include <cmath>
#include <cfloat>

namespace Thot {

    class Layer;
    class Network;

    class RCNNLayer : public Layer {
    private:
        friend class Network;
        Conv2DLayer conv_;
        int pooled_h_;
        int pooled_w_;
        int num_rois_per_image_;
        Utils::Tensor rois_;
        Utils::Tensor conv_output_;
        ConvAlgo conv_algo_;

    public:
        RCNNLayer(int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding,
            int pooled_h, int pooled_w,
            Activation activation_type = Activation::ReLU,
            Initialization weight_init = Initialization::Xavier,
            ConvAlgo conv_algo = ConvAlgo::Auto,
            const std::string& name = "RCNN")
            : Layer(name),
            conv_(in_channels, in_height, in_width, out_channels, kernel_size, stride, padding,
            activation_type, weight_init, conv_algo, name + "_conv"),
            pooled_h_(pooled_h), pooled_w_(pooled_w), num_rois_per_image_(4), conv_algo_(conv_algo) {}

        size_t get_flops(int batch_size = 1) const override {
            int rois = num_rois_per_image_ * batch_size;
            size_t pool_flops = static_cast<size_t>(rois) * conv_.out_channels() * pooled_h_ * pooled_w_;
            return conv_.get_flops(batch_size) + pool_flops;
        }

        size_t get_parameters() const override { return conv_.get_parameters(); }
        Activation get_activation() const override { return conv_.get_activation(); }
        Initialization get_initialization() const override { return conv_.get_initialization(); }
        float get_latency() const override { return conv_.get_latency(); }

        int get_input_size() const override { return conv_.get_input_size(); }
        int get_output_size() const override {
            return conv_.out_channels() * pooled_h_ * pooled_w_ * num_rois_per_image_;
        }

        Utils::Tensor& weights() { return conv_.weights(); }
        Utils::Tensor& bias() { return conv_.bias(); }

        Utils::Tensor forward(const Utils::Tensor& input) override {
            conv_output_ = conv_.forward(input);
            int batch = conv_output_.shape()[0];
            int channels = conv_output_.shape()[1];
            int height = conv_output_.shape()[2];
            int width = conv_output_.shape()[3];

            int num_rois = num_rois_per_image_ * batch;
            std::vector<float> host_rois(num_rois * 5);
            for (int b = 0; b < batch; ++b) {
                float x_mid = width / 2.0f;
                float y_mid = height / 2.0f;
                int base = b * num_rois_per_image_ * 5;
                host_rois[base + 0] = b; host_rois[base + 1] = 0;     host_rois[base + 2] = 0;     host_rois[base + 3] = x_mid;  host_rois[base + 4] = y_mid;
                host_rois[base + 5] = b; host_rois[base + 6] = x_mid; host_rois[base + 7] = 0;     host_rois[base + 8] = width;  host_rois[base + 9] = y_mid;
                host_rois[base +10] = b; host_rois[base +11] = 0;     host_rois[base +12] = y_mid; host_rois[base +13] = x_mid;  host_rois[base +14] = height;
                host_rois[base +15] = b; host_rois[base +16] = x_mid; host_rois[base +17] = y_mid; host_rois[base +18] = width;  host_rois[base +19] = height;
            }
            rois_.reshape({ num_rois, 5 });
            rois_.upload(host_rois);

            Utils::Tensor output({ num_rois, channels, pooled_h_, pooled_w_ });
            ::cuda::layers::launchROIPoolForward(
                static_cast<float*>(conv_output_.data()),
                static_cast<float*>(rois_.data()),
                static_cast<float*>(output.data()),
                num_rois, channels, height, width, pooled_h_, pooled_w_
            );
            return std::move(output);
        }

        Utils::Tensor backward(const Utils::Tensor& grad_output) override {
            int batch = conv_output_.shape()[0];
            int channels = conv_output_.shape()[1];
            int height = conv_output_.shape()[2];
            int width = conv_output_.shape()[3];
            int num_rois = num_rois_per_image_ * batch;

            Utils::Tensor grad_conv({ batch, channels, height, width }, true);

            ::cuda::layers::launchROIPoolBackward(
                static_cast<float*>(grad_output.data()),
                static_cast<float*>(conv_output_.data()),
                static_cast<float*>(rois_.data()),
                static_cast<float*>(grad_conv.data()),
                num_rois, channels, height, width, pooled_h_, pooled_w_
            );

            return conv_.backward(grad_conv);
        }
    };
}