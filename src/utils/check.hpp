#ifndef THOT_CHECK_HPP
#define THOT_CHECK_HPP
#pragma once

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace Thot {
    inline std::string Model::CheckReport::to_string() const
    {
        auto format_shape = [](const std::vector<int64_t>& shape) {
            if (shape.empty()) {
                return std::string{"()"};
            }

            std::ostringstream stream;
            stream << '(';
            for (std::size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) {
                    stream << ", ";
                }
                stream << shape[i];
            }
            stream << ')';
            return stream.str();
        };

        std::ostringstream stream;
        stream << "Model diagnostics: " << (ok ? "PASS" : "FAIL") << '\n';

        if (!warnings.empty()) {
            stream << "Warnings:" << '\n';
            for (const auto& warning : warnings) {
                stream << "  • " << warning << '\n';
            }
        }

        for (const auto& layer : layers) {
            stream << "[#" << layer.index << "] " << layer.module_name;
            if (!layer.activation.empty() && layer.activation != "Identity") {
                stream << " + " << layer.activation;
            }
            stream << '\n';
            stream << "    input : " << format_shape(layer.input_shape) << '\n';
            if (!layer.expected_input_shape.empty()) {
                stream << "    expect input : " << format_shape(layer.expected_input_shape) << '\n';
            }
            if (layer.ok) {
                stream << "    output: " << format_shape(layer.output_shape) << '\n';
                if (!layer.expected_output_shape.empty()) {
                    stream << "    expect output: " << format_shape(layer.expected_output_shape) << '\n';
                }
                if (!layer.message.empty()) {
                    stream << "    note  : " << layer.message << '\n';
                }
            } else {
                stream << "    status: FAILED" << '\n';
                if (!layer.output_shape.empty()) {
                    stream << "    output: " << format_shape(layer.output_shape) << '\n';
                }
                if (!layer.expected_output_shape.empty()) {
                    stream << "    expect output: " << format_shape(layer.expected_output_shape) << '\n';
                }
                if (!layer.message.empty()) {
                    stream << "    reason: " << layer.message << '\n';
                }
            }
        }

        stream << "Backward diagnostics: " << (backward_ok ? "PASS" : "FAIL") << '\n';
        if (!backward_ok && !backward_message.empty()) {
            stream << "  reason: " << backward_message << '\n';
        }

        return stream.str();
    }

    inline Model::CheckReport Model::check() const
    {
        return check_with_shape(last_input_shape_);
    }

    inline Model::CheckReport Model::check(const torch::Tensor& prototype_input) const
    {
        if (!prototype_input.defined()) {
            throw std::invalid_argument("Model::check requires a defined prototype input tensor.");
        }

        return run_diagnostics(prototype_input);
    }

    inline Model::CheckReport Model::check(const std::vector<int64_t>& input_shape) const
    {
        if (input_shape.empty()) {
            throw std::invalid_argument("Model::check requires a non-empty input shape.");
        }

        if (std::any_of(input_shape.begin(), input_shape.end(), [](int64_t dimension) { return dimension <= 0; })) {
            throw std::invalid_argument("Model::check requires all input dimensions to be positive.");
        }

        auto tensor = torch::randn(input_shape, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        return run_diagnostics(tensor);
    }

    inline Model::CheckReport Model::check(std::initializer_list<int64_t> input_shape) const
    {
        return check(std::vector<int64_t>(input_shape));
    }

    inline Model::CheckReport Model::check_with_shape(const std::optional<std::vector<int64_t>>& shape) const
    {
        if (!shape || shape->empty()) {
            CheckReport report{};
            report.ok = false;
            report.warnings.push_back("Model::check() requires either a previous forward pass or an explicit input shape. Run a forward pass or provide a prototype tensor to Model::check().");
            return report;
        }
        return check(*shape);
    }

    inline Model::CheckReport Model::run_diagnostics(torch::Tensor prototype_input) const
    {
        CheckReport report{};

        torch::Tensor input = std::move(prototype_input);
        if (!input.defined()) {
            throw std::invalid_argument("Model::check requires a defined prototype input tensor.");
        }

        if (input.device() != device_) {
            input = input.to(device_);
        }
        if (!input.is_contiguous()) {
            input = input.contiguous();
        }

        input = input.detach();
        input.requires_grad_(true);

        last_input_shape_ = tensor_shape_vector(input);

        if (layers_.empty()) {
            report.ok = false;
            report.backward_ok = false;
            report.backward_message = "Backward pass skipped because the model has no layers.";
            report.warnings.push_back("Model does not contain any registered layers.");
            return report;
        }

        if (!has_optimizer()) {
            report.warnings.push_back("Optimizer has not been configured.");
        }
        if (!has_loss()) {
            report.warnings.push_back("Loss function has not been configured.");
        }




        report.layers.reserve(layers_.size());

        bool success = true;
        torch::InferenceMode guard{};
        auto current = input;

        auto pad_shape = [](const std::vector<int64_t>& actual, std::size_t minimum_rank) {
            std::vector<int64_t> padded = actual;
            if (padded.size() < minimum_rank) {
                padded.resize(minimum_rank, static_cast<int64_t>(-1));
                for (std::size_t index = 0; index < actual.size(); ++index) {
                    padded[index] = actual[index];
                }
            }
            return padded;
        };

        auto shape_to_string = [](const std::vector<int64_t>& shape) {
            if (shape.empty()) {
                return std::string{"()"};
            }

            std::ostringstream stream;
            stream << '(';
            for (std::size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) {
                    stream << ", ";
                }
                stream << shape[i];
            }
            stream << ')';
            return stream.str();
        };

        auto append_message = [](std::string& target, const std::string& addition) {
            if (addition.empty()) {
                return;
            }
            if (!target.empty() && target.back() != ' ') {
                target.push_back(' ');
            }
            target += addition;
        };

        auto assign_expected_input = [&](LayerCheck& entry, const Layer::Details::RegisteredLayer& layer) {
            if (!layer.module) {
                return;
            }

            if (auto linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(layer.module)) {
                auto expected = entry.input_shape;
                if (expected.empty()) {
                    expected.push_back(linear->options.in_features());
                } else {
                    expected.back() = linear->options.in_features();
                }
                entry.expected_input_shape = std::move(expected);
            } else if (auto conv2d = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(layer.module)) {
                auto expected = pad_shape(entry.input_shape, 4);
                if (expected.size() > 1) {
                    expected[1] = conv2d->options.in_channels();
                }
                entry.expected_input_shape = std::move(expected);
            } else if (auto batchnorm2d = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(layer.module)) {
                auto expected = pad_shape(entry.input_shape, 4);
                if (expected.size() > 1) {
                    expected[1] = batchnorm2d->options.num_features();
                }
                entry.expected_input_shape = std::move(expected);
            } else if (std::dynamic_pointer_cast<torch::nn::DropoutImpl>(layer.module)) {
                entry.expected_input_shape = entry.input_shape;
            } else if (std::dynamic_pointer_cast<torch::nn::FlattenImpl>(layer.module)) {
                entry.expected_input_shape = entry.input_shape;
            }
        };

        auto assign_expected_output = [&](LayerCheck& entry,
                                          const Layer::Details::RegisteredLayer& layer,
                                          const std::vector<int64_t>* actual_output_shape) {
            if (!layer.module) {
                return;
            }

            const auto& fallback = actual_output_shape ? *actual_output_shape : entry.input_shape;
            if (auto linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(layer.module)) {
                auto expected = fallback;
                if (expected.empty()) {
                    expected.push_back(linear->options.out_features());
                } else {
                    expected.back() = linear->options.out_features();
                }
                entry.expected_output_shape = std::move(expected);
            } else if (auto conv2d = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(layer.module)) {
                auto expected = pad_shape(fallback, 4);
                if (expected.size() > 1) {
                    expected[1] = conv2d->options.out_channels();
                }
                entry.expected_output_shape = std::move(expected);
            } else if (auto batchnorm2d = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(layer.module)) {
                auto expected = pad_shape(fallback, 4);
                if (expected.size() > 1) {
                    expected[1] = batchnorm2d->options.num_features();
                }
                entry.expected_output_shape = std::move(expected);
            } else if (std::dynamic_pointer_cast<torch::nn::DropoutImpl>(layer.module)) {
                entry.expected_output_shape = fallback;
            } else if (std::dynamic_pointer_cast<torch::nn::FlattenImpl>(layer.module)) {
                entry.expected_output_shape = fallback;
            }
        };

        for (std::size_t index = 0; index < layers_.size(); ++index) {
            const auto& layer = layers_[index];
            LayerCheck entry{};
            entry.index = index;
            entry.module_name = describe_module(layer);
            entry.activation = describe_activation(layer.activation);
            entry.input_shape = tensor_shape_vector(current);
            assign_expected_input(entry, layer);

            try {
                auto output = layer.forward(current);
                auto activated = Activation::Details::apply(layer.activation, std::move(output));
                entry.output_shape = tensor_shape_vector(activated);
                assign_expected_output(entry, layer, &entry.output_shape);
                entry.ok = true;
                if (!entry.expected_input_shape.empty() && entry.expected_input_shape != entry.input_shape) {
                    append_message(
                        entry.message,
                        "Input shape " + shape_to_string(entry.input_shape) +
                            " differs from expected " + shape_to_string(entry.expected_input_shape) + '.');
                }
                if (!entry.expected_output_shape.empty() && entry.expected_output_shape != entry.output_shape) {
                    append_message(
                        entry.message,
                        "Output shape " + shape_to_string(entry.output_shape) +
                            " differs from expected " + shape_to_string(entry.expected_output_shape) + '.');
                }
                current = std::move(activated);
            } catch (const c10::Error& error) {
                entry.ok = false;
                entry.message = error.msg();
                assign_expected_output(entry, layer, nullptr);
                if (!entry.expected_input_shape.empty() && entry.expected_input_shape != entry.input_shape) {
                    append_message(
                        entry.message,
                        "Expected input " + shape_to_string(entry.expected_input_shape) + " but received " +
                            shape_to_string(entry.input_shape) + '.');
                }
                success = false;
                report.layers.push_back(std::move(entry));
                break;
            } catch (const std::exception& error) {
                entry.ok = false;
                entry.message = error.what();
                assign_expected_output(entry, layer, nullptr);
                if (!entry.expected_input_shape.empty() && entry.expected_input_shape != entry.input_shape) {
                    append_message(
                        entry.message,
                        "Expected input " + shape_to_string(entry.expected_input_shape) + " but received " +
                            shape_to_string(entry.input_shape) + '.');
                }
                success = false;
                report.layers.push_back(std::move(entry));
                break;
            } catch (...) {
                entry.ok = false;
                entry.message = "Unknown error during forward propagation.";
                assign_expected_output(entry, layer, nullptr);
                if (!entry.expected_input_shape.empty() && entry.expected_input_shape != entry.input_shape) {
                    append_message(
                        entry.message,
                        "Expected input " + shape_to_string(entry.expected_input_shape) + " but received " +
                            shape_to_string(entry.input_shape) + '.');
                }
                success = false;
                report.layers.push_back(std::move(entry));
                break;
            }

            report.layers.push_back(std::move(entry));
        }

        if (success && report.layers.size() != layers_.size()) {
            success = false;
        }

        if (!success) {
            report.backward_ok = false;
            if (report.backward_message.empty()) {
                report.backward_message = "Backward pass skipped because forward diagnostics failed.";
            }
        } else {
            try {
                auto loss = current;
                if (loss.dim() != 0) {
                    loss = loss.sum();
                }
                loss.backward();
                const_cast<Model*>(this)->zero_grad(true);
            } catch (const c10::Error& error) {
                report.backward_ok = false;
                report.backward_message = error.msg();
                success = false;
            } catch (const std::exception& error) {
                report.backward_ok = false;
                report.backward_message = error.what();
                success = false;
            } catch (...) {
                report.backward_ok = false;
                report.backward_message = "Unknown error during backward propagation.";
                success = false;
            }
        }

        report.ok = success && report.backward_ok;
        return report;
    }

    inline std::vector<int64_t> Model::tensor_shape_vector(const torch::Tensor& tensor)
    {
        return tensor.sizes().vec();
    }

    inline std::string Model::describe_activation(Activation::Type type)
    {
        switch (type) {
            case Activation::Type::Raw:
                return "Raw";
            case Activation::Type::ReLU:
                return "ReLU";
            case Activation::Type::Sigmoid:
                return "Sigmoid";
            case Activation::Type::Tanh:
                return "Tanh";
            case Activation::Type::LeakyReLU:
                return "LeakyReLU";
            case Activation::Type::Softmax:
                return "Softmax";
            case Activation::Type::SiLU:
                return "SiLU";
            case Activation::Type::GeLU:
                return "GeLU";
            case Activation::Type::GLU:
                return "GLU";
            case Activation::Type::SwiGLU:
                return "SwiGLU";
            case Activation::Type::dSiLU:
                return "dSiLU";
            case Activation::Type::PSiLU:
                return "PSiLU";
            case Activation::Type::Mish:
                return "Mish";
            case Activation::Type::Swish:
                return "Swish";
            case Activation::Type::Identity:
            default:
                return "Identity";
        }
    }

    inline std::string Model::describe_module(const Layer::Details::RegisteredLayer& layer)
    {
        if (!layer.module) {
            return "Functional layer";
        }

        if (auto sequential = std::dynamic_pointer_cast<ModelDetails::SequentialBlockModuleImpl>(layer.module)) {
            (void)sequential;
            return "Sequential block";
        }

        const auto& name = layer.module->name();
        if (!name.empty()) {
            return name;
        }

        return std::string{layer.module->name()};
    }
}
#endif //THOT_CHECK_HPP