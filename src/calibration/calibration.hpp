#ifndef THOT_CALIBRATION_HPP
#define THOT_CALIBRATION_HPP
// This file is an factory, must exempt it from any logical-code. For functions look into "/details"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <torch/torch.h>

#include "../utils/gnuplot.hpp"


namespace Thot::Calibration {
    template <class...>
    inline constexpr bool always_false_v = false;

    struct TemperatureScalingDescriptor {
        std::size_t max_iterations{50};
        double learning_rate{0.01};
    };

    using Descriptor = std::variant<TemperatureScalingDescriptor>;

    class Method {
    public:
        virtual ~Method() = default;

        virtual void attach(torch::nn::Module& model, const torch::Device& device) = 0;
        virtual void fit(const torch::Tensor& logits, const torch::Tensor& targets) = 0;
        virtual void plot(std::ostream& stream) const = 0;
        [[nodiscard]] virtual torch::Tensor transform(torch::Tensor logits) const = 0;
    };

    using MethodPtr = std::shared_ptr<Method>;

    struct Options {
        std::size_t reliability_bins{15};
        std::ostream* stream{nullptr};
    };


    namespace Details {
        struct ReliabilityBin {
            double confidence_sum{0.0};
            double accuracy_sum{0.0};
            std::size_t count{0};
        };

        inline std::size_t clamp_bin_index(double confidence, std::size_t bins)
        {
            if (bins == 0) {
                return 0;
            }

            const double clamped = std::clamp(confidence, 0.0, 1.0);
            auto index = static_cast<std::size_t>(clamped * static_cast<double>(bins));
            if (index >= bins) {
                index = bins - 1;
            }
            return index;
        }

        inline torch::Tensor normalise_targets(torch::Tensor targets, std::int64_t num_classes)
        {
            if (!targets.defined()) {
                throw std::invalid_argument("Calibration requires defined target tensor.");
            }

            auto prepared = targets.detach();
            if (prepared.device().type() != torch::kCPU) {
                prepared = prepared.to(torch::kCPU);
            }

            if (prepared.dim() > 1) {
                const auto last_dim = prepared.dim() - 1;
                if (prepared.size(last_dim) == num_classes) {
                    prepared = prepared.argmax(last_dim);
                } else if (prepared.size(last_dim) == 1) {
                    prepared = prepared.squeeze(last_dim);
                } else {
                    prepared = prepared.reshape({prepared.size(0)});
                }
            }

            if (prepared.dtype() != torch::kLong) {
                prepared = prepared.to(torch::kLong);
            }

            if (prepared.dim() != 1) {
                throw std::runtime_error("Calibration targets must be reducible to a one-dimensional tensor.");
            }

            return prepared.contiguous();
        }

        inline std::vector<ReliabilityBin> compute_reliability_bins(torch::Tensor logits,
                                                                    torch::Tensor targets,
                                                                    std::size_t bin_count)
        {
            if (!logits.defined()) {
                throw std::invalid_argument("Calibration requires defined logits tensor.");
            }

            if (logits.dim() == 1) {
                logits = logits.unsqueeze(1);
            } else if (logits.dim() > 2) {
                logits = logits.reshape({logits.size(0), -1});
            }

            if (logits.size(0) == 0) {
                return {};
            }

            const auto classes = logits.size(1);
            auto prepared_targets = normalise_targets(std::move(targets), classes);
            if (prepared_targets.size(0) != logits.size(0)) {
                throw std::runtime_error("Calibration logits and targets must share the same number of samples.");
            }

            auto logits_cpu = logits.detach().to(torch::kCPU, torch::kFloat64).contiguous();
            auto probabilities = torch::softmax(logits_cpu, 1);
            auto max_result = probabilities.max(1);
            auto confidences = std::get<0>(max_result).to(torch::kCPU).contiguous();
            auto predictions = std::get<1>(max_result).to(torch::kCPU, torch::kLong).contiguous();

            const auto total = confidences.size(0);
            const auto* confidence_ptr = confidences.data_ptr<double>();
            const auto* prediction_ptr = predictions.data_ptr<long>();
            const auto* target_ptr = prepared_targets.data_ptr<long>();

            const std::size_t bins = std::max<std::size_t>(1, bin_count);
            std::vector<ReliabilityBin> bucket(bins);

            for (std::int64_t idx = 0; idx < total; ++idx) {
                const double confidence = confidence_ptr[idx];
                const std::size_t bin_index = clamp_bin_index(confidence, bins);
                auto& entry = bucket[bin_index];
                entry.count += 1;
                entry.confidence_sum += confidence;
                entry.accuracy_sum += (prediction_ptr[idx] == target_ptr[idx]) ? 1.0 : 0.0;
            }

            return bucket;
        }

        inline void plot_reliability_diagram(const Method& method,
                                             const std::pair<torch::Tensor, torch::Tensor>& validation,
                                             const Options& options,
                                             const torch::Device& device) {
            auto logits = validation.first;
            auto targets = validation.second;

            if (!logits.defined() || !targets.defined()) {
                throw std::invalid_argument("Validation logits and targets must be defined for reliability plotting.");
            }
            auto raw_bins = compute_reliability_bins(logits, targets.clone(), options.reliability_bins);

            auto device_logits = logits;
            if (device_logits.device() != device) {
                device_logits = device_logits.to(device);
            }

            torch::NoGradGuard guard;
            auto transformed = method.transform(std::move(device_logits)).detach().to(torch::kCPU);

            auto calibrated_bins = compute_reliability_bins(std::move(transformed), std::move(targets), options.reliability_bins);


            struct ReliabilityStats {
                std::vector<double> confidence_points{};
                std::vector<double> accuracy_points{};
                double ece{0.0};
                double mce{0.0};
                std::size_t total_samples{0};

                [[nodiscard]] bool has_points() const
                {
                    return !confidence_points.empty();
                }
            };

            const auto compute_stats = [](const std::vector<ReliabilityBin>& bins) {
                ReliabilityStats stats{};

                for (const auto& bin : bins) {
                    stats.total_samples += bin.count;
                }

                if (stats.total_samples == 0) {
                    return stats;
                }

                stats.confidence_points.reserve(bins.size());
                stats.accuracy_points.reserve(bins.size());

                for (const auto& bin : bins) {
                    if (bin.count == 0) {
                        continue;
                    }

                    const double avg_confidence = bin.confidence_sum / static_cast<double>(bin.count);
                    const double avg_accuracy = bin.accuracy_sum / static_cast<double>(bin.count);
                    stats.confidence_points.push_back(avg_confidence);
                    stats.accuracy_points.push_back(avg_accuracy);


                    const double difference = std::abs(avg_confidence - avg_accuracy);
                    stats.ece += difference * (static_cast<double>(bin.count) / static_cast<double>(stats.total_samples));
                    stats.mce = std::max(stats.mce, difference);
                }

                return stats;
            };

            auto raw_stats = compute_stats(raw_bins);
            auto calibrated_stats = compute_stats(calibrated_bins);

            if (calibrated_stats.total_samples == 0) {
                if (options.stream != nullptr) {
                    *options.stream << "Calibration: no validation samples available for reliability plotting." << '\n';
                }
                return;
            }

            if (options.stream != nullptr) {
                auto& out = *options.stream;
                const auto previous_flags = out.flags();
                const auto previous_precision = out.precision();
                out << std::fixed << std::setprecision(6)
                << "Calibration ECE: " << calibrated_stats.ece
                << ", MCE: " << calibrated_stats.mce << '\n';
                if (raw_stats.total_samples != 0 && raw_stats.has_points()) {
                    out << "Uncalibrated ECE: " << raw_stats.ece
                        << ", MCE: " << raw_stats.mce << '\n';
                    out.flags(previous_flags);
                    out.precision(previous_precision);
                }
                if (!calibrated_stats.has_points()) {
                    if (options.stream != nullptr) {
                        *options.stream << "Calibration: insufficient populated bins for reliability plot." << '\n';
                    }
                    return;
                }

                Utils::Gnuplot plotter{};
                plotter.setTitle("Reliability diagram");
                plotter.setXLabel("Confidence");
                plotter.setYLabel("Accuracy");
                plotter.command("set xrange [0:1]");
                plotter.command("set yrange [0:1]");
                plotter.command("set key top left");
                plotter.command("set grid");

                Utils::Gnuplot::PlotStyle diagonal_style{};
                diagonal_style.mode = Utils::Gnuplot::PlotMode::Lines;
                diagonal_style.lineWidth = 1.5;
                diagonal_style.lineColor = "rgb '#7f7f7f'";

                Utils::Gnuplot::PlotStyle model_style{};
                model_style.mode = Utils::Gnuplot::PlotMode::LinesPoints;
                model_style.lineWidth = 2.0;
                model_style.pointType = 7;
                model_style.pointSize = 1.2;
                model_style.lineColor = "rgb '#1f77b4'";

                Utils::Gnuplot::PlotStyle raw_model_style = model_style;
                raw_model_style.lineColor = "rgb '#d62728'";
                raw_model_style.pointType = 5;


                Utils::Gnuplot::DataSet2D diagonal{
                    std::vector<double>{0.0, 1.0},
                    std::vector<double>{0.0, 1.0},
                    "Perfect calibration",
                    diagonal_style
                };
                std::vector<Utils::Gnuplot::DataSet2D> datasets;
                datasets.reserve(3);
                datasets.push_back(std::move(diagonal));

                if (raw_stats.has_points()) {
                    datasets.push_back(Utils::Gnuplot::DataSet2D{
                        std::move(raw_stats.confidence_points),
                        std::move(raw_stats.accuracy_points),
                        "Uncalibrated model",
                        raw_model_style
                    });
                }

                Utils::Gnuplot::DataSet2D model_curve{
                    std::move(calibrated_stats.confidence_points),
                    std::move(calibrated_stats.accuracy_points),
                    "Calibrated model",
                    model_style
                };

                datasets.push_back(std::move(model_curve));

                plotter.plot(std::move(datasets));
            }
        }
    }

#include "details/temperature_scaling.hpp"


namespace Thot::Calibration {

    inline MethodPtr make_method(const Descriptor& descriptor)
    {
        return std::visit(
            [](const auto& concrete_descriptor) -> MethodPtr {
                using DescriptorType = std::decay_t<decltype(concrete_descriptor)>;
                if constexpr (std::is_same_v<DescriptorType, TemperatureScalingDescriptor>) {
                    return Details::make_temperature_scaling_method(concrete_descriptor);
                } else {
                    static_assert(always_false_v<DescriptorType>, "Unsupported calibration descriptor");
                }
            },
            descriptor);
    }

    using CalibrationDataCallback = std::function<std::pair<torch::Tensor, torch::Tensor>(torch::nn::Module&)>;

    inline MethodPtr Calibrate(torch::nn::Module& model,
                        const torch::Device& device,
                        const Descriptor& descriptor,
                        CalibrationDataCallback data_callback,
                        std::optional<std::pair<torch::Tensor, torch::Tensor>> validation = std::nullopt,
                        Options options = {},
                        bool plot = false) {
        if (!data_callback) {
            throw std::invalid_argument("Calibration requires a valid logits callback.");
        }

        auto method = make_method(descriptor);
        method->attach(model, device);
        auto calibration_pair = data_callback(model);
        auto& logits = calibration_pair.first;
        auto& targets = calibration_pair.second;
        if (!logits.defined() || !targets.defined()) {
            throw std::invalid_argument("Calibration callback returned undefined logits or targets.");
        }

        method->fit(logits, targets);

        if (options.stream != nullptr) {
            *options.stream << "Calibration method: ";
            method->plot(*options.stream);
            *options.stream << '\n';
        }

        if (plot) {
            if (!validation.has_value()) {
                if (options.stream != nullptr) {
                    *options.stream << "Calibration: validation data required for reliability plot." << '\n';
                }
            } else {
                try {
                    Details::plot_reliability_diagram(*method, *validation, options, device);
                } catch (const std::exception& ex) {
                    if (options.stream != nullptr) {
                        *options.stream << "Calibration: failed to plot reliability diagram: " << ex.what() << '\n';
                    }
                }
            }
        }
        return method;
    }

#endif //THOT_CALIBRATION_HPP