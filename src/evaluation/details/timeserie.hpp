#ifndef Nott_TIMESERIE_HPP
#define Nott_TIMESERIE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "../../common/streaming.hpp"
#include "../../metric/metric.hpp"
#include "../../utils/terminal.hpp"

namespace Nott::Evaluation::Details::Timeseries {
    struct Descriptor { };

    struct Options {
        std::size_t batch_size{8};
        std::size_t buffer_vram{0};
        bool print_summary{true};
        std::ostream* stream{&std::cout};
        Utils::Terminal::FrameStyle frame_style{Utils::Terminal::FrameStyle::Box};
    };

    struct Report {
        std::vector<Metric::Timeseries::Kind> order{};
        std::vector<double> values{};
        std::size_t total_series{0};
        std::size_t total_points{0};
    };

    namespace detail {
        inline double safe_div(double num, double den)
        {
            constexpr double kEps = 1e-12;
            return (std::abs(den) < kEps) ? std::numeric_limits<double>::quiet_NaN() : (num / den);
        }

        inline std::string format_double(double value)
        {
            if (!std::isfinite(value)) {
                return "nan";
            }
            std::ostringstream out;
            out << std::fixed << std::setprecision(6) << value;
            return out.str();
        }

        inline std::string_view metric_name(Metric::Timeseries::Kind kind)
        {
            using MetricKind = Metric::Timeseries::Kind;
            switch (kind) {
                case MetricKind::MeanAbsoluteError: return "Mean absolute error";
                case MetricKind::MeanAbsolutePercentageError: return "Mean absolute percentage error";
                case MetricKind::MeanBiasError: return "Mean bias error";
                case MetricKind::MeanSquaredError: return "Mean squared error";
                case MetricKind::MedianAbsoluteError: return "Median absolute error";
                case MetricKind::R2Score: return "R2 score";
                case MetricKind::RootMeanSquaredError: return "Root mean squared error";
                case MetricKind::SymmetricMeanAbsolutePercentageError: return "Symmetric mean absolute percentage error";
                case MetricKind::WeightedAbsolutePercentageError: return "Weighted absolute percentage error";
                case MetricKind::MeanPercentageError: return "Mean percentage error";
                case MetricKind::ExplainedVariance: return "Explained variance";
                case MetricKind::TheilsU1: return "Theil's U1";
                case MetricKind::TheilsU2: return "Theil's U2";
                case MetricKind::MeanAbsoluteScaledError: return "Mean absolute scaled error";
                case MetricKind::RootMeanSquaredScaledError: return "Root mean squared scaled error";
                case MetricKind::MedianRelativeAbsoluteError: return "Median relative absolute error";
                case MetricKind::GeometricMeanRelativeAbsoluteError: return "Geometric mean relative absolute error";
                case MetricKind::OverallWeightedAverage: return "Overall weighted average";
                case MetricKind::DynamicTimeWarpingDistance: return "Dynamic time warping distance";
                case MetricKind::TimeWarpEditDistance: return "Time warp edit distance";
                case MetricKind::SpectralDistance: return "Spectral distance";
                case MetricKind::CosineSimilarity: return "Cosine similarity";
                case MetricKind::NegativeLogLikelihood: return "Negative log-likelihood";
                case MetricKind::ContinuousRankedProbabilityScore: return "Continuous ranked probability score";
                case MetricKind::EnergyScore: return "Energy score";
                case MetricKind::PinballLossAverage: return "Pinball loss average";
                case MetricKind::BrierScore: return "Brier score";
                case MetricKind::PredictionIntervalCoverageProbability: return "Prediction interval coverage probability";
                case MetricKind::MeanPredictionIntervalWidth: return "Mean prediction interval width";
                case MetricKind::WinklerScore: return "Winkler score";
                case MetricKind::ConditionalCoverageError: return "Conditional coverage error";
                case MetricKind::QuantileCrossingRate: return "Quantile crossing rate";
                case MetricKind::AutocorrelationOfResiduals: return "Residual autocorrelation";
                case MetricKind::PartialAutocorrelationOfResiduals: return "Residual partial autocorrelation";
                case MetricKind::LjungBoxStatistic: return "Ljung-Box statistic";
                case MetricKind::BoxPierceStatistic: return "Box-Pierce statistic";
                case MetricKind::DurbinWatsonStatistic: return "Durbin-Watson statistic";
                case MetricKind::JarqueBeraStatistic: return "Jarque-Bera statistic";
                case MetricKind::AndersonDarlingStatistic: return "Anderson-Darling statistic";
                case MetricKind::BreuschPaganStatistic: return "Breusch-Pagan statistic";
                case MetricKind::WhiteStatistic: return "White statistic";
                case MetricKind::PopulationStabilityIndex: return "Population stability index";
                case MetricKind::KullbackLeiblerDivergence: return "Kullback-Leibler divergence";
                case MetricKind::JensenShannonDivergence: return "Jensen-Shannon divergence";
                case MetricKind::WassersteinDistance: return "Wasserstein distance";
                case MetricKind::MaximumMeanDiscrepancy: return "Maximum mean discrepancy";
                case MetricKind::LossDriftSlope: return "Loss drift slope";
                case MetricKind::LossCusumStatistic: return "Loss CUSUM statistic";
                case MetricKind::ResidualChangePointScore: return "Residual change point score";
                case MetricKind::QLIKE: return "QLIKE";
                case MetricKind::LogVarianceMeanSquaredError: return "Log variance mean squared error";
                case MetricKind::SqrtVarianceMeanSquaredError: return "Sqrt variance mean squared error";
                case MetricKind::msIC: return "msIC";
                case MetricKind::msIR: return "msIR";
            }
            return "Metric";
        }

        inline double median(std::vector<double> values)
        {
            if (values.empty()) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const auto mid = values.size() / 2;
            std::nth_element(values.begin(), values.begin() + mid, values.end());
            if (values.size() % 2 == 0) {
                const double a = values[mid];
                const double b = *std::max_element(values.begin(), values.begin() + mid);
                return 0.5 * (a + b);
            }
            return values[mid];
        }
    }

    void Print(const Report& report, const Options& options);

    template <class Container>
    [[nodiscard]] auto normalise_requests(const Container& metrics)
        -> std::vector<Metric::Timeseries::Kind>
    {
        std::vector<Metric::Timeseries::Kind> kinds;
        kinds.reserve(metrics.size());
        for (const auto& descriptor : metrics) {
            kinds.push_back(descriptor.kind);
        }
        return kinds;
    }

    template <class Model>
    [[nodiscard]] auto Evaluate(Model& model,
                                torch::Tensor inputs,
                                torch::Tensor targets,
                                const std::vector<Metric::Timeseries::Descriptor>& descriptors,
                                const Options& options) -> Report
    {
        auto metric_order = normalise_requests(descriptors);
        if (metric_order.empty()) {
            throw std::invalid_argument("At least one timeseries metric must be requested for evaluation.");
        }
        if (!inputs.defined() || !targets.defined()) {
            throw std::invalid_argument("Evaluation inputs and targets must be defined.");
        }
        if (inputs.size(0) != targets.size(0)) {
            throw std::invalid_argument("Evaluation inputs and targets must have the same number of samples.");
        }

        Report report{};
        report.order = metric_order;
        report.total_series = static_cast<std::size_t>(inputs.size(0));

        const auto device = model.device();
        const bool non_blocking_transfers = device.is_cuda();
        const bool device_supports_buffer = device.is_cuda();
        if (options.buffer_vram > 0 && !device_supports_buffer) {
            throw std::runtime_error("VRAM buffering during evaluation requires the model to be on a CUDA device.");
        }
        const bool use_buffer = options.buffer_vram > 0;

        const std::size_t batch_size = options.batch_size > 0
                                       ? options.batch_size
                                       : static_cast<std::size_t>(inputs.size(0));

        torch::NoGradGuard guard;
        const bool was_training = model.is_training();
        model.eval();

        if (use_buffer) {
            if (inputs.defined() && !inputs.device().is_cpu()) {
                inputs = inputs.to(torch::kCPU);
            }
            if (targets.defined() && !targets.device().is_cpu()) {
                targets = targets.to(torch::kCPU);
            }
        }

        Nott::StreamingOptions streaming_options{};
        if (batch_size > 0) {
            streaming_options.batch_size = batch_size;
        }
        if (use_buffer) {
            streaming_options.buffer_batches = options.buffer_vram + 1;
        }

        std::vector<double> predictions;
        std::vector<double> references;
        predictions.reserve(static_cast<std::size_t>(targets.numel()));
        references.reserve(static_cast<std::size_t>(targets.numel()));
        std::size_t total_points = 0;

        auto prepare_batch = [&](torch::Tensor input_batch, torch::Tensor target_batch)
            -> std::optional<Nott::StreamingBatch>
        {
            if (!input_batch.defined() || !target_batch.defined()) {
                return std::nullopt;
            }

            Nott::StreamingBatch batch{};
            batch.inputs = std::move(input_batch);
            batch.targets = std::move(target_batch);
            if (batch.targets.defined()) {
                batch.reference_targets = Nott::DeferredHostTensor::from_tensor(batch.targets, non_blocking_transfers);
            }

            return batch;
        };

        auto process_batch = [&](torch::Tensor prediction_tensor, Nott::StreamingBatch batch) {
            if (!prediction_tensor.defined()) {
                return;
            }

            auto preds = prediction_tensor.detach();
            if (!preds.device().is_cpu()) {
                if (non_blocking_transfers) {
                    auto deferred = Nott::DeferredHostTensor::from_tensor(preds, /*non_blocking=*/true);
                    preds = deferred.materialize();
                } else {
                    preds = preds.to(torch::kCPU, preds.scalar_type(), /*non_blocking=*/false);
                }
            }
            preds = preds.to(torch::kDouble).contiguous();
            auto preds_flat = preds.reshape({-1});

            torch::Tensor target_cpu;
            if (batch.reference_targets.defined()) {
                target_cpu = batch.reference_targets.materialize();
            } else {
                target_cpu = batch.targets;
            }

            if (!target_cpu.defined()) {
                return;
            }
            if (!target_cpu.device().is_cpu()) {
                if (non_blocking_transfers) {
                    auto deferred = Nott::DeferredHostTensor::from_tensor(target_cpu, /*non_blocking=*/true);
                    target_cpu = deferred.materialize();
                } else {
                    target_cpu = target_cpu.to(torch::kCPU, target_cpu.scalar_type(), /*non_blocking=*/false);
                }
            }
            target_cpu = target_cpu.to(torch::kDouble).contiguous();
            auto target_flat = target_cpu.reshape({-1});

            if (preds_flat.numel() != target_flat.numel()) {
                throw std::runtime_error("Timeseries evaluation expects predictions and targets to have the same number of elemen"
                                         "ts.");
            }

            const auto count = static_cast<std::size_t>(preds_flat.numel());
            const auto* pred_ptr = preds_flat.data_ptr<double>();
            const auto* ref_ptr = target_flat.data_ptr<double>();
            predictions.insert(predictions.end(), pred_ptr, pred_ptr + count);
            references.insert(references.end(), ref_ptr, ref_ptr + count);
            total_points += count;
        };

        model.stream_forward(std::move(inputs), std::move(targets), streaming_options, prepare_batch, process_batch);

        report.total_points = total_points;
        if (total_points == 0) {
            if (was_training) {
                model.train();
            } else {
                model.eval();
            }
            return report;
        }

        double sum_error = 0.0;
        double sum_abs_error = 0.0;
        double sum_squared_error = 0.0;
        double sum_target = 0.0;
        double sum_target_abs = 0.0;
        double sum_pred_abs = 0.0;

        std::vector<double> absolute_errors;
        absolute_errors.reserve(total_points);
        std::vector<double> relative_errors;
        relative_errors.reserve(total_points);

        std::size_t mape_count = 0;
        double mape_sum = 0.0;
        std::size_t mpe_count = 0;
        double mpe_sum = 0.0;
        std::size_t smape_count = 0;
        double smape_sum = 0.0;
        double log_ratio_sum = 0.0;
        std::size_t log_ratio_count = 0;

        for (std::size_t i = 0; i < total_points; ++i) {
            const double pred = predictions[i];
            const double target = references[i];
            const double error = pred - target;
            const double abs_error = std::abs(error);

            sum_error += error;
            sum_abs_error += abs_error;
            sum_squared_error += error * error;
            sum_target += target;
            sum_target_abs += std::abs(target);
            sum_pred_abs += std::abs(pred);

            absolute_errors.push_back(abs_error);

            const double denom = std::abs(target);
            if (denom > 1e-12) {
                mape_sum += abs_error / denom;
                mape_count += 1;
                mpe_sum += error / denom;
                mpe_count += 1;
                const double ratio = abs_error / denom;
                if (ratio > 0.0) {
                    log_ratio_sum += std::log(ratio);
                    log_ratio_count += 1;
                }
                relative_errors.push_back(ratio);
            } else {
                relative_errors.push_back(std::numeric_limits<double>::quiet_NaN());
            }

            const double smape_den = std::abs(target) + std::abs(pred);
            if (smape_den > 1e-12) {
                smape_sum += (2.0 * abs_error) / smape_den;
                smape_count += 1;
            }
        }

        const double n = static_cast<double>(total_points);
        const double mean_error = sum_error / n;
        const double mean_target = sum_target / n;

        double ss_tot = 0.0;
        double var_error_num = 0.0;
        for (std::size_t i = 0; i < total_points; ++i) {
            const double error = predictions[i] - references[i];
            const double target = references[i];
            ss_tot += (target - mean_target) * (target - mean_target);
            var_error_num += (error - mean_error) * (error - mean_error);
        }

        double mase_den = 0.0;
        double rmsse_den = 0.0;
        if (references.size() > 1) {
            for (std::size_t i = 1; i < references.size(); ++i) {
                const double diff = references[i] - references[i - 1];
                mase_den += std::abs(diff);
                rmsse_den += diff * diff;
            }
        }

        std::vector<double> metric_values;
        metric_values.reserve(metric_order.size());

        for (auto kind : metric_order) {
            double value = std::numeric_limits<double>::quiet_NaN();
            switch (kind) {
                case Metric::Timeseries::Kind::MeanAbsoluteError:
                    value = sum_abs_error / n;
                    break;
                case Metric::Timeseries::Kind::MeanSquaredError:
                    value = sum_squared_error / n;
                    break;
                case Metric::Timeseries::Kind::RootMeanSquaredError:
                    value = std::sqrt(sum_squared_error / n);
                    break;
                case Metric::Timeseries::Kind::MeanBiasError:
                    value = mean_error;
                    break;
                case Metric::Timeseries::Kind::MeanAbsolutePercentageError:
                    value = (mape_count > 0) ? (mape_sum / static_cast<double>(mape_count)) * 100.0
                                             : std::numeric_limits<double>::quiet_NaN();
                    break;
                case Metric::Timeseries::Kind::MeanPercentageError:
                    value = (mpe_count > 0) ? (mpe_sum / static_cast<double>(mpe_count)) * 100.0
                                            : std::numeric_limits<double>::quiet_NaN();
                    break;
                case Metric::Timeseries::Kind::SymmetricMeanAbsolutePercentageError:
                    value = (smape_count > 0) ? (smape_sum / static_cast<double>(smape_count)) * 100.0
                                              : std::numeric_limits<double>::quiet_NaN();
                    break;
                case Metric::Timeseries::Kind::WeightedAbsolutePercentageError:
                    value = detail::safe_div(sum_abs_error, sum_target_abs) * 100.0;
                    break;
                case Metric::Timeseries::Kind::MedianAbsoluteError:
                    value = detail::median(absolute_errors);
                    break;
                case Metric::Timeseries::Kind::MedianRelativeAbsoluteError: {
                    std::vector<double> filtered;
                    filtered.reserve(relative_errors.size());
                    for (double ratio : relative_errors) {
                        if (std::isfinite(ratio)) {
                            filtered.push_back(ratio);
                        }
                    }
                    value = detail::median(filtered);
                    break;
                }
                case Metric::Timeseries::Kind::GeometricMeanRelativeAbsoluteError:
                    value = (log_ratio_count > 0) ? std::exp(log_ratio_sum / static_cast<double>(log_ratio_count))
                                                  : std::numeric_limits<double>::quiet_NaN();
                    break;
                case Metric::Timeseries::Kind::MeanAbsoluteScaledError: {
                    const double denom = (references.size() > 1)
                        ? (mase_den / static_cast<double>(references.size() - 1))
                        : 0.0;
                    value = (denom > 1e-12) ? (sum_abs_error / n) / denom : std::numeric_limits<double>::quiet_NaN();
                    break;
                }
                case Metric::Timeseries::Kind::RootMeanSquaredScaledError: {
                    const double denom = (references.size() > 1)
                        ? (rmsse_den / static_cast<double>(references.size() - 1))
                        : 0.0;
                    value = (denom > 1e-12) ? std::sqrt((sum_squared_error / n) / denom)
                                             : std::numeric_limits<double>::quiet_NaN();
                    break;
                }
                case Metric::Timeseries::Kind::R2Score: {
                    const double ss_res = sum_squared_error;
                    if (std::abs(ss_tot) > 1e-12) {
                        value = 1.0 - (ss_res / ss_tot);
                    }
                    break;
                }
                case Metric::Timeseries::Kind::ExplainedVariance:
                    value = (std::abs(ss_tot) > 1e-12)
                        ? 1.0 - (var_error_num / ss_tot)
                        : std::numeric_limits<double>::quiet_NaN();
                    break;
                case Metric::Timeseries::Kind::BrierScore:
                    value = sum_squared_error / n;
                    break;
                default:
                    break;
            }
            metric_values.push_back(value);
        }

        report.values = std::move(metric_values);

        if (was_training) {
            model.train();
        } else {
            model.eval();
        }

        if (options.stream && options.print_summary) {
            Print(report, options);
        }

        return report;
    }

    inline void Print(const Report& report, const Options& options)
    {
        if (!options.stream) {
            return;
        }

        auto& stream = *options.stream;
        const auto& metrics = report.order;
        if (metrics.empty() || report.values.size() != metrics.size()) {
            return;
        }

        using namespace Utils::Terminal;
        const auto color = Colors::kBrightBlue;

        std::size_t metric_width = std::string("Metric").size();
        std::size_t value_width = std::string("Value").size();
        std::vector<std::string> metric_names;
        std::vector<std::string> value_strings;
        metric_names.reserve(metrics.size());
        value_strings.reserve(metrics.size());

        for (std::size_t i = 0; i < metrics.size(); ++i) {
            auto name = std::string(detail::metric_name(metrics[i]));
            auto value = detail::format_double(report.values[i]);
            metric_width = std::max(metric_width, name.size());
            value_width = std::max(value_width, value.size());
            metric_names.push_back(std::move(name));
            value_strings.push_back(std::move(value));
        }

        auto make_spacing = [](std::size_t width) { return width + 2; };
        const std::vector<std::size_t> spacings{
            make_spacing(metric_width),
            make_spacing(value_width)
        };

        const auto top = HTop(spacings, color, options.frame_style);
        const auto mid = HMid(spacings, color);
        const auto bottom = HBottom(spacings, color, options.frame_style);

        stream << '\n' << top << '\n';

        auto print_row = [&](std::string_view metric, std::string_view value) {
            std::ostringstream row;
            row << std::setfill(' ');
            row << Symbols::kBoxVertical << ' ';
            row << std::left << std::setw(static_cast<int>(metric_width)) << metric;
            row << std::right;
            row << ' ' << Symbols::kBoxVertical << ' ';
            row << std::setw(static_cast<int>(value_width)) << value;
            row << ' ' << Symbols::kBoxVertical;
            stream << row.str() << '\n';
        };

        print_row("Evaluation: Timeseries", "");
        stream << mid << '\n';
        print_row("Metric", "Value");
        stream << mid << '\n';
        for (std::size_t i = 0; i < metrics.size(); ++i) {
            print_row(metric_names[i], value_strings[i]);
        }
        stream << bottom << '\n';

        stream << "\nTotal sequences: " << report.total_series;
        stream << "\nTotal points: " << report.total_points << '\n';
    }
}

#endif //Nott_TIMESERIE_HPP