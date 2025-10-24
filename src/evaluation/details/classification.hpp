#ifndef THOT_CLASSIFICATION_HPP
#define THOT_CLASSIFICATION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "../../metric/metric.hpp"
#include "../../utils/terminal.hpp"

namespace Thot::Evaluation::Details::Classification {

    struct Descriptor { };

    struct Options {
        std::size_t batch_size{0};
        std::size_t calibration_bins{15};
        bool print_summary{true};
        bool print_per_class{true};
        std::ostream* stream{&std::cout};
        Utils::Terminal::FrameStyle frame_style{Utils::Terminal::FrameStyle::Box};
    };

    struct Report {
        struct SummaryRow {
            Metric::Classification::Kind metric{};
            double macro{0.0};
            double weighted{0.0};
        };

        std::vector<Metric::Classification::Kind> order{};
        std::vector<SummaryRow> summary{};
        std::vector<std::int64_t> labels{};
        std::vector<std::size_t> support{};
        std::vector<std::vector<double>> per_class{};
        std::size_t total_samples{0};
    };

    namespace detail {
        inline double safe_div(double num, double den) {
            constexpr double kEps = 1e-12;
            return (std::abs(den) < kEps) ? 0.0 : (num / den);
        }

        inline std::size_t clamp_bin(double value, std::size_t bins) {
            if (bins == 0) {
                return 0;
            }
            const double clamped = std::clamp(value, 0.0, 1.0);
            if (clamped >= 1.0) {
                return bins - 1;
            }
            const double scaled = clamped * static_cast<double>(bins);
            std::size_t idx = static_cast<std::size_t>(scaled);
            if (idx >= bins) {
                idx = bins - 1;
            }
            return idx;
        }

        inline std::string_view metric_name(Metric::Classification::Kind kind) {
            using MetricKind = Metric::Classification::Kind;
            switch (kind) {
                case MetricKind::Accuracy: return "Accuracy";
                case MetricKind::Precision: return "Precision";
                case MetricKind::Recall: return "Recall";
                case MetricKind::F1: return "F1 score";
                case MetricKind::TruePositiveRate: return "True positive rate";
                case MetricKind::TrueNegativeRate: return "True negative rate";
                case MetricKind::Top1Accuracy: return "Top-1 accuracy";
                case MetricKind::ExpectedCalibrationError: return "Expected calibration error";
                case MetricKind::MaximumCalibrationError: return "Maximum calibration error";
                case MetricKind::CohensKappa: return "Cohen's kappa";
                case MetricKind::LogLoss: return "Log loss";
                case MetricKind::BrierScore: return "Brier score";
                default: return "Metric";
            }
        }

        inline std::string format_double(double value) {
            if (!std::isfinite(value)) {
                return "nan";
            }
            std::ostringstream out;
            out << std::fixed << std::setprecision(6) << value;
            return out.str();
        }

        inline std::string format_size(std::size_t value) {
            return std::to_string(value);
        }

        struct ClassCounts {
            std::size_t support{0};
            std::size_t predicted{0};
            std::size_t tp{0};
            std::size_t fp{0};
            std::size_t fn{0};
            std::size_t tn{0};
            double log_loss_sum{0.0};
            double brier_sum{0.0};
        };
    } // namespace detail

    template <class Container>
    [[nodiscard]] auto normalise_requests(const Container& metrics)
        -> std::vector<Metric::Classification::Kind>
    {
        std::vector<Metric::Classification::Kind> kinds;
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
                                const std::vector<Metric::Classification::Descriptor>& descriptors,
                                const Options& options = Options{}) -> Report
    {
        using MetricKind = Metric::Classification::Kind;

        auto metric_order = normalise_requests(descriptors);
        if (metric_order.empty()) {
            throw std::invalid_argument("At least one classification metric must be requested for evaluation.");
        }

        if (!inputs.defined() || !targets.defined()) {
            throw std::invalid_argument("Evaluation inputs and targets must be defined.");
        }

        if (inputs.size(0) != targets.size(0)) {
            throw std::invalid_argument("Evaluation inputs and targets must have the same number of samples.");
        }

        Report report{};
        report.order = metric_order;
        report.total_samples = static_cast<std::size_t>(inputs.size(0));

        if (report.total_samples == 0) {
            return report;
        }

        const std::size_t total_samples = report.total_samples;
        const std::size_t batch_size = options.batch_size > 0 ? options.batch_size : total_samples;

        bool needs_probabilities = false;
        bool needs_calibration = false;
        bool needs_log_loss = false;
        bool needs_brier = false;

        for (auto kind : metric_order) {
            switch (kind) {
                case MetricKind::ExpectedCalibrationError:
                case MetricKind::MaximumCalibrationError:
                    needs_probabilities = true;
                    needs_calibration = true;
                    break;
                case MetricKind::LogLoss:
                    needs_probabilities = true;
                    needs_log_loss = true;
                    break;
                case MetricKind::BrierScore:
                    needs_probabilities = true;
                    needs_brier = true;
                    break;
                default:
                    break;
            }
        }

        torch::NoGradGuard guard;
        const bool was_training = model.is_training();
        model.eval();

        std::size_t num_classes = 0;
        std::vector<detail::ClassCounts> class_counts;
        std::vector<std::int64_t> class_labels;

        std::vector<std::vector<std::size_t>> class_bin_counts;
        std::vector<std::vector<double>> class_bin_confidence;
        std::vector<std::vector<double>> class_bin_correct;
        std::vector<std::size_t> global_bin_counts;
        std::vector<double> global_bin_confidence;
        std::vector<double> global_bin_correct;

        double total_correct = 0.0;
        double total_log_loss = 0.0;
        double total_brier = 0.0;

        for (std::size_t start = 0; start < total_samples; start += batch_size) {
            const std::size_t remaining = total_samples - start;
            const std::size_t current_batch = std::min(batch_size, remaining);

            auto input_batch = inputs.slice(0, start, start + current_batch);
            auto target_batch = targets.slice(0, start, start + current_batch);

            auto predictions = model.forward(input_batch);
            auto logits = predictions.detach();

            if (logits.dim() == 1) {
                logits = logits.unsqueeze(1);
            } else if (logits.dim() > 2) {
                logits = logits.reshape({logits.size(0), -1});
            }

            logits = logits.to(torch::kCPU, torch::kFloat64);
            if (logits.dim() != 2) {
                throw std::runtime_error("Classification evaluation expects two-dimensional logits.");
            }

            const std::size_t batch_classes = static_cast<std::size_t>(logits.size(1));
            if (num_classes == 0) {
                num_classes = batch_classes;
                class_counts.assign(num_classes, {});
                class_labels.resize(num_classes);
                std::iota(class_labels.begin(), class_labels.end(), 0);

                if (needs_calibration) {
                    const auto bins = std::max<std::size_t>(1, options.calibration_bins);
                    class_bin_counts.assign(num_classes, std::vector<std::size_t>(bins, 0));
                    class_bin_confidence.assign(num_classes, std::vector<double>(bins, 0.0));
                    class_bin_correct.assign(num_classes, std::vector<double>(bins, 0.0));

                    global_bin_counts.assign(bins, 0);
                    global_bin_confidence.assign(bins, 0.0);
                    global_bin_correct.assign(bins, 0.0);
                }
            } else if (num_classes != batch_classes) {
                throw std::runtime_error("Inconsistent number of classes encountered during evaluation.");
            }

            auto predicted = logits.argmax(1);
            predicted = predicted.to(torch::kCPU, torch::kLong);

            torch::Tensor probabilities;
            double* probabilities_ptr = nullptr;
            std::int64_t probability_stride = 0;
            if (needs_probabilities) {
                probabilities = torch::softmax(logits, 1).contiguous().to(torch::kDouble);
                probabilities_ptr = probabilities.data_ptr<double>();
                probability_stride = probabilities.size(1);
            }

            auto target_cpu = target_batch.to(torch::kCPU);
            if (target_cpu.dtype() == torch::kFloat32 || target_cpu.dtype() == torch::kFloat64) {
                if (target_cpu.dim() > 1 && target_cpu.size(1) == static_cast<long>(num_classes)) {
                    target_cpu = target_cpu.argmax(1);
                } else {
                    target_cpu = target_cpu.to(torch::kLong);
                }
            } else if (target_cpu.dtype() != torch::kLong) {
                target_cpu = target_cpu.to(torch::kLong);
            }

            if (target_cpu.dim() > 1) {
                if (target_cpu.size(1) == 1) {
                    target_cpu = target_cpu.squeeze(1);
                } else {
                    target_cpu = target_cpu.reshape({target_cpu.size(0)});
                }
            }

            if (target_cpu.sizes() != predicted.sizes()) {
                throw std::runtime_error("Evaluation targets and predictions must share the same leading shape.");
            }

            auto pred_accessor = predicted.accessor<long, 1>();
            auto target_accessor = target_cpu.accessor<long, 1>();

            for (std::size_t i = 0; i < current_batch; ++i) {
                const auto label = target_accessor[i];
                const auto pred = pred_accessor[i];

                if (label < 0 || label >= static_cast<long>(num_classes)) {
                    throw std::out_of_range("Encountered classification target outside the configured range.");
                }
                if (pred < 0 || pred >= static_cast<long>(num_classes)) {
                    throw std::out_of_range("Encountered classification prediction outside the configured range.");
                }

                auto& label_counts = class_counts[static_cast<std::size_t>(label)];
                auto& pred_counts = class_counts[static_cast<std::size_t>(pred)];

                label_counts.support += 1;
                pred_counts.predicted += 1;

                if (label == pred) {
                    label_counts.tp += 1;
                    total_correct += 1.0;
                } else {
                    label_counts.fn += 1;
                    pred_counts.fp += 1;
                }

                if (needs_probabilities) {
                    const double* row = probabilities_ptr + static_cast<std::size_t>(i) * probability_stride;
                    const auto label_index = static_cast<std::size_t>(label);

                    if (needs_log_loss) {
                        const double prob_true = std::clamp(row[label_index], 1e-12, 1.0);
                        const double loss = -std::log(prob_true);
                        label_counts.log_loss_sum += loss;
                        total_log_loss += loss;
                    }

                    if (needs_brier) {
                        double sample_brier = 0.0;
                        for (std::size_t cls = 0; cls < num_classes; ++cls) {
                            const double expected = (cls == label_index) ? 1.0 : 0.0;
                            const double diff = row[cls] - expected;
                            sample_brier += diff * diff;
                        }
                        label_counts.brier_sum += sample_brier;
                        total_brier += sample_brier;
                    }

                    if (needs_calibration) {
                        const auto bins = global_bin_counts.size();
                        const double confidence = std::clamp(row[static_cast<std::size_t>(pred)], 0.0, 1.0);
                        const auto bin = detail::clamp_bin(confidence, bins);
                        global_bin_counts[bin] += 1;
                        global_bin_confidence[bin] += confidence;
                        global_bin_correct[bin] += (label == pred) ? 1.0 : 0.0;

                        for (std::size_t cls = 0; cls < num_classes; ++cls) {
                            const double probability = std::clamp(row[cls], 0.0, 1.0);
                            const auto class_bin = detail::clamp_bin(probability, bins);
                            class_bin_counts[cls][class_bin] += 1;
                            class_bin_confidence[cls][class_bin] += probability;
                            class_bin_correct[cls][class_bin] += (cls == label_index) ? 1.0 : 0.0;
                        }
                    }
                }
            }
        }

        const double total_samples_d = static_cast<double>(total_samples);
        for (auto& counts : class_counts) {
            const std::size_t remainder = total_samples - (counts.tp + counts.fp + counts.fn);
            counts.tn = remainder;
        }

        std::vector<double> per_class_ece(num_classes, 0.0);
        std::vector<double> per_class_mce(num_classes, 0.0);
        double global_ece = 0.0;
        double global_mce = 0.0;

        if (needs_calibration) {
            const auto bins = global_bin_counts.size();
            for (std::size_t b = 0; b < bins; ++b) {
                const double count = static_cast<double>(global_bin_counts[b]);
                if (count <= 0.0) {
                    continue;
                }
                const double mean_confidence = global_bin_confidence[b] / count;
                const double accuracy = global_bin_correct[b] / count;
                const double diff = std::abs(mean_confidence - accuracy);
                global_ece += (count / total_samples_d) * diff;
                global_mce = std::max(global_mce, diff);
            }

            for (std::size_t cls = 0; cls < num_classes; ++cls) {
                double ece = 0.0;
                double mce = 0.0;
                for (std::size_t b = 0; b < bins; ++b) {
                    const double count = static_cast<double>(class_bin_counts[cls][b]);
                    if (count <= 0.0) {
                        continue;
                    }
                    const double mean_confidence = class_bin_confidence[cls][b] / count;
                    const double accuracy = class_bin_correct[cls][b] / count;
                    const double diff = std::abs(mean_confidence - accuracy);
                    ece += (count / total_samples_d) * diff;
                    mce = std::max(mce, diff);
                }
                per_class_ece[cls] = ece;
                per_class_mce[cls] = mce;
            }
        }

        std::vector<std::vector<double>> per_class_values(metric_order.size(), std::vector<double>(num_classes, 0.0));
        std::vector<Report::SummaryRow> summary;
        summary.reserve(metric_order.size());

        const double total_support = std::accumulate(
            class_counts.begin(), class_counts.end(), 0.0,
            [](double acc, const detail::ClassCounts& counts) {
                return acc + static_cast<double>(counts.support);
            });

        const double accuracy_global = detail::safe_div(total_correct, total_samples_d);
        const double top1_global = accuracy_global;

        double global_kappa = 0.0;
        {
            double pe = 0.0;
            for (const auto& counts : class_counts) {
                const double actual = detail::safe_div(static_cast<double>(counts.support), total_samples_d);
                const double predicted = detail::safe_div(static_cast<double>(counts.predicted), total_samples_d);
                pe += actual * predicted;
            }
            const double numerator = accuracy_global - pe;
            const double denominator = 1.0 - pe;
            global_kappa = detail::safe_div(numerator, denominator);
        }

        const double log_loss_global = needs_log_loss ? detail::safe_div(total_log_loss, total_samples_d) : 0.0;
        const double brier_global = needs_brier ? detail::safe_div(total_brier, total_samples_d) : 0.0;

        for (std::size_t metric_index = 0; metric_index < metric_order.size(); ++metric_index) {
            const auto kind = metric_order[metric_index];
            double macro_sum = 0.0;
            double macro_count = 0.0;
            double weighted_sum = 0.0;

            for (std::size_t cls = 0; cls < num_classes; ++cls) {
                const auto& counts = class_counts[cls];
                double value = 0.0;
                switch (kind) {
                    case MetricKind::Accuracy:
                    case MetricKind::Top1Accuracy: {
                        value = detail::safe_div(static_cast<double>(counts.tp + counts.tn), total_samples_d);
                        break;
                    }
                    case MetricKind::Precision: {
                        value = detail::safe_div(static_cast<double>(counts.tp),
                                                 static_cast<double>(counts.tp + counts.fp));
                        break;
                    }
                    case MetricKind::Recall:
                    case MetricKind::TruePositiveRate: {
                        value = detail::safe_div(static_cast<double>(counts.tp),
                                                 static_cast<double>(counts.tp + counts.fn));
                        break;
                    }
                    case MetricKind::F1: {
                        const double precision = detail::safe_div(static_cast<double>(counts.tp),
                                                                  static_cast<double>(counts.tp + counts.fp));
                        const double recall = detail::safe_div(static_cast<double>(counts.tp),
                                                               static_cast<double>(counts.tp + counts.fn));
                        const double denom = precision + recall;
                        value = (denom > 0.0) ? ((2.0 * precision * recall) / denom) : 0.0;
                        break;
                    }
                    case MetricKind::TrueNegativeRate: {
                        value = detail::safe_div(static_cast<double>(counts.tn),
                                                 static_cast<double>(counts.tn + counts.fp));
                        break;
                    }
                    case MetricKind::CohensKappa: {
                        const double po = detail::safe_div(static_cast<double>(counts.tp + counts.tn), total_samples_d);
                        const double pred_pos = detail::safe_div(static_cast<double>(counts.tp + counts.fp), total_samples_d);
                        const double actual_pos = detail::safe_div(static_cast<double>(counts.tp + counts.fn), total_samples_d);
                        const double pred_neg = 1.0 - pred_pos;
                        const double actual_neg = 1.0 - actual_pos;
                        const double pe = pred_pos * actual_pos + pred_neg * actual_neg;
                        value = detail::safe_div(po - pe, 1.0 - pe);
                        break;
                    }
                    case MetricKind::LogLoss: {
                        value = counts.support > 0
                            ? detail::safe_div(counts.log_loss_sum, static_cast<double>(counts.support))
                            : 0.0;
                        break;
                    }
                    case MetricKind::BrierScore: {
                        value = counts.support > 0
                            ? detail::safe_div(counts.brier_sum, static_cast<double>(counts.support))
                            : 0.0;
                        break;
                    }
                    case MetricKind::ExpectedCalibrationError: {
                        value = per_class_ece[cls];
                        break;
                    }
                    case MetricKind::MaximumCalibrationError: {
                        value = per_class_mce[cls];
                        break;
                    }
                }

                per_class_values[metric_index][cls] = value;

                if (std::isfinite(value)) {
                    macro_sum += value;
                    macro_count += 1.0;
                    weighted_sum += value * static_cast<double>(counts.support);
                }
            }

            double macro = (macro_count > 0.0) ? (macro_sum / macro_count) : 0.0;
            double weighted = (total_support > 0.0) ? (weighted_sum / total_support) : 0.0;

            switch (kind) {
                case MetricKind::Accuracy:
                    macro = accuracy_global;
                    weighted = accuracy_global;
                    break;
                case MetricKind::Top1Accuracy:
                    macro = top1_global;
                    weighted = top1_global;
                    break;
                case MetricKind::CohensKappa:
                    macro = global_kappa;
                    weighted = global_kappa;
                    break;
                case MetricKind::ExpectedCalibrationError:
                    macro = global_ece;
                    weighted = global_ece;
                    break;
                case MetricKind::MaximumCalibrationError:
                    macro = global_mce;
                    weighted = global_mce;
                    break;
                case MetricKind::LogLoss:
                    macro = log_loss_global;
                    weighted = log_loss_global;
                    break;
                case MetricKind::BrierScore:
                    macro = brier_global;
                    weighted = brier_global;
                    break;
                default:
                    break;
            }

            summary.push_back(Report::SummaryRow{kind, macro, weighted});
        }

        report.summary = std::move(summary);
        report.per_class = std::move(per_class_values);
        report.labels = std::move(class_labels);
        report.support.clear();
        report.support.reserve(class_counts.size());
        for (const auto& counts : class_counts) {
            report.support.push_back(counts.support);
        }

        if (was_training) {
            model.train();
        } else {
            model.eval();
        }

        if (options.stream && (options.print_summary || options.print_per_class)) {
            Print(report, options);
        }

        return report;
    }

    inline void Print(const Report& report, const Options& options) {
        if (!options.stream) {
            return;
        }

        auto& stream = *options.stream;
        const auto& metrics = report.order;
        const auto metric_count = metrics.size();

        using namespace Utils::Terminal;
        const auto color = Colors::kBrightBlue;

        auto make_spacing = [](std::size_t width) { return width + 2; };

        if (options.print_summary && !report.summary.empty()) {
            std::vector<std::string> metric_names;
            std::vector<std::string> macro_values;
            std::vector<std::string> weighted_values;
            metric_names.reserve(report.summary.size());
            macro_values.reserve(report.summary.size());
            weighted_values.reserve(report.summary.size());

            std::size_t metric_width = std::string("Metric").size();
            std::size_t macro_width = std::string("Macro").size();
            std::size_t weighted_width = std::string("Weighted (support)").size();

            for (const auto& row : report.summary) {
                auto name = std::string(detail::metric_name(row.metric));
                auto macro = detail::format_double(row.macro);
                auto weighted = detail::format_double(row.weighted);
                metric_width = std::max(metric_width, name.size());
                macro_width = std::max(macro_width, macro.size());
                weighted_width = std::max(weighted_width, weighted.size());
                metric_names.push_back(std::move(name));
                macro_values.push_back(std::move(macro));
                weighted_values.push_back(std::move(weighted));
            }

            const std::vector<std::size_t> spacings{
                make_spacing(metric_width),
                make_spacing(macro_width),
                make_spacing(weighted_width)
            };

            const auto top = HTop(spacings, color, options.frame_style);
            const auto mid = HMid(spacings, color);
            const auto bottom = HBottom(spacings, color, options.frame_style);

            auto print_row = [&](std::string_view c1, std::string_view c2, std::string_view c3) {
                std::ostringstream row;
                row << std::setfill(' ');
                row << Symbols::kBoxVertical << ' ';
                row << std::left << std::setw(static_cast<int>(metric_width)) << c1;
                row << std::right;
                row << ' ' << Symbols::kBoxVertical << ' ';
                row << std::setw(static_cast<int>(macro_width)) << c2;
                row << ' ' << Symbols::kBoxVertical << ' ';
                row << std::setw(static_cast<int>(weighted_width)) << c3;
                row << ' ' << Symbols::kBoxVertical;
                stream << row.str() << '\n';
            };

            stream << '\n' << top << '\n';
            print_row("Evaluation: Classification", "", "");
            stream << mid << '\n';
            print_row("Metric", "Macro", "Weighted (support)");
            stream << mid << '\n';
            for (std::size_t i = 0; i < report.summary.size(); ++i) {
                print_row(metric_names[i], macro_values[i], weighted_values[i]);
            }
            stream << bottom << '\n';
        }

        if (options.print_per_class && !report.per_class.empty() && !report.labels.empty()) {
            const auto num_classes = report.labels.size();

            std::size_t metric_width = std::string("Metric").size();
            metric_width = std::max(metric_width, std::string("Support").size());
            for (auto kind : metrics) {
                metric_width = std::max(metric_width, std::string(detail::metric_name(kind)).size());
            }

            std::vector<std::string> headers;
            std::vector<std::size_t> class_widths(num_classes, 0);
            headers.reserve(num_classes);
            for (std::size_t i = 0; i < num_classes; ++i) {
                auto label = std::string("Label ") + std::to_string(report.labels[i]);
                class_widths[i] = std::max(class_widths[i], label.size());
                headers.push_back(std::move(label));
            }

            std::vector<std::string> support_values;
            support_values.reserve(num_classes);
            for (std::size_t i = 0; i < num_classes; ++i) {
                auto value = detail::format_size(report.support[i]);
                class_widths[i] = std::max(class_widths[i], value.size());
                support_values.push_back(std::move(value));
            }

            for (std::size_t metric_index = 0; metric_index < metric_count && metric_index < report.per_class.size(); ++metric_index) {
                for (std::size_t cls = 0; cls < num_classes; ++cls) {
                    auto value = detail::format_double(report.per_class[metric_index][cls]);
                    class_widths[cls] = std::max(class_widths[cls], value.size());
                }
            }

            std::vector<std::size_t> spacings;
            spacings.reserve(num_classes + 1);
            spacings.push_back(make_spacing(metric_width));
            for (std::size_t i = 0; i < num_classes; ++i) {
                spacings.push_back(make_spacing(class_widths[i]));
            }

            const auto top = HTop(spacings, color, options.frame_style);
            const auto mid = HMid(spacings, color);
            const auto bottom = HBottom(spacings, color, options.frame_style);

            auto print_row = [&](std::string_view name, const std::vector<std::string>& values) {
                std::ostringstream row;
                row << std::setfill(' ');
                row << Symbols::kBoxVertical << ' ';
                row << std::left << std::setw(static_cast<int>(metric_width)) << name;
                row << std::right;
                for (std::size_t i = 0; i < num_classes; ++i) {
                    row << ' ' << Symbols::kBoxVertical << ' ';
                    row << std::setw(static_cast<int>(class_widths[i])) << values[i];
                }
                row << ' ' << Symbols::kBoxVertical;
                stream << row.str() << '\n';
            };

            stream << '\n' << top << '\n';
            print_row("Per-class metrics", std::vector<std::string>(num_classes));
            stream << mid << '\n';
            print_row("Metric", headers);
            stream << mid << '\n';
            print_row("Support", support_values);

            if (!report.per_class.empty()) {
                stream << mid << '\n';
                for (std::size_t metric_index = 0; metric_index < metric_count && metric_index < report.per_class.size(); ++metric_index) {
                    std::vector<std::string> values;
                    values.reserve(num_classes);
                    for (std::size_t cls = 0; cls < num_classes; ++cls) {
                        values.push_back(detail::format_double(report.per_class[metric_index][cls]));
                    }
                    print_row(std::string(detail::metric_name(metrics[metric_index])), values);
                    if (metric_index + 1 < metric_count && metric_index + 1 < report.per_class.size()) {
                        stream << mid << '\n';
                    }
                }
            }

            stream << bottom << '\n';
        }
    }
}

#endif //THOT_CLASSIFICATION_HPP