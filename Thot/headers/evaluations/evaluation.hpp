#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "../tensor.hpp"
#include "details/timeserie_eval.hpp"
#include "details/regression_eval.hpp"
#include "details/classification_eval.hpp"
#include "details/binary_eval.hpp"

namespace Thot {
    enum class Evaluation {
        Binary,
        Timeseries,
        Regression,
        Classification
    };

    namespace Evaluations {
        inline void evaluate(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets, const std::vector<float>& latencies,
            size_t flops, size_t input_size, size_t output_size, Evaluation type, size_t total_flops, size_t total_parm, bool verbose = false) {

            switch (type) {
            case Evaluation::Binary:
                ::Evaluations::evaluate_binary(predictions, targets, latencies, flops, input_size, output_size, total_flops, total_parm, verbose);
                break;
            case Evaluation::Timeseries:
                ::Evaluations::evaluate_timeseries(predictions, targets, latencies, flops, input_size, output_size, total_flops, total_parm, verbose);
                break;
            case Evaluation::Regression:
                    ::Evaluations::evaluate_regression(predictions, targets, latencies, flops, input_size, output_size, total_flops, total_parm, verbose);
                break;
            case Evaluation::Classification:
                ::Evaluations::evaluate_classification(predictions, targets, latencies, flops, input_size, output_size, total_flops, total_parm, verbose);
                break;
            }
        }

        inline std::string to_string(Evaluation eval) {
            switch (eval) {
            case Evaluation::Binary: return "Binary";
            case Evaluation::Timeseries: return "Timeseries";
            case Evaluation::Regression: return "Regression";
            case Evaluation::Classification: return "Classification";
            default: return "Unknown";
            }
        }
    }
}