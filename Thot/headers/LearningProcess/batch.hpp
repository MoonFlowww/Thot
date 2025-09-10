#ifndef THOT_BATCH_H
#define THOT_BATCH_H
#pragma once

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../optimizations/optimizations.hpp"
#include "../tensor.hpp"

namespace Thot {
    namespace Batch {

        class Classic {
        private:
            int batch_size_;
            int epochs_;

        public:
            Classic(int batch_size = 1, int epochs = 1)
                : batch_size_(batch_size), epochs_(epochs) {}

            int get_epochs() const { return epochs_; }

            template <typename Net>
            float train_epoch(Net &net,
                              const std::vector<std::vector<float>> &inputs,
                              const std::vector<std::vector<float>> &targets,
                              const std::vector<std::vector<int>> &input_shapes,
                              int log_interval, bool verbose,
                              int current_epoch, int total_epochs, int fold = 0) const;

            template <typename Net>
            float train_epoch(Net &net,
                              const std::vector<std::vector<float>> &inputs,
                              const std::vector<std::vector<float>> &targets,
                              int log_interval, bool verbose,
                              int current_epoch, int total_epochs, int fold = 0) const;

        };

    } // namespace Batch
} // namespace Thot

namespace Thot {
    namespace Batch {

        template <typename Net>
        inline float Classic::train_epoch(Net &net,
                                          const std::vector<std::vector<float>> &inputs,
                                          const std::vector<std::vector<float>> &targets,
                                          const std::vector<std::vector<int>> &input_shapes,
                                          int log_interval,
                                          bool verbose,
                                          int current_epoch,
                                          int total_epochs,
                                          int fold) const {
            if (auto opt = net.get_optimizer()) {
                opt->step_lr(current_epoch, fold);
            }

            if (inputs.empty()) {
                throw std::logic_error("inputs is empty");
            }

            // Compute current pack boundaries and size
            const int pack_start_epoch = ((current_epoch - 1) / std::max(1, log_interval)) * std::max(1, log_interval) + 1;
            const int pack_end_epoch   = std::min(pack_start_epoch + std::max(1, log_interval) - 1, total_epochs);
            const int pack_size_epochs = pack_end_epoch - pack_start_epoch + 1;

            static thread_local auto interval_start = std::chrono::high_resolution_clock::now();
            if (current_epoch == pack_start_epoch) {
                interval_start = std::chrono::high_resolution_clock::now();
            }

            float total_loss = 0.0f;

            for (size_t b = 0; b < inputs.size(); b += batch_size_) {
                size_t end = std::min(b + static_cast<size_t>(batch_size_), inputs.size());
                for (size_t i = b; i < end; ++i) {
                    const std::vector<int> &input_shape = input_shapes[i];
                    std::vector<float> output = net.forward(inputs[i], input_shape);

                    Utils::Tensor prediction_tensor({1, static_cast<int>(output.size())});
                    prediction_tensor.upload(output);

                    Utils::Tensor target_tensor({1, static_cast<int>(targets[i].size())});
                    target_tensor.upload(targets[i]);

                    float loss = net.compute_loss(prediction_tensor, target_tensor);
                    total_loss += loss;

                    Utils::Tensor grad_tensor =
                        net.compute_gradients(prediction_tensor, target_tensor);

                    net.backward(std::move(grad_tensor));

                    if (verbose) {
                        auto now = std::chrono::high_resolution_clock::now();
                        double elapsed = std::chrono::duration<double>(now - interval_start).count();

                        const size_t batches_per_epoch = inputs.size(); // counting per-sample updates
                        const size_t total_batches = static_cast<size_t>(pack_size_epochs) * batches_per_epoch;
                        const size_t completed_batches =
                            static_cast<size_t>(current_epoch - pack_start_epoch) * batches_per_epoch + (i + 1);

                        double progress = total_batches ? static_cast<double>(completed_batches) / static_cast<double>(total_batches) : 0.0;
                        if (progress > 1.0) progress = 1.0;

                        double eta = (progress > 0.0)
                                         ? elapsed * (1.0 / progress - 1.0)
                                         : std::numeric_limits<double>::infinity();

                        std::ostringstream oss;
                        oss << std::fixed << std::setprecision(2);
                        oss << "\r[" << current_epoch << "] -> "
                            << "Progress: " << std::setw(3) << static_cast<int>(progress * 100.0) << "% | "
                            << "Elapsed: " << std::setw(6) << elapsed << "s | "
                            << "ETA: " << std::setw(6) << eta << "s ";
                        std::cout << oss.str() << std::flush;
                    }
                }
            }

            if (verbose) {
                std::cout << "\r" << std::string(80, ' ') << "\r" << std::flush;
            }

            return total_loss / static_cast<float>(inputs.size());
        }

        template <typename Net>
        inline float Classic::train_epoch(Net &net,
                                          const std::vector<std::vector<float>> &inputs,
                                          const std::vector<std::vector<float>> &targets,
                                          int log_interval,
                                          bool verbose,
                                          int current_epoch,
                                          int total_epochs,
                                          int fold) const {
            std::vector<std::vector<int>> input_shapes;
            input_shapes.reserve(inputs.size());
            for (const auto &input : inputs) {
                input_shapes.push_back({1, static_cast<int>(input.size())});
            }

            return this->train_epoch(net, inputs, targets, input_shapes,
                                     log_interval, verbose, current_epoch, total_epochs, fold);
        }

    } // namespace Batch
} // namespace Thot

#endif // THOT_BATCH_H
