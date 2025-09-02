#ifndef THOT_BATCH_H
#define THOT_BATCH_H
#pragma once

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../tensor.hpp"

namespace Thot {

namespace Batch {

class Classic {
public:
    Classic(int batch_size = 1, int epochs = 1)
        : batch_size_(batch_size), epochs_(epochs) {}

    int get_epochs() const { return epochs_; }

    template <typename Net>
    float train_epoch(Net &net,
                    const std::vector<std::vector<float>> &inputs,
                    const std::vector<std::vector<float>> &targets,
                    int log_interval, bool verbose,
                    int current_epoch, int total_epochs) const;

private:
    int batch_size_;
    int epochs_;
};

} // namespace Batch
} // namespace Thot

namespace Thot {
namespace Batch {

template <typename Net>
    inline float Classic::train_epoch(
        Net &net,
        const std::vector<std::vector<float>> &inputs,
        const std::vector<std::vector<float>> &targets,
        int log_interval,
        bool verbose,
        int current_epoch,
        int total_epochs) const {

    float total_loss = 0.0f;
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t b = 0; b < inputs.size(); b += batch_size_) {
        size_t end = std::min(b + batch_size_, inputs.size());
        for (size_t i = b; i < end; ++i) {
            std::vector<int> input_shape = {1, static_cast<int>(inputs[i].size())};
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

            if (verbose && ((i + 1) % log_interval == 0 || i == inputs.size() - 1)) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start).count();
                double progress = (i + 1) / static_cast<double>(inputs.size());
                double eta = elapsed / progress - elapsed;

                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2);
                oss << "\r[" << current_epoch-1 << "] -> "
                    << "Progress: " << std::setw(3) << int(progress * 100)
                    << "% | "
                    << "Elapsed: " << std::setw(6) << elapsed << "s | "
                    << "ETA: " << std::setw(6) << eta << "s | "
                    ;

                std::cout << oss.str() << std::flush;
            }
        }
    }

    if (verbose) {
        std::cout << "\r" << std::string(80, ' ') << "\r" << std::flush;
    }

    if (inputs.empty()) {
        std::__throw_logic_error("inputs is empty");
        return 0.0f;
    }

    return total_loss / inputs.size();
}

} // namespace Batch
} // namespace Thot
#endif //THOT_BATCH_H