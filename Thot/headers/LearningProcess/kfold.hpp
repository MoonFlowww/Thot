//
// Created by moonfloww on 28/08/2025.
//

#ifndef THOT_KFOLD_H
#define THOT_KFOLD_H
#pragma once

#include <vector>

namespace Thot {
namespace KFold {

class Classic {
private:
    int folds_;
public:
    Classic(int folds = 1) : folds_(folds) {}

    int get_folds() const { return folds_; }

    void split(
        const std::vector<std::vector<float>> &inputs,
        const std::vector<std::vector<float>> &targets,
        int fold,
        std::vector<std::vector<float>> &train_inputs,
        std::vector<std::vector<float>> &train_targets,
        std::vector<std::vector<float>> &val_inputs,
        std::vector<std::vector<float>> &val_targets) const {
        size_t fold_size = inputs.size() / folds_;
        size_t start_idx = fold * fold_size;
        size_t end_idx = (fold == folds_ - 1) ? inputs.size() : (fold + 1) * fold_size;

        train_inputs.clear();
        train_targets.clear();
        val_inputs.clear();
        val_targets.clear();

        for (size_t i = 0; i < inputs.size(); ++i) {
            if (i >= start_idx && i < end_idx) {
                val_inputs.push_back(inputs[i]);
                val_targets.push_back(targets[i]);
            } else {
                train_inputs.push_back(inputs[i]);
                train_targets.push_back(targets[i]);
            }
        }
    }


};

class Sequential {
private:
    int folds_;
public:
    Sequential(int folds = 1) : folds_(folds) {}

    int get_folds() const { return folds_; }

    void split(
    const std::vector<std::vector<float>> &inputs,
    const std::vector<std::vector<float>> &targets,
    int fold,
    std::vector<std::vector<float>> &train_inputs,
    std::vector<std::vector<float>> &train_targets,
    std::vector<std::vector<float>> &val_inputs,
    std::vector<std::vector<float>> &val_targets) const {
        size_t fold_size = inputs.size() / folds_;
        size_t start_idx = fold * fold_size;
        size_t end_idx = (fold == folds_ - 1) ? inputs.size() : (fold + 1) * fold_size;

        train_inputs.assign(inputs.begin(), inputs.begin() + start_idx);
        train_targets.assign(targets.begin(), targets.begin() + start_idx);
        val_inputs.assign(inputs.begin() + start_idx, inputs.begin() + end_idx);
        val_targets.assign(targets.begin() + start_idx, targets.begin() + end_idx);
    }


};

} // namespace KFold
} // namespace Thot


#endif //THOT_KFOLD_H