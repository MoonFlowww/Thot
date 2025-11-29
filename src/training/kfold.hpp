#ifndef OMNI_KFOLD_HPP
#define OMNI_KFOLD_HPP

#include <algorithm>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace Omni {
    namespace Kfold {
        struct ClassicOptions {
            int64_t folds{5};
            bool shuffle{true};
            std::optional<uint64_t> seed{};
        };

        struct StratifiedOptions {
            int64_t folds{5};
            bool shuffle{true};
            std::optional<uint64_t> seed{};
        };

        struct PurgedOptions {
            int64_t folds{5};
            int64_t purge_window{0};
        };

        namespace Detail {

            inline void validate_inputs(const torch::Tensor& inputs, const torch::Tensor& targets) {
                if (!inputs.defined() || !targets.defined())
                    throw std::invalid_argument("Input tensors must be defined for k-fold splitting.");
                if (inputs.dim() == 0 || targets.dim() == 0)
                    throw std::invalid_argument("Input tensors must have a batch dimension for k-fold splitting.");
                if (inputs.size(0) != targets.size(0))
                    throw std::invalid_argument("Input and target tensors must contain the same number of samples.");

            }

            inline void validate_fold_arguments(int64_t total_samples, int64_t folds) {
                if (folds <= 0)
                    throw std::invalid_argument("Number of folds must be greater than zero.");
                if (total_samples == 0)
                    throw std::invalid_argument("Cannot create folds from an empty dataset.");
                if (total_samples % folds != 0)
                    throw std::invalid_argument("Total number of samples must be divisible by the number of folds.");

            }

            inline std::vector<int64_t> range_indices(int64_t count) {
                std::vector<int64_t> indices(static_cast<std::size_t>(count));
                std::iota(indices.begin(), indices.end(), int64_t{0});
                return indices;
            }

            inline torch::Tensor vector_to_index_tensor(const std::vector<int64_t>& indices) {
                return torch::tensor(indices, torch::TensorOptions().dtype(torch::kLong));
            }

            inline torch::Tensor permute_and_view(torch::Tensor tensor, const torch::Tensor& permutation, int64_t folds, int64_t fold_size) {
                auto device = tensor.device();
                auto perm_on_device = permutation.to(device, torch::kLong);
                auto permuted = tensor.index_select(0, perm_on_device).contiguous();

                const auto original_sizes = tensor.sizes();
                std::vector<int64_t> new_shape;
                new_shape.reserve(static_cast<std::size_t>(tensor.dim()) + 1);
                new_shape.push_back(folds);
                new_shape.push_back(fold_size);
                new_shape.insert(new_shape.end(), original_sizes.begin() + 1, original_sizes.end());

                return permuted.view(new_shape);
            }

            inline std::mt19937_64 build_rng(const std::optional<uint64_t>& seed) {
                if (seed) {
                    return std::mt19937_64(*seed);
                }
                std::random_device rd;
                return std::mt19937_64(rd());
            }

            inline torch::Tensor build_classic_permutation(int64_t total_samples, bool shuffle, const std::optional<uint64_t>& seed) {
                auto indices = range_indices(total_samples);
                if (shuffle) {
                    auto rng = build_rng(seed);
                    std::shuffle(indices.begin(), indices.end(), rng);
                }
                return vector_to_index_tensor(indices);
            }

            inline torch::Tensor ensure_label_tensor(torch::Tensor targets) {
                if (targets.dim() == 1) {
                    return targets.to(torch::kLong);
                }

                if (targets.size(1) == 1) {
                    auto squeezed = targets.squeeze(1);
                    return squeezed.to(torch::kLong);
                }

                auto flattened = targets.reshape({targets.size(0), -1});
                auto label_info = std::get<1>(flattened.max(1, false));
                return label_info.to(torch::kLong);
            }

            inline std::vector<std::vector<int64_t>> build_stratified_folds(int64_t total_samples, int64_t fold_count, int64_t fold_size,
                                                                            torch::Tensor labels,
                                                                            bool shuffle, const std::optional<uint64_t>& seed) {
                auto labels_cpu = labels.to(torch::kCPU, torch::kLong).contiguous();
                const auto* label_ptr = labels_cpu.data_ptr<int64_t>();

                std::unordered_map<int64_t, std::vector<int64_t>> buckets;
                buckets.reserve(static_cast<std::size_t>(labels_cpu.numel()));
                for (int64_t idx = 0; idx < total_samples; ++idx) {
                    buckets[label_ptr[idx]].push_back(idx);
                }

                std::mt19937_64 rng{};
                if (shuffle) {
                    rng = build_rng(seed);
                }

                std::vector<std::vector<int64_t>> folds(static_cast<std::size_t>(fold_count));
                std::vector<int64_t> fold_counts(static_cast<std::size_t>(fold_count), 0);
                std::size_t cursor = 0;

                auto next_fold = [&]() -> std::size_t {
                    for (std::size_t attempts = 0; attempts < folds.size(); ++attempts) {
                        auto candidate = (cursor + attempts) % folds.size();
                        if (fold_counts[candidate] < fold_size) {
                            cursor = (candidate + 1) % folds.size();
                            return candidate;
                        }
                    }
                    throw std::logic_error("Unable to assign sample to a fold without exceeding capacity.");
                };

                for (auto& [label, bucket] : buckets) {
                    if (shuffle) {
                        std::shuffle(bucket.begin(), bucket.end(), rng);
                    }
                    for (auto index : bucket) {
                        auto fold_index = next_fold();
                        folds[fold_index].push_back(index);
                        ++fold_counts[fold_index];
                    }
                }

                for (const auto& fold : folds) {
                    if (static_cast<int64_t>(fold.size()) != fold_size) {
                        throw std::runtime_error("Stratified assignment produced uneven fold sizes.");
                    }
                }

                return folds;
            }

            inline torch::Tensor fold_indices_to_permutation(const std::vector<std::vector<int64_t>>& folds){
                std::vector<int64_t> ordering;
                ordering.reserve(std::accumulate(folds.begin(), folds.end(), std::size_t{0},
                                                 [](std::size_t acc, const auto& fold) { return acc + fold.size(); }));
                for (const auto& fold : folds) {
                    ordering.insert(ordering.end(), fold.begin(), fold.end());
                }
                return vector_to_index_tensor(ordering);
            }

            inline torch::Tensor build_purged_permutation(int64_t total_samples, int64_t folds, int64_t purge_window, int64_t& fold_size){
                if (purge_window < 0)
                    throw std::invalid_argument("purge_window must be non-negative.");
                if (folds <= 0)
                    throw std::invalid_argument("Number of folds must be greater than zero.");
                const auto effective_samples = total_samples - purge_window * (folds - 1);
                if (effective_samples <= 0)
                    throw std::invalid_argument("Not enough samples to satisfy the requested purge window and fold count.");
                if (effective_samples % folds != 0)
                    throw std::invalid_argument("Effective sample count must be divisible by the number of folds.");

                fold_size = effective_samples / folds;

                std::vector<int64_t> ordering;
                ordering.reserve(static_cast<std::size_t>(effective_samples));

                int64_t cursor = 0;
                for (int64_t fold = 0; fold < folds; ++fold) {
                    if (cursor + fold_size > total_samples) {
                        throw std::invalid_argument("Purge window and fold configuration exceed dataset length.");
                    }
                    for (int64_t idx = cursor; idx < cursor + fold_size; ++idx) {
                        ordering.push_back(idx);
                    }
                    cursor += fold_size + purge_window;
                }

                return vector_to_index_tensor(ordering);
            }

        }

        inline std::tuple<torch::Tensor, torch::Tensor> Classic(torch::Tensor inputs, torch::Tensor targets, const ClassicOptions& options = {}) {
            Detail::validate_inputs(inputs, targets);

            const auto total_samples = inputs.size(0);
            Detail::validate_fold_arguments(total_samples, options.folds);

            const auto fold_size = total_samples / options.folds;
            auto permutation = Detail::build_classic_permutation(total_samples, options.shuffle, options.seed);

            auto folded_inputs = Detail::permute_and_view(std::move(inputs), permutation, options.folds, fold_size);
            auto folded_targets = Detail::permute_and_view(std::move(targets), permutation, options.folds, fold_size);

            return {std::move(folded_inputs), std::move(folded_targets)};
        }

        inline std::tuple<torch::Tensor, torch::Tensor> Stratified(torch::Tensor inputs, torch::Tensor targets, const StratifiedOptions& options = {}) {
            Detail::validate_inputs(inputs, targets);

            const auto total_samples = inputs.size(0);
            Detail::validate_fold_arguments(total_samples, options.folds);
            const auto fold_size = total_samples / options.folds;

            auto labels = Detail::ensure_label_tensor(targets);
            auto fold_indices = Detail::build_stratified_folds(total_samples, options.folds, fold_size, labels,
                                                               options.shuffle, options.seed);
            auto permutation = Detail::fold_indices_to_permutation(fold_indices);

            auto folded_inputs = Detail::permute_and_view(std::move(inputs), permutation, options.folds, fold_size);
            auto folded_targets = Detail::permute_and_view(std::move(targets), permutation, options.folds, fold_size);

            return {std::move(folded_inputs), std::move(folded_targets)};
        }

        inline std::tuple<torch::Tensor, torch::Tensor> Purged(torch::Tensor inputs, torch::Tensor targets, const PurgedOptions& options = {}) {
            Detail::validate_inputs(inputs, targets);

            const auto total_samples = inputs.size(0);
            int64_t fold_size = 0;
            auto permutation = Detail::build_purged_permutation(total_samples, options.folds, options.purge_window, fold_size);

            auto folded_inputs = Detail::permute_and_view(std::move(inputs), permutation, options.folds, fold_size);
            auto folded_targets = Detail::permute_and_view(std::move(targets), permutation, options.folds, fold_size);

            return {std::move(folded_inputs), std::move(folded_targets)};
        }
    }
}

#endif //OMNI_KFOLD_HPP