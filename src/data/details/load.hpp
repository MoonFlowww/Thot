#ifndef THOT_LOAD_HPP
#define THOT_LOAD_HPP
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "../../core.hpp"

namespace Thot::Data::Load {
    namespace Details {
        namespace fs = std::filesystem;

        inline fs::path resolve_cifar10_root(const std::string& root) {
            auto candidate = fs::path(root);
            if (!fs::exists(candidate)) {
                throw std::runtime_error("Provided CIFAR-10 root directory does not exist: " + root);
            }

            if (fs::is_directory(candidate / "cifar-10-batches-bin")) {
                candidate /= "cifar-10-batches-bin";
            }

            if (!fs::exists(candidate / "data_batch_1.bin")) {
                throw std::runtime_error(
                    "Unable to locate CIFAR-10 binary batches in the provided root directory: " + candidate.string());
            }

            return candidate;
        }

        struct CIFAR10RawBatch {
            torch::Tensor inputs;
            torch::Tensor targets;
        };

        inline CIFAR10RawBatch read_cifar10_batch(const fs::path& file_path) {
            constexpr int64_t kChannels = 3;
            constexpr int64_t kHeight = 32;
            constexpr int64_t kWidth = 32;
            constexpr int64_t kImageSize = kChannels * kHeight * kWidth;
            constexpr int64_t kRecordSize = kImageSize + 1; // label byte + image bytes

            std::ifstream file(file_path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Failed to open CIFAR-10 batch file: " + file_path.string());
            }

            file.seekg(0, std::ios::end);
            const auto file_size = static_cast<std::size_t>(file.tellg());
            file.seekg(0, std::ios::beg);

            if (file_size % kRecordSize != 0) {
                throw std::runtime_error("Unexpected CIFAR-10 batch file size for: " + file_path.string());
            }

            const auto num_samples = static_cast<int64_t>(file_size / kRecordSize);
            std::vector<std::uint8_t> buffer(file_size);
            if (!file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()))) {
                throw std::runtime_error("Failed to read CIFAR-10 batch file: " + file_path.string());
            }

            auto inputs = torch::empty({num_samples, kChannels, kHeight, kWidth}, torch::kUInt8);
            auto targets = torch::empty({num_samples}, torch::kInt64);

            auto inputs_accessor = inputs.accessor<std::uint8_t, 4>();
            auto targets_accessor = targets.accessor<int64_t, 1>();

            for (int64_t sample_index = 0; sample_index < num_samples; ++sample_index) {
                const auto base_offset = static_cast<std::size_t>(sample_index) * kRecordSize;
                targets_accessor[sample_index] = static_cast<int64_t>(buffer[base_offset]);

                const auto* pixel_ptr = buffer.data() + base_offset + 1;
                for (int64_t channel = 0; channel < kChannels; ++channel) {
                    for (int64_t pixel = 0; pixel < kHeight * kWidth; ++pixel) {
                        const auto h = pixel / kWidth;
                        const auto w = pixel % kWidth;
                        inputs_accessor[sample_index][channel][h][w] = *(pixel_ptr++);
                    }
                }
            }

            return {inputs, targets};
        }

        inline std::pair<torch::Tensor, torch::Tensor> concatenate_batches(
            std::vector<CIFAR10RawBatch>& batches) {
            if (batches.empty()) {
                return {torch::empty({0, 3, 32, 32}), torch::empty({0}, torch::kLong)};
            }

            std::vector<torch::Tensor> inputs_list;
            std::vector<torch::Tensor> targets_list;
            inputs_list.reserve(batches.size());
            targets_list.reserve(batches.size());

            for (auto& batch : batches) {
                inputs_list.push_back(std::move(batch.inputs));
                targets_list.push_back(std::move(batch.targets));
            }

            auto inputs = torch::cat(inputs_list, 0);
            auto targets = torch::cat(targets_list, 0);
            return {inputs, targets};
        }

        template <class Tensor>
        Tensor apply_fraction(Tensor tensor, std::size_t count) {
            if (tensor.dim() == 0) {
                return tensor;
            }
            if (tensor.size(0) <= static_cast<int64_t>(count)) {
                return tensor;
            }
            const auto copy_count = static_cast<int64_t>(count);
            return tensor.narrow(0, 0, copy_count).clone();
        }

        inline torch::Tensor normalise_inputs(const torch::Tensor& tensor) {
            if (tensor.numel() == 0) {
                return tensor.to(torch::kFloat32);
            }
            return tensor.to(torch::kFloat32) / 255.0f;
        }
    }

    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>>
    [[nodiscard]] inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    CIFAR10(const std::string& root,
            float train_fraction = 1.0f,
            float test_fraction = 1.0f,
            bool normalise = true) {
        const auto dataset_root = Details::resolve_cifar10_root(root);

        const std::array<const char*, 5> train_files = {
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin"
        };

        std::vector<Details::CIFAR10RawBatch> train_batches;
        train_batches.reserve(train_files.size());
        for (const auto* filename : train_files) {
            train_batches.push_back(Details::read_cifar10_batch(dataset_root / filename));
        }

        auto test_batch = Details::read_cifar10_batch(dataset_root / "test_batch.bin");
        std::vector<Details::CIFAR10RawBatch> test_batches;
        test_batches.push_back(std::move(test_batch));

        auto [train_inputs, train_targets] = Details::concatenate_batches(train_batches);
        auto [test_inputs, test_targets] = Details::concatenate_batches(test_batches);

        const auto total_train = static_cast<std::size_t>(train_inputs.size(0));
        const auto total_test = static_cast<std::size_t>(test_inputs.size(0));

        const auto effective_train = std::clamp<std::size_t>(
            static_cast<std::size_t>(std::round(train_fraction * static_cast<float>(total_train))), 0, total_train);
        const auto effective_test = std::clamp<std::size_t>(
            static_cast<std::size_t>(std::round(test_fraction * static_cast<float>(total_test))), 0, total_test);

        train_inputs = Details::apply_fraction(std::move(train_inputs), effective_train);
        train_targets = Details::apply_fraction(std::move(train_targets), effective_train);
        test_inputs = Details::apply_fraction(std::move(test_inputs), effective_test);
        test_targets = Details::apply_fraction(std::move(test_targets), effective_test);

        if (normalise) {
            train_inputs = Details::normalise_inputs(train_inputs);
            test_inputs = Details::normalise_inputs(test_inputs);
        } else {
            train_inputs = train_inputs.to(torch::kFloat32);
            test_inputs = test_inputs.to(torch::kFloat32);
        }

        if constexpr (BufferVRAM) {
            const auto device = DevicePolicyT::select();
            train_inputs = train_inputs.to(device);
            train_targets = train_targets.to(device);
            test_inputs = test_inputs.to(device);
            test_targets = test_targets.to(device);
        }

        return {train_inputs, train_targets, test_inputs, test_targets};
    }
}

#endif //THOT_LOAD_HPP