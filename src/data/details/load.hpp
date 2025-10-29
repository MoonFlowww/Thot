#ifndef THOT_LOAD_HPP
#define THOT_LOAD_HPP
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
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

        inline std::string trim(std::string value) {
            const auto not_space = [](unsigned char c) { return !std::isspace(c); };
            value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
            value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
            return value;
        }

        inline std::vector<float> parse_timeseries_csv(const fs::path& file_path, std::size_t expected_columns) {
            std::ifstream file(file_path);
            if (!file) {
                throw std::runtime_error("Failed to open timeseries CSV: " + file_path.string());
            }

            std::vector<float> samples;
            std::string line;
            std::size_t line_number = 0;
            while (std::getline(file, line)) {
                ++line_number;
                auto cleaned = trim(line);
                if (cleaned.empty()) {
                    continue;
                }

                std::vector<float> row_values;
                std::string cell;
                std::stringstream ss(cleaned);
                while (std::getline(ss, cell, ',')) {
                    cell = trim(cell);
                    if (cell.empty()) {
                        continue;
                    }
                    try {
                        row_values.push_back(std::stof(cell));
                    } catch (const std::exception&) {
                        throw std::runtime_error("Non-numeric value found in " + file_path.string() +
                                                 " at line " + std::to_string(line_number));
                    }
                }

                if (row_values.empty()) {
                    continue;
                }

                if (row_values.size() != expected_columns) {
                    throw std::runtime_error("Unexpected column count in " + file_path.string() +
                                             " at line " + std::to_string(line_number));
                }

                samples.insert(samples.end(), row_values.begin(), row_values.end());
            }

            if (samples.size() % expected_columns != 0) {
                throw std::runtime_error("Malformed CSV data in: " + file_path.string());
            }

            return samples;
        }

        struct PatientSignals {
            std::string identifier;
            std::vector<float> ecg;
            std::vector<float> accelerometer;
            std::optional<int64_t> target;

            [[nodiscard]] std::size_t sample_count() const {
                if (ecg.empty()) {
                    return 0;
                }
                return ecg.size();
            }
        };

        inline std::optional<int64_t> parse_patient_label(const fs::path& label_path) {
            if (!fs::exists(label_path)) {
                return std::nullopt;
            }

            std::ifstream file(label_path);
            if (!file) {
                throw std::runtime_error("Failed to open label file: " + label_path.string());
            }

            std::string line;
            while (std::getline(file, line)) {
                line = trim(line);
                if (line.empty()) {
                    continue;
                }

                try {
                    return static_cast<int64_t>(std::stoll(line));
                } catch (const std::exception&) {
                    throw std::runtime_error("Invalid label entry in: " + label_path.string());
                }
            }

            return std::nullopt;
        }

        inline PatientSignals load_patient_directory(const fs::path& patient_root) {
            const auto identifier = patient_root.filename().string();
            const auto ecg_path = patient_root / "ecg.csv";
            const auto acc_path = patient_root / "acc.csv";
            const auto label_path = patient_root / "label.txt";

            if (!fs::exists(ecg_path)) {
                throw std::runtime_error("Missing ecg.csv for patient: " + identifier);
            }
            if (!fs::exists(acc_path)) {
                throw std::runtime_error("Missing acc.csv for patient: " + identifier);
            }

            auto ecg = parse_timeseries_csv(ecg_path, 1);
            auto accelerometer = parse_timeseries_csv(acc_path, 3);

            if (ecg.size() != accelerometer.size() / 3) {
                throw std::runtime_error("Sample count mismatch between ECG and accelerometer data for patient: " +
                                         identifier);
            }

            PatientSignals patient{identifier, std::move(ecg), std::move(accelerometer), std::nullopt};
            patient.target = parse_patient_label(label_path);
            return patient;
        }

        inline torch::Tensor stack_patient_signals(std::vector<PatientSignals>& patients, bool normalise) {
            if (patients.empty()) {
                return torch::empty({0, 0, 4}, torch::TensorOptions().dtype(torch::kFloat32));
            }

            std::size_t max_samples = 0;
            for (const auto& patient : patients) {
                max_samples = std::max(max_samples, patient.sample_count());
            }

            std::vector<torch::Tensor> patient_tensors;
            patient_tensors.reserve(patients.size());

            for (auto& patient : patients) {
                const auto samples = patient.sample_count();
                std::vector<float> combined(max_samples * 4, 0.0f);

                for (std::size_t idx = 0; idx < std::min(samples, max_samples); ++idx) {
                    combined[idx * 4 + 0] = patient.ecg[idx];
                    combined[idx * 4 + 1] = patient.accelerometer[idx * 3 + 0];
                    combined[idx * 4 + 2] = patient.accelerometer[idx * 3 + 1];
                    combined[idx * 4 + 3] = patient.accelerometer[idx * 3 + 2];
                }

                auto tensor = torch::from_blob(
                    combined.data(),
                    {static_cast<int64_t>(max_samples), 4},
                    torch::TensorOptions().dtype(torch::kFloat32)).clone();
                patient_tensors.push_back(std::move(tensor));
            }

            auto stacked = torch::stack(patient_tensors, 0);

            if (normalise && stacked.numel() != 0) {
                auto flattened = stacked.view({stacked.size(0) * stacked.size(1), stacked.size(2)});
                auto mean = flattened.mean(0);
                auto std = flattened.std(0, /*unbiased=*/false).clamp_min(1e-6f);
                stacked = (stacked - mean.view({1, 1, stacked.size(2)})) /
                          std.view({1, 1, stacked.size(2)});
            }

            return stacked;
        }

        inline torch::Tensor stack_patient_labels(std::vector<PatientSignals>& patients) {
            if (patients.empty()) {
                return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64));
            }

            const bool all_have_labels = std::all_of(
                patients.begin(), patients.end(), [](const PatientSignals& patient) {
                    return patient.target.has_value();
                });

            if (!all_have_labels) {
                int64_t counter = 0;
                for (auto& patient : patients) {
                    patient.target = counter++;
                }
            }

            std::vector<int64_t> labels;
            labels.reserve(patients.size());
            for (const auto& patient : patients) {
                labels.push_back(patient.target.value());
            }

            torch::TensorOptions options;
            options = options.dtype(torch::kInt64);
            return torch::from_blob(labels.data(), {static_cast<int64_t>(labels.size())}, options).clone();
        }

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

    /**
     * Expected directory layout:
     * root/
     *   patient_000/
     *     ecg.csv           # single-column ECG readings
     *     acc.csv           # three-column accelerometer readings (x,y,z)
     *     label.txt         # optional integer target label
     *   patient_001/
     *     ...
     */
    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>>
    [[nodiscard]] inline std::pair<torch::Tensor, torch::Tensor>
    PatientTimeseries(const std::string& root, bool normalise = false) {
        const auto dataset_root = fs::path(root);
        if (!fs::exists(dataset_root) || !fs::is_directory(dataset_root)) {
            throw std::runtime_error("Provided timeseries root directory does not exist: " + root);
        }

        std::vector<Details::PatientSignals> patients;
        for (const auto& entry : fs::directory_iterator(dataset_root)) {
            if (!entry.is_directory()) {
                continue;
            }
            patients.push_back(Details::load_patient_directory(entry.path()));
        }

        if (patients.empty()) {
            return {
                torch::empty({0, 0, 4}, torch::TensorOptions().dtype(torch::kFloat32)),
                torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64))
            };
        }

        std::sort(patients.begin(), patients.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.identifier < rhs.identifier;
        });

        auto signals = Details::stack_patient_signals(patients, normalise);
        auto labels = Details::stack_patient_labels(patients);

        if constexpr (BufferVRAM) {
            const auto device = DevicePolicyT::select();
            signals = signals.to(device);
            labels = labels.to(device);
        }

        return {signals, labels};
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