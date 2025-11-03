#ifndef THOT_LOAD_HPP
#define THOT_LOAD_HPP
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <charconv>

#include <vector>
#include <optional>
#include <regex>
#include <sstream>
#include <unordered_map>

#include <torch/torch.h>

#include "../../core.hpp"

namespace Thot::Data::Load {
    namespace Details {
        namespace fs = std::filesystem;

        inline std::string trim_copy(const std::string& value)
        {
            const auto not_space = [](unsigned char character) { return !std::isspace(character); };
            auto first = std::find_if(value.begin(), value.end(), not_space);
            auto last = std::find_if(value.rbegin(), value.rend(), not_space).base();
            if (first >= last) {
                return {};
            }
            std::string result(first, last);
            while (!result.empty() && (result.front() == '"' || result.front() == '\'')) {
                result.erase(result.begin());
            }
            while (!result.empty() && (result.back() == '"' || result.back() == '\'')) {
                result.pop_back();
            }
            return result;
        }

        inline std::vector<std::string> split_csv_line(const std::string& line, char delimiter = ',')
        {
            std::vector<std::string> tokens;
            tokens.reserve(16);

            std::string cur;
            cur.reserve(128);
            bool in_quote = false;

            for (size_t i = 0; i < line.size(); ++i) {
                char c = line[i];

                if (c == '"') {
                    // handle escaped double quotes ("")
                    if (in_quote && i + 1 < line.size() && line[i + 1] == '"') {
                        cur.push_back('"');
                        ++i; // skip the escaped quote
                    } else {
                        in_quote = !in_quote;
                    }
                    continue;
                }

                if (!in_quote && c == delimiter) {
                    tokens.emplace_back(trim_copy(cur));
                    cur.clear();
                    continue;
                }

                cur.push_back(c);
            }

            // push last token
            tokens.emplace_back(trim_copy(cur));
            return tokens;
        }

        inline std::string strip_utf8_bom(std::string value)
        {
            constexpr unsigned char bom[] = {0xEF, 0xBB, 0xBF};
            if (value.size() >= 3 &&
                static_cast<unsigned char>(value[0]) == bom[0] &&
                static_cast<unsigned char>(value[1]) == bom[1] &&
                static_cast<unsigned char>(value[2]) == bom[2]) {
                value.erase(0, 3);
                }
            return value;
        }


        inline std::unordered_map<std::string, std::size_t> header_index_map(const std::vector<std::string>& header)
        {
            std::unordered_map<std::string, std::size_t> mapping;
            for (std::size_t index = 0; index < header.size(); ++index) {
                mapping[strip_utf8_bom(trim_copy(header[index]))] = index;
            }
            return mapping;
        }

        inline std::filesystem::path resolve_ptbxl_record_base(const std::filesystem::path& root,
                                                  const std::filesystem::path& preferred_records_dir,
                                                  const std::string& relative_field)
        {
            auto trimmed = trim_copy(relative_field);
            if (trimmed.empty()) {
                return {};
            }

            std::filesystem::path candidate(trimmed);
            if (candidate.is_relative()) {
                const auto first_component = candidate.begin() != candidate.end() ? *candidate.begin() : std::filesystem::path{};
                if (first_component != preferred_records_dir.filename()) {
                    candidate = preferred_records_dir / candidate;
                }
            }
            candidate = root / candidate;
            if (candidate.has_extension()) {
                candidate.replace_extension();
            }
            return candidate.lexically_normal();
        }


        inline std::optional<double> parse_float(const std::string& token)
        {
            const auto trimmed = trim_copy(token);
            if (trimmed.empty()) return std::nullopt;

            // fast, locale-independent parsing
            double value{};
            auto first = trimmed.data();
            auto last = trimmed.data() + trimmed.size();
            auto result = std::from_chars(first, last, value);

            if (result.ec == std::errc() && result.ptr == last) {
                return value;
            }

            // fallback to stod if from_chars didn't parse (some implementations had partial double support)
            try {
                std::size_t processed = 0;
                const double v = std::stod(trimmed, &processed);
                if (processed == 0) return std::nullopt;
                return v;
            } catch (...) {
                return std::nullopt;
            }
        }


        inline std::unordered_map<std::string, std::string> load_scp_superclass_map(
            const std::filesystem::path& scp_csv,
            const std::unordered_map<std::string, std::int64_t>& allowed_classes)
        {
            std::ifstream scp_file(scp_csv);
            if (!scp_file) {
                throw std::runtime_error("Failed to open scp_statements.csv at " + scp_csv.string());
            }

            std::string header_line;
            if (!std::getline(scp_file, header_line)) {
                throw std::runtime_error("scp_statements.csv is empty");
            }

            const auto header_tokens = split_csv_line(header_line);
            const auto indices = header_index_map(header_tokens);
            const auto code_it = indices.find("description");
            const auto diagnostic_flag_it = indices.find("diagnostic");
            const auto diagnostic_class_it = indices.find("diagnostic_class");
            if (code_it == indices.end() || diagnostic_flag_it == indices.end() || diagnostic_class_it == indices.end()) {
                throw std::runtime_error("scp_statements.csv missing required columns: description, diagnostic, diagnostic_class");
            }

            std::unordered_map<std::string, std::string> scp_to_superclass;
            std::string row;
            while (std::getline(scp_file, row)) {
                if (row.empty()) {
                    continue;
                }
                const auto fields = split_csv_line(row);
                if (fields.size() <=
                    std::max({code_it->second, diagnostic_flag_it->second, diagnostic_class_it->second})) {
                    continue;
                }

                const auto diagnostic_flag = trim_copy(fields[diagnostic_flag_it->second]);
                if (diagnostic_flag != "1") {
                    continue;
                }

                const auto scp_code = trim_copy(fields[code_it->second]);
                const auto diagnostic_class = trim_copy(fields[diagnostic_class_it->second]);
                if (scp_code.empty()) {
                    continue;
                }

                if (allowed_classes.find(diagnostic_class) == allowed_classes.end()) {
                    continue;
                }

                scp_to_superclass[scp_code] = diagnostic_class;
            }

            return scp_to_superclass;
        }

        inline std::unordered_map<std::string, float> extract_superclass_votes(
            const std::string& scp_codes_field,
            const std::unordered_map<std::string, std::string>& scp_to_superclass)
        {
            static const std::regex code_weight_regex("'([^']+)'\\s*:\\s*([0-9]*\\.?[0-9]+)");
            std::unordered_map<std::string, float> superclass_votes;

            for (std::sregex_iterator match(scp_codes_field.begin(), scp_codes_field.end(), code_weight_regex);
                 match != std::sregex_iterator();
                 ++match) {
                const auto scp_code = (*match)[1].str();
                const auto weight_token = (*match)[2].str();
                const auto weight = parse_float(weight_token);
                if (!weight) {
                    continue;
                }
                const auto superclass_it = scp_to_superclass.find(scp_code);
                if (superclass_it == scp_to_superclass.end()) {
                    continue;
                }
                superclass_votes[superclass_it->second] += static_cast<float>(*weight);
            }

            return superclass_votes;
        }

        inline std::optional<std::int64_t> select_superclass_label(
            const std::unordered_map<std::string, float>& superclass_votes,
            const std::unordered_map<std::string, std::int64_t>& class_to_index)
        {
            if (superclass_votes.empty()) {
                return std::nullopt;
            }

            const auto dominant = std::max_element(
                superclass_votes.begin(), superclass_votes.end(),
                [](const auto& left, const auto& right) { return left.second < right.second; });

            const auto label_it = class_to_index.find(dominant->first);
            if (label_it == class_to_index.end()) {
                return std::nullopt;
            }

            return label_it->second;
        }

        inline std::optional<torch::Tensor> read_ptbxl_signal(const std::filesystem::path& base_path,
                                                              std::optional<std::int64_t>& expected_length,
                                                              bool normalise)
        {
            std::filesystem::path hea_path = base_path;
            hea_path += ".hea";
            std::filesystem::path dat_path = base_path;
            dat_path += ".dat";

            if (!std::filesystem::exists(hea_path) || !std::filesystem::exists(dat_path)) {
                return std::nullopt;
            }

            std::ifstream header_stream(hea_path);
            if (!header_stream) {
                return std::nullopt;
            }

            std::string wfdb_header;
            if (!std::getline(header_stream, wfdb_header)) {
                return std::nullopt;
            }

            std::istringstream header_tokens(wfdb_header);
            std::string record_name;
            std::int64_t signal_count{};
            double frequency{};
            std::int64_t sample_count{};
            header_tokens >> record_name >> signal_count >> frequency >> sample_count;

            if (signal_count != 12 || sample_count <= 0) {
                return std::nullopt;
            }

            if (!expected_length) {
                expected_length = sample_count;
            } else if (*expected_length != sample_count) {
                return std::nullopt;
            }

            const auto expected_bytes = static_cast<std::uintmax_t>(signal_count) *
                                        static_cast<std::uintmax_t>(sample_count) * sizeof(std::int16_t);
            std::error_code size_error;
            const auto dat_size = std::filesystem::file_size(dat_path, size_error);
            if (size_error || dat_size < expected_bytes) {
                return std::nullopt;
            }

            std::ifstream dat_stream(dat_path, std::ios::binary);
            if (!dat_stream) {
                return std::nullopt;
            }

            std::vector<std::int16_t> raw(static_cast<std::size_t>(signal_count) *
                                          static_cast<std::size_t>(sample_count));
            dat_stream.read(reinterpret_cast<char*>(raw.data()),
                            static_cast<std::streamsize>(raw.size() * sizeof(std::int16_t)));
            if (dat_stream.gcount() != static_cast<std::streamsize>(raw.size() * sizeof(std::int16_t))) {
                return std::nullopt;
            }

            auto tensor = torch::from_blob(raw.data(), {sample_count, signal_count},
                                           torch::TensorOptions().dtype(torch::kInt16))
                               .clone()
                               .to(torch::kFloat32)
                               .transpose(0, 1);

            if (normalise) {
                auto mean = tensor.mean(1, true);
                auto std = tensor.std(1, true, true);
                std = torch::where(std < 1e-6, torch::ones_like(std), std);
                tensor = (tensor - mean) / std;
            }

            return tensor;
        }


        inline std::filesystem::path resolve_cifar10_root(const std::string& root) {
            auto candidate = std::filesystem::path(root);
            if (!std::filesystem::exists(candidate)) {
                throw std::runtime_error("Provided CIFAR-10 root directory does not exist: " + root);
            }

            if (std::filesystem::is_directory(candidate / "cifar-10-batches-bin")) {
                candidate /= "cifar-10-batches-bin";
            }

            if (!std::filesystem::exists(candidate / "data_batch_1.bin")) {
                throw std::runtime_error(
                    "Unable to locate CIFAR-10 binary batches in the provided root directory: " + candidate.string());
            }

            return candidate;
        }

        struct CIFAR10RawBatch {
            torch::Tensor inputs;
            torch::Tensor targets;
        };

        inline CIFAR10RawBatch read_cifar10_batch(const std::filesystem::path& file_path) {
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
        inline std::filesystem::path resolve_mnist_root(const std::string& root) {
            const std::array<std::filesystem::path, 3> candidates = {
                std::filesystem::path(root),
                std::filesystem::path(root) / "MNIST",
                std::filesystem::path(root) / "MNIST" / "raw"
            };

            const std::array<const char*, 4> required_files = {
                "train-images-idx3-ubyte",
                "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte",
                "t10k-labels-idx1-ubyte"
            };

            for (const auto& candidate : candidates) {
                if (!std::filesystem::exists(candidate)) {
                    continue;
                }

                const bool has_all_files = std::all_of(required_files.begin(), required_files.end(), [&](const char* file) {
                    return std::filesystem::exists(candidate / file);
                });

                if (has_all_files) {
                    return candidate;
                }
            }

            throw std::runtime_error("Unable to locate MNIST dataset in the provided root: " + root);
        }

        inline std::uint32_t read_big_endian_u32(std::ifstream& file, const std::filesystem::path& file_path, const char* context) {
            std::array<std::uint8_t, 4> buffer{};
            if (!file.read(reinterpret_cast<char*>(buffer.data()), 4)) {
                throw std::runtime_error("Failed to read " + std::string(context) + " from " + file_path.string());
            }

            return (static_cast<std::uint32_t>(buffer[0]) << 24U) |
                   (static_cast<std::uint32_t>(buffer[1]) << 16U) |
                   (static_cast<std::uint32_t>(buffer[2]) << 8U) |
                   static_cast<std::uint32_t>(buffer[3]);
        }

        inline torch::Tensor read_idx_images(const std::filesystem::path& file_path) {
            std::ifstream file(file_path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Failed to open MNIST image file: " + file_path.string());
            }

            const auto magic = read_big_endian_u32(file, file_path, "magic number");
            if (magic != 2051) {
                throw std::runtime_error("Unexpected MNIST image file magic number in " + file_path.string());
            }

            const auto count = static_cast<int64_t>(read_big_endian_u32(file, file_path, "image count"));
            const auto rows = static_cast<int64_t>(read_big_endian_u32(file, file_path, "rows"));
            const auto cols = static_cast<int64_t>(read_big_endian_u32(file, file_path, "columns"));

            if (rows != 28 || cols != 28) {
                throw std::runtime_error("MNIST image dimensions must be 28x28 in file: " + file_path.string());
            }

            if (count < 0) {
                throw std::runtime_error("Negative image count encountered in file: " + file_path.string());
            }

            const auto expected_size = static_cast<std::size_t>(count * rows * cols);
            std::vector<std::uint8_t> buffer(expected_size);
            if (!file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()))) {
                throw std::runtime_error("Failed to read MNIST image payload from: " + file_path.string());
            }

            if (static_cast<std::size_t>(file.gcount()) != expected_size) {
                throw std::runtime_error("MNIST image file truncated: " + file_path.string());
            }

            auto tensor = torch::empty({count, 1, rows, cols}, torch::kUInt8);
            std::memcpy(tensor.data_ptr<std::uint8_t>(), buffer.data(), buffer.size());
            return tensor;
        }

        inline torch::Tensor read_idx_labels(const std::filesystem::path& file_path) {
            std::ifstream file(file_path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Failed to open MNIST label file: " + file_path.string());
            }

            const auto magic = read_big_endian_u32(file, file_path, "magic number");
            if (magic != 2049) {
                throw std::runtime_error("Unexpected MNIST label file magic number in " + file_path.string());
            }

            const auto count = static_cast<int64_t>(read_big_endian_u32(file, file_path, "label count"));
            if (count < 0) {
                throw std::runtime_error("Negative label count encountered in file: " + file_path.string());
            }

            std::vector<std::uint8_t> buffer(static_cast<std::size_t>(count));
            if (!file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()))) {
                throw std::runtime_error("Failed to read MNIST label payload from: " + file_path.string());
            }

            if (static_cast<std::size_t>(file.gcount()) != buffer.size()) {
                throw std::runtime_error("MNIST label file truncated: " + file_path.string());
            }

            auto tensor = torch::empty({count}, torch::kInt64);
            auto accessor = tensor.accessor<int64_t, 1>();
            for (int64_t index = 0; index < count; ++index) {
                accessor[index] = static_cast<int64_t>(buffer[static_cast<std::size_t>(index)]);
            }

            return tensor;
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

    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>>
    [[nodiscard]] inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    MNIST(const std::string& root,
          float train_fraction = 1.0f,
          float test_fraction = 1.0f,
          bool normalise = true) {
        const auto dataset_root = Details::resolve_mnist_root(root);

        auto train_inputs = Details::read_idx_images(dataset_root / "train-images-idx3-ubyte");
        auto train_targets = Details::read_idx_labels(dataset_root / "train-labels-idx1-ubyte");
        auto test_inputs = Details::read_idx_images(dataset_root / "t10k-images-idx3-ubyte");
        auto test_targets = Details::read_idx_labels(dataset_root / "t10k-labels-idx1-ubyte");

        if (train_inputs.size(0) != train_targets.size(0)) {
            throw std::runtime_error("MNIST training images and labels count mismatch");
        }

        if (test_inputs.size(0) != test_targets.size(0)) {
            throw std::runtime_error("MNIST test images and labels count mismatch");
        }

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
            train_inputs = train_inputs.to(torch::kFloat32) / 255.0f;
            test_inputs = test_inputs.to(torch::kFloat32) / 255.0f;
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

    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>>
    [[nodiscard]] inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    PTBXL(const std::string& root,
          bool low_resolution,
          float train_fraction = 0.8f,
          bool normalise = true)
    {
        namespace fs = std::filesystem;
        const std::filesystem::path root_path = std::filesystem::path(root);
        const std::filesystem::path database_csv = root_path / "scp_statements.csv";
        const std::filesystem::path scp_csv = root_path / "scp_statements.csv";
        const std::filesystem::path records_dir = root_path / (low_resolution ? "records100" : "records500");

        static const std::unordered_map<std::string, std::int64_t> class_to_index{{"NORM", 0},
                                                                                 {"MI", 1},
                                                                                 {"STTC", 2},
                                                                                 {"CD", 3},
                                                                                 {"HYP", 4}};

        const auto scp_to_superclass = Details::load_scp_superclass_map(scp_csv, class_to_index);

        std::ifstream database_file(database_csv);
        if (!database_file) {
            throw std::runtime_error("Failed to open scp_statements.csv at " + database_csv.string());
        }

        std::string header_line;
        if (!std::getline(database_file, header_line)) {
            throw std::runtime_error("scp_statements.csv is empty");
        }

        const auto header_tokens = Details::split_csv_line(header_line);
        const auto header_indices = Details::header_index_map(header_tokens);
        const auto filename_key = low_resolution ? "filename_lr" : "filename_hr";
        const auto filename_it = header_indices.find(filename_key);
        const auto scp_codes_it = header_indices.find("scp_codes");
        const auto sampling_it = header_indices.find("sampling_frequency");
        if (filename_it == header_indices.end() ||
            scp_codes_it == header_indices.end() ||
            sampling_it == header_indices.end()) {
            throw std::runtime_error("scp_statements.csv missing required columns: description, diagnostic, diagnostic_class");
        }

        const auto expected_sampling_frequency = low_resolution ? 100LL : 500LL;

        std::vector<torch::Tensor> signal_tensors;
        std::vector<std::int64_t> label_ids;
        std::optional<std::int64_t> expected_signal_length;

        std::string row;
        while (std::getline(database_file, row)) {
            if (row.empty()) {
                continue;
            }

            const auto fields = Details::split_csv_line(row);
            if (fields.size() <=
                std::max({filename_it->second, scp_codes_it->second, sampling_it->second})) {
                continue;
            }

            const auto sampling_frequency_token = fields[sampling_it->second];
            const auto sampling_frequency_value = Details::parse_float(sampling_frequency_token);
            if (!sampling_frequency_value) {
                continue;
            }

            const auto sampling_frequency = static_cast<std::int64_t>(std::llround(*sampling_frequency_value));
            if (sampling_frequency != expected_sampling_frequency) {
                continue;
            }

            const auto base_path = Details::resolve_ptbxl_record_base(
                root_path, records_dir, fields[filename_it->second]);
            if (base_path.empty()) {
                continue;
            }

            const auto votes = Details::extract_superclass_votes(
                fields[scp_codes_it->second], scp_to_superclass);
            const auto label = Details::select_superclass_label(votes, class_to_index);
            if (!label) {
                continue;
            }

            auto signal = Details::read_ptbxl_signal(base_path, expected_signal_length, normalise);
            if (!signal) {
                continue;
            }

            signal_tensors.push_back(std::move(*signal));
            label_ids.push_back(*label);
        }

        if (signal_tensors.empty()) {
            throw std::runtime_error("No PTB-XL records matched the selection criteria.");
        }

        auto inputs = torch::stack(signal_tensors);
        auto targets = torch::tensor(label_ids, torch::TensorOptions().dtype(torch::kInt64));

        const auto sample_count = inputs.size(0);
        auto permutation = torch::randperm(sample_count, torch::TensorOptions().dtype(torch::kLong));
        inputs = inputs.index_select(0, permutation);
        targets = targets.index_select(0, permutation);

        train_fraction = std::clamp(train_fraction, 0.0f, 1.0f);
        auto train_count = static_cast<std::int64_t>(std::round(train_fraction * static_cast<float>(sample_count)));
        train_count = std::clamp<std::int64_t>(train_count, 1, sample_count);
        const auto validation_count = sample_count - train_count;

        auto train_inputs = inputs.narrow(0, 0, train_count).clone();
        auto train_targets = targets.narrow(0, 0, train_count).clone();

        torch::Tensor validation_inputs;
        torch::Tensor validation_targets;
        if (validation_count > 0) {
            validation_inputs = inputs.narrow(0, train_count, validation_count).clone();
            validation_targets = targets.narrow(0, train_count, validation_count).clone();
        } else {
            validation_inputs = torch::empty({0, inputs.size(1), inputs.size(2)}, inputs.options());
            validation_targets = torch::empty({0}, targets.options());
        }

        if (!normalise) {
            train_inputs = train_inputs.to(torch::kFloat32);
            validation_inputs = validation_inputs.to(torch::kFloat32);
        }

        if constexpr (BufferVRAM) {
            const auto device = DevicePolicyT::select();
            train_inputs = train_inputs.to(device);
            train_targets = train_targets.to(device);
            validation_inputs = validation_inputs.to(device);
            validation_targets = validation_targets.to(device);
        }

        return {train_inputs, train_targets, validation_inputs, validation_targets};
    }
}

#endif //THOT_LOAD_HPP