#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "../include/Thot.h"

namespace {
    struct ECGDataset {
        torch::Tensor signals;
        torch::Tensor labels;
    };

    struct ECGDatasetSplit {
        ECGDataset train;
        ECGDataset validation;
    };

    std::string trim_copy(const std::string& value)
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

    std::vector<std::string> split_csv_line(const std::string& line, char delimiter = ';')
    {
        std::vector<std::string> tokens;
        std::string token;
        std::stringstream stream(line);
        while (std::getline(stream, token, delimiter)) {
            tokens.emplace_back(std::move(token));
        }
        if (!line.empty() && line.back() == delimiter) {
            tokens.emplace_back();
        }
        return tokens;
    }

    std::unordered_map<std::string, std::size_t> header_index_map(const std::vector<std::string>& header)
    {
        std::unordered_map<std::string, std::size_t> mapping;
        for (std::size_t index = 0; index < header.size(); ++index) {
            mapping[trim_copy(header[index])] = index;
        }
        return mapping;
    }

    std::string resolve_record_base(const std::filesystem::path& root,
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
        return candidate.lexically_normal().string();
    }

    float parse_float(const std::string& token)
    {
        try {
            return std::stof(token);
        } catch (...) {
            return std::numeric_limits<float>::quiet_NaN();
        }
    }

} // namespace

ECGDatasetSplit load_ptbxl_dataset(const std::string& root, bool low_res, float train_split)
{
    namespace fs = std::filesystem;
    const fs::path root_path = fs::path(root);
    const fs::path database_csv = root_path / "ptbxl_database.csv";
    const fs::path scp_csv = root_path / "scp_statements.csv";
    const fs::path records_dir = root_path / (low_res ? "records100" : "records500");
    std::ifstream scp_file(scp_csv);
    if (!scp_file) {
        throw std::runtime_error("Failed to open scp_statements.csv at " + scp_csv.string());
    }

    std::string header_line;
    if (!std::getline(scp_file, header_line)) {
        throw std::runtime_error("scp_statements.csv is empty");
    }
    const auto scp_header = split_csv_line(header_line);
    const auto scp_indices = header_index_map(scp_header);
    const auto code_it = scp_indices.find("scp_code");
    const auto diagnostic_flag_it = scp_indices.find("diagnostic");
    const auto diagnostic_class_it = scp_indices.find("diagnostic_class");
    if (code_it == scp_indices.end() || diagnostic_flag_it == scp_indices.end() || diagnostic_class_it == scp_indices.end()) {
        throw std::runtime_error("scp_statements.csv missing required columns");
    }
    std::unordered_map<std::string, std::string> scp_to_superclass;
    for (std::string row; std::getline(scp_file, row);) {
        if (row.empty()) {
            continue;
        }
        const auto fields = split_csv_line(row);
        if (fields.size() <= std::max({code_it->second, diagnostic_flag_it->second, diagnostic_class_it->second})) {
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
        static const std::unordered_map<std::string, int> allowed_classes{{"NORM", 0}, {"MI", 1}, {"STTC", 2}, {"CD", 3}, {"HYP", 4}};
        if (allowed_classes.find(diagnostic_class) == allowed_classes.end()) {
            continue;
        }
        scp_to_superclass[scp_code] = diagnostic_class;
    }

    std::ifstream database_file(database_csv);
    if (!database_file) {
        throw std::runtime_error("Failed to open ptbxl_database.csv at " + database_csv.string());
    }

    if (!std::getline(database_file, header_line)) {
        throw std::runtime_error("ptbxl_database.csv is empty");
    }
    const auto database_header = split_csv_line(header_line);
    const auto database_indices = header_index_map(database_header);
    const auto filename_key = low_res ? "filename_lr" : "filename_hr";
    const auto filename_it = database_indices.find(filename_key);
    const auto scp_codes_it = database_indices.find("scp_codes");
    const auto sampling_it = database_indices.find("sampling_frequency");
    if (filename_it == database_indices.end() || scp_codes_it == database_indices.end() || sampling_it == database_indices.end()) {
        throw std::runtime_error("ptbxl_database.csv missing required columns");
    }

    std::vector<torch::Tensor> signal_tensors;
    std::vector<std::int64_t> label_ids;
    std::optional<std::int64_t> expected_signal_length;

    static const std::unordered_map<std::string, std::int64_t> class_to_index{{"NORM", 0}, {"MI", 1}, {"STTC", 2}, {"CD", 3}, {"HYP", 4}};
    const std::regex code_weight_regex("'([^']+)'\\s*:\\s*([0-9]*\\.?[0-9]+)");

    for (std::string row; std::getline(database_file, row);) {
        if (row.empty()) {
            continue;
        }
        const auto fields = split_csv_line(row);
        if (fields.size() <= std::max({filename_it->second, scp_codes_it->second, sampling_it->second})) {
            continue;
        }

        const auto sampling_frequency = parse_float(fields[sampling_it->second]);
        if (std::isnan(sampling_frequency)) {
            continue;
        }
        if ((low_res && std::llround(sampling_frequency) != 100) || (!low_res && std::llround(sampling_frequency) != 500)) {
            continue;
        }

        const auto base_path = resolve_record_base(root_path, records_dir, fields[filename_it->second]);
        if (base_path.empty()) {
            continue;
        }

        const std::string scp_codes_field = fields[scp_codes_it->second];
        std::unordered_map<std::string, float> superclass_votes;
        for (std::sregex_iterator match(scp_codes_field.begin(), scp_codes_field.end(), code_weight_regex); match != std::sregex_iterator(); ++match) {
            const auto scp_code = (*match)[1].str();
            const auto weight = parse_float((*match)[2].str());
            if (std::isnan(weight)) {
                continue;
            }
            const auto superclass_it = scp_to_superclass.find(scp_code);
            if (superclass_it == scp_to_superclass.end()) {
                continue;
            }
            superclass_votes[superclass_it->second] += weight;
        }

        if (superclass_votes.empty()) {
            continue;
        }

        const auto dominant_class = std::max_element(
            superclass_votes.begin(),
            superclass_votes.end(),
            [](const auto& left, const auto& right) { return left.second < right.second; })->first;

        const auto label_mapping = class_to_index.find(dominant_class);
        if (label_mapping == class_to_index.end()) {
            continue;
        }

        const fs::path hea_path = base_path + ".hea";
        const fs::path dat_path = base_path + ".dat";
        if (!fs::exists(hea_path) || !fs::exists(dat_path)) {
            continue;
        }

        std::ifstream header_stream(hea_path);
        if (!header_stream) {
            continue;
        }

        std::string wfdb_header;
        if (!std::getline(header_stream, wfdb_header)) {
            continue;
        }

        std::istringstream header_tokens(wfdb_header);
        std::string record_name;
        std::int64_t signal_count{};
        double frequency{};
        std::int64_t sample_count{};
        header_tokens >> record_name >> signal_count >> frequency >> sample_count;
        if (signal_count != 12) {
            continue;
        }
        if (sample_count <= 0) {
            continue;
        }

        const auto expected_bytes = static_cast<std::uintmax_t>(signal_count) * static_cast<std::uintmax_t>(sample_count) * sizeof(std::int16_t);
        std::error_code size_error;
        const auto dat_size = fs::file_size(dat_path, size_error);
        if (size_error || dat_size < expected_bytes) {
            continue;
        }

        std::ifstream dat_stream(dat_path, std::ios::binary);
        if (!dat_stream) {
            continue;
        }

        std::vector<std::int16_t> raw(static_cast<std::size_t>(signal_count * sample_count));
        dat_stream.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(std::int16_t)));
        if (dat_stream.gcount() != static_cast<std::streamsize>(raw.size() * sizeof(std::int16_t))) {
            continue;
        }

        auto tensor = torch::from_blob(raw.data(), {sample_count, signal_count}, torch::TensorOptions().dtype(torch::kInt16)).clone().to(torch::kFloat32);
        tensor = tensor.transpose(0, 1);
        auto mean = tensor.mean(1, true);
        auto std = tensor.std(1, true, true);
        std = torch::where(std < 1e-6, torch::ones_like(std), std);
        tensor = (tensor - mean) / std;

        if (!expected_signal_length) {
            expected_signal_length = tensor.size(1);
        } else if (*expected_signal_length != tensor.size(1)) {
            continue;
        }

        signal_tensors.push_back(std::move(tensor));
        label_ids.push_back(label_mapping->second);
    }
    if (signal_tensors.empty()) {
        throw std::runtime_error("No ECG records matched the selection criteria.");
    }
    auto signals = torch::stack(signal_tensors).to(torch::kFloat32);
    auto labels = torch::tensor(label_ids, torch::TensorOptions().dtype(torch::kInt64));

    const auto sample_count = signals.size(0);
    auto permutation = torch::randperm(sample_count, torch::TensorOptions().dtype(torch::kLong));
    signals = signals.index_select(0, permutation);
    labels = labels.index_select(0, permutation);

    train_split = std::clamp(train_split, 0.0f, 1.0f);
    std::int64_t train_count = static_cast<std::int64_t>(std::round(train_split * static_cast<float>(sample_count)));
    train_count = std::clamp<std::int64_t>(train_count, 1, sample_count);
    const std::int64_t validation_count = sample_count - train_count;

    auto train_signals = signals.narrow(0, 0, train_count).clone();
    auto train_labels = labels.narrow(0, 0, train_count).clone();

    torch::Tensor validation_signals;
    torch::Tensor validation_labels;
    if (validation_count > 0) {
        validation_signals = signals.narrow(0, train_count, validation_count).clone();
        validation_labels = labels.narrow(0, train_count, validation_count).clone();
    } else {
        validation_signals = torch::empty({0, signals.size(1), signals.size(2)}, signals.options());
        validation_labels = torch::empty({0}, labels.options());
    }

    return {{std::move(train_signals), std::move(train_labels)},
            {std::move(validation_signals), std::move(validation_labels)}};
}

int main()
{
    Thot::Model model("PTBXL_ECG");
    constexpr bool load_existing_model = false;
    const bool use_cuda = torch::cuda::is_available();
    std::cout << "Cuda: " << use_cuda << std::endl;
    model.to_device(use_cuda);

    const std::string dataset_root = "/home/moonfloww/Projects/DATASETS/physionet.org/files/ptb-xl/1.0.3";
    const auto dataset = load_ptbxl_dataset(dataset_root, true, 0.8f);

    const std::int64_t batch_size = 64;
    const std::int64_t epochs = 40;
    const std::int64_t num_classes = 5;
    const std::int64_t steps_per_epoch = std::max<std::int64_t>(1, (dataset.train.signals.size(0) + batch_size - 1) / batch_size);

    if (!load_existing_model) {
        model.add(Thot::Block::Sequential({
            Thot::Layer::Conv1d({12, 64, {7}, {2}, {3}}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
                      Thot::Layer::Conv1d({64, 64, {5}, {1}, {2}}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
                      Thot::Layer::MaxPool1d({{2}, {2}})
                  }),
                  "stem");

        model.add(Thot::Block::Sequential({
            Thot::Layer::Conv1d({64, 128, {5}, {2}, {2}}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
                      Thot::Layer::Conv1d({128, 128, {3}, {1}, {1}}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
                      Thot::Layer::MaxPool1d({{2}, {2}})
                  }),
                  "features1");

        model.add(Thot::Block::Sequential({
            Thot::Layer::Conv1d({128, 256, {3}, {1}, {1}}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
                      Thot::Layer::Conv1d({256, 256, {3}, {1}, {1}}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
                      Thot::Layer::MaxPool1d({{2}, {2}})
                  }),
                  "features2");
        model.add(Thot::Block::Sequential({
                      Thot::Layer::Conv1d({256, 256, {3}, {1}, {1}}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
                      Thot::Layer::AdaptiveAvgPool1d({{1}})
                  }),
                  "head");
        model.add(Thot::Layer::Flatten(), "flatten");
        model.add(Thot::Layer::FC({256, 128, true}, Thot::Activation::SiLU, Thot::Initialization::HeNormal), "fc1");
        model.add(Thot::Layer::HardDropout({.probability = 0.5}), "dropout");
        model.add(Thot::Layer::FC({128, num_classes, true}, Thot::Activation::Identity, Thot::Initialization::HeNormal), "classifier");

        model.links({
            Thot::LinkSpec{Thot::Port::parse("@input"), Thot::Port::parse("stem")},
            Thot::LinkSpec{Thot::Port::parse("stem"), Thot::Port::parse("features1")},
            Thot::LinkSpec{Thot::Port::parse("features1"), Thot::Port::parse("features2")},
            Thot::LinkSpec{Thot::Port::parse("features2"), Thot::Port::parse("head")},
            Thot::LinkSpec{Thot::Port::parse("head"), Thot::Port::parse("flatten")},
            Thot::LinkSpec{Thot::Port::parse("flatten"), Thot::Port::parse("fc1")},
            Thot::LinkSpec{Thot::Port::parse("fc1"), Thot::Port::parse("dropout")},
            Thot::LinkSpec{Thot::Port::parse("dropout"), Thot::Port::parse("classifier")},
            Thot::LinkSpec{Thot::Port::parse("classifier"), Thot::Port::parse("@output")}
        }, true);

        model.set_optimizer(
        Thot::Optimizer::AdamW({.learning_rate = 2e-4, .weight_decay = 1e-4}),
        Thot::LrScheduler::CosineAnnealing({
            .T_max = static_cast<std::size_t>(epochs) * static_cast<std::size_t>(steps_per_epoch),
            .eta_min = 1e-6,
            .warmup_steps = static_cast<std::size_t>(steps_per_epoch),
            .warmup_start_factor = 0.1
        }));

        model.set_loss(Thot::Loss::CrossEntropy({.label_smoothing = 0.01f}));
    }

    Thot::TrainOptions train_options{};
    train_options.epoch = static_cast<std::size_t>(epochs);
    train_options.batch_size = static_cast<std::size_t>(batch_size);
    train_options.shuffle = true;
    train_options.buffer_vram = 0;
    train_options.graph_mode = Thot::GraphMode::Capture;
    train_options.restore_best_state = true;
    train_options.enable_amp = true;
    train_options.memory_format = torch::MemoryFormat::Contiguous;
    train_options.test = std::make_pair(dataset.validation.signals, dataset.validation.labels);

    if (!load_existing_model) {
        model.train(dataset.train.signals, dataset.train.labels, train_options);
    }

    if (dataset.validation.signals.size(0) > 0) {
        model.evaluate(dataset.validation.signals,
                       dataset.validation.labels,
                       Thot::Evaluation::Classification,
                       {
                           Thot::Metric::Classification::Accuracy,
                           Thot::Metric::Classification::Precision,
                           Thot::Metric::Classification::Recall,
                           Thot::Metric::Classification::F1
                       },
                       {.batch_size = static_cast<std::size_t>(batch_size)});
    }

    return 0;
}