#pragma once

#include <algorithm>
#include <chrono>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <limits>

#include "layers/layers.hpp"
#include "tensor.hpp"

#include "activations/activations.hpp"
#include "evaluations/evaluation.hpp"
#include "losses/losses.hpp"
#include "metrics/metrics.hpp"
#include "optimizations/optimizations.hpp"

#include "utils/translators.hpp"

#include "LearningProcess/batch.hpp"
#include "LearningProcess/kfold.hpp"


class Layer;
class Optimizer;

namespace Thot {

    inline Activation activation_from_string(const std::string &name) {
        if (name == "Linear") return Activation::Linear;
        if (name == "ReLU") return Activation::ReLU;
        if (name == "Sigmoid") return Activation::Sigmoid;
        if (name == "Tanh") return Activation::Tanh;
        if (name == "LeakyReLU") return Activation::LeakyReLU;
        if (name == "ELU") return Activation::ELU;
        if (name == "GELU") return Activation::GELU;
        if (name == "Softmax") return Activation::Softmax;
        throw std::runtime_error("Unknown activation: " + name);
    }

class Network {
private:
    std::string name_;
    std::vector<std::shared_ptr<Layer>> layers_;
    bool Istraining_;
    std::shared_ptr<Optimizer> optimizer_;
    std::shared_ptr<Losses> loss_function_;
    std::mutex mutex_;

    std::vector<float> latencies_;
    std::vector<std::vector<float>> model_parameters_;

    size_t max_gpu_batches_;
    std::vector<cudaStream_t> cuda_streams_;

    void print_vector(const std::vector<float> &vec) {
        std::cout << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << vec[i];
            if (i < vec.size() - 1)
                std::cout << ", ";
        }
        std::cout << "]";
    }

    bool verify_layer_dimensions() {
        for (size_t i = 0; i + 1 < layers_.size(); ++i) {
            int expected = layers_[i]->get_output_size();
            int actual = layers_[i + 1]->get_input_size();
            if (expected > 0 && actual > 0 && expected != actual) {
                std::cout << "Layer dimension mismatch between layer n" << i
                          << " (" << layers_[i]->get_name() << ") output size "
                          << expected << " and layer n" << i + 1
                          << " (" << layers_[i + 1]->get_name() << ") input size "
                          << actual << std::endl;
                return false;
            }
        }
        return true;
    }


    using LayerParams = std::vector<std::vector<float>>;
    using ModelParams = std::vector<LayerParams>;

    ModelParams capture_parameters() const {
        ModelParams params;
        for (const auto &layer : layers_) {
            LayerParams lp;
            if (auto fc = std::dynamic_pointer_cast<FCLayer>(layer)) {
                lp.push_back(fc->weights_.download());
                lp.push_back(fc->bias_.download());
            } else if (auto conv = std::dynamic_pointer_cast<Conv2DLayer>(layer)) {
                lp.push_back(conv->weights_.download());
                lp.push_back(conv->bias_.download());
            } else if (auto rnn = std::dynamic_pointer_cast<RNNLayer>(layer)) {
                lp.push_back(rnn->weights_ih_.download());
                lp.push_back(rnn->weights_hh_.download());
                lp.push_back(rnn->bias_.download());
            } else if (auto rbm = std::dynamic_pointer_cast<RBMLayer>(layer)) {
                lp.push_back(rbm->weights_.download());
                lp.push_back(rbm->visible_bias_.download());
                lp.push_back(rbm->hidden_bias_.download());
            }
            params.push_back(std::move(lp));
        }
        return params;
    }

    void apply_parameters(const ModelParams &params) {
        for (size_t i = 0; i < layers_.size() && i < params.size(); ++i) {
            const LayerParams &lp = params[i];
            if (auto fc = std::dynamic_pointer_cast<FCLayer>(layers_[i])) {
                if (lp.size() >= 2) {
                    fc->weights_.upload(lp[0]);
                    fc->bias_.upload(lp[1]);
                }
            } else if (auto conv = std::dynamic_pointer_cast<Conv2DLayer>(layers_[i])) {
                if (lp.size() >= 2) {
                    conv->weights_.upload(lp[0]);
                    conv->bias_.upload(lp[1]);
                }
            } else if (auto rnn = std::dynamic_pointer_cast<RNNLayer>(layers_[i])) {
                if (lp.size() >= 3) {
                    rnn->weights_ih_.upload(lp[0]);
                    rnn->weights_hh_.upload(lp[1]);
                    rnn->bias_.upload(lp[2]);
                }
            } else if (auto rbm = std::dynamic_pointer_cast<RBMLayer>(layers_[i])) {
                if (lp.size() >= 3) {
                    rbm->weights_.upload(lp[0]);
                    rbm->visible_bias_.upload(lp[1]);
                    rbm->hidden_bias_.upload(lp[2]);
                }
            }
        }
    }



public:
    Network(const std::string &name = "Thot_Network")
        : name_(name), Istraining_(true) {};

    inline void add(std::shared_ptr<Layer> layer) { layers_.push_back(layer); }

    inline void set_optimizer(std::shared_ptr<Optimizer> optimizer) {
        optimizer_ = optimizer;
        for (auto &L : layers_) {
            L->set_optimizer(optimizer);
        }
    }

    inline Utils::Tensor forward_gpu(Utils::Tensor input) {
        for (auto &L : layers_) {
            input = L->forward(input);
        }
        return input;
    }

    inline std::vector<float> forward(const std::vector<float> &input,
                                      const std::vector<int> &input_shape) {
        if (!verify_layer_dimensions()) {
            throw std::runtime_error("Invalid layer dimensions");
        }
        if (!layers_.empty()) {
            int NetworkInput = layers_.front()->get_input_size();
            if (NetworkInput > 0 && input.size() != static_cast<size_t>(NetworkInput)) {
                throw std::invalid_argument("Input size does not match network input layer size\n - [Input] Network: " + std::to_string(NetworkInput) + "  ||  Data: " + std::to_string(input.size()));
            }
        }

        Utils::Tensor input_tensor(input_shape);
        input_tensor.upload(input);
        Utils::Tensor output_tensor = forward_gpu(std::move(input_tensor));
        return output_tensor.download();
    }

    inline void backward(Utils::Tensor grad_output) {
        for (int i = layers_.size() - 1; i >= 0; --i) {
            grad_output = layers_[i]->backward(grad_output);
        }
    }

    size_t get_flops(int batch_size = 1) const {
        size_t total_flops = 0;
        for (const auto &layer : layers_) {
            total_flops += layer->get_flops(batch_size);
        }
        return total_flops;
    }

    int get_model_input_size() const { return layers_.front()->get_input_size(); }
    int get_model_output_size() const { return layers_.back()->get_output_size(); }

    void evaluate(const std::vector<std::vector<float>> &inputs,
                  const std::vector<std::vector<float>> &targets,
                  Evaluation type = Evaluation::Regression, bool verbose = true) {

        if (inputs.size() != targets.size()) {
            throw std::invalid_argument("Inputs and targets must have the same number of samples (x:" + std::to_string(inputs.size()) + " || z: " + std::to_string(targets.size()));
        }
        if (!layers_.empty()) {
            int expected_input = layers_.front()->get_input_size();
            int expected_output = layers_.back()->get_output_size();
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (expected_input > 0 && inputs[i].size() != static_cast<size_t>(expected_input)) {
                    throw std::invalid_argument("Input size mismatch at sample " + std::to_string(i));
                }
                if (expected_output > 0 && targets[i].size() != static_cast<size_t>(expected_output)) {
                    throw std::invalid_argument("Target size mismatch at sample " + std::to_string(i));
                }
            }
        }

        std::vector<std::vector<float>> predictions;
        std::vector<float> latencies;

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<float> output =
                forward(inputs[i], {1, static_cast<int>(inputs[i].size())});
            auto end = std::chrono::high_resolution_clock::now();

            float latency = std::chrono::duration<float>(end - start).count();
            latencies.push_back(latency);
            predictions.push_back(output);
        }

        Evaluations::evaluate(predictions, targets, latencies, get_flops(), get_model_input_size(), get_model_output_size(), type, verbose);
    }

    void set_loss(Loss type, float epsilon = 1e-8f, float delta = 1.0f) {
        loss_function_ = std::make_shared<Losses>(type, epsilon, delta);
    }

    float compute_loss(const Utils::Tensor &predictions,
                       const Utils::Tensor &targets) {
        return loss_function_->compute(predictions, targets);
    }

    Utils::Tensor compute_gradients(const Utils::Tensor &predictions,
                                    const Utils::Tensor &targets) {
        return loss_function_->compute_gradients(predictions, targets);
    }



    void save(const std::string &path) {
        namespace fs = std::filesystem;

        // Create parent directory if needed
        fs::path p(path);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }

        std::cout << "Model Saved in: " << path << std::endl;

        std::ofstream ofs(path, std::ios::out | std::ios::trunc);
        if (!ofs) {
            throw std::runtime_error("Failed to open file for saving: " + path);
        }

        ofs << layers_.size() << "\n";
        for (auto &layer : layers_) {
            if (auto fc = std::dynamic_pointer_cast<FCLayer>(layer)) {
                ofs << "FC " << Activations::to_string(layer->get_activation()) << " "
                    << fc->get_input_size() << " " << fc->get_output_size() << "\n";
                auto w = fc->weights().download();
                auto b = fc->bias().download();
                ofs << w.size();
                for (auto &v : w) ofs << ' ' << v;
                ofs << "\n";
                ofs << b.size();
                for (auto &v : b) ofs << ' ' << v;
                ofs << "\n";
            } else if (auto conv = std::dynamic_pointer_cast<Conv2DLayer>(layer)) {
                ofs << "Conv2D " << Activations::to_string(layer->get_activation()) << " "
                    << conv->in_channels() << " " << conv->in_height() << " "
                    << conv->in_width() << " " << conv->out_channels() << " "
                    << conv->kernel_size() << " " << conv->stride() << " "
                    << conv->padding() << "\n";
                auto w = conv->weights().download();
                auto b = conv->bias().download();
                ofs << w.size();
                for (auto &v : w) ofs << ' ' << v;
                ofs << "\n";
                ofs << b.size();
                for (auto &v : b) ofs << ' ' << v;
                ofs << "\n";
            } else if (auto rnn = std::dynamic_pointer_cast<RNNLayer>(layer)) {
                ofs << "RNN " << Activations::to_string(layer->get_activation()) << " "
                    << rnn->get_input_size() << " " << rnn->get_output_size() << " "
                    << rnn->get_seq_length() << "\n";
                auto wih = rnn->W_ih().download();
                auto whh = rnn->W_hh().download();
                auto b = rnn->bias().download();
                ofs << wih.size();
                for (auto &v : wih) ofs << ' ' << v;
                ofs << "\n";
                ofs << whh.size();
                for (auto &v : whh) ofs << ' ' << v;
                ofs << "\n";
                ofs << b.size();
                for (auto &v : b) ofs << ' ' << v;
                ofs << "\n";
            } else if (auto rbm = std::dynamic_pointer_cast<RBMLayer>(layer)) {
                ofs << "RBM " << Activations::to_string(layer->get_activation()) << " "
                    << rbm->get_input_size() << " " << rbm->get_output_size() << " "
                    << rbm->get_cd_steps() << "\n";
                auto w = rbm->weights().download();
                auto vb = rbm->visible_bias().download();
                auto hb = rbm->hidden_bias().download();
                ofs << w.size();
                for (auto &v : w) ofs << ' ' << v;
                ofs << "\n";
                ofs << vb.size();
                for (auto &v : vb) ofs << ' ' << v;
                ofs << "\n";
                ofs << hb.size();
                for (auto &v : hb) ofs << ' ' << v;
                ofs << "\n";
            }
        }
    }


    void load(const std::string &path) {
        std::ifstream ifs(path);
        if (!ifs)
            throw std::runtime_error("Failed to open file for loading");

        layers_.clear();

        size_t layer_count = 0;
        ifs >> layer_count;
        for (size_t i = 0; i < layer_count; ++i) {
            std::string type;
            ifs >> type;
            if (type == "FC") {
                std::string act_str; int in_size, out_size;
                ifs >> act_str >> in_size >> out_size;
                Activation act = activation_from_string(act_str);
                auto layer = Layer::FC(in_size, out_size, act);
                size_t w_size; ifs >> w_size;
                std::vector<float> w(w_size);
                for (size_t j = 0; j < w_size; ++j) ifs >> w[j];
                size_t b_size; ifs >> b_size;
                std::vector<float> b(b_size);
                for (size_t j = 0; j < b_size; ++j) ifs >> b[j];
                auto fc = std::dynamic_pointer_cast<FCLayer>(layer);
                fc->weights().upload(w);
                fc->bias().upload(b);
                if (optimizer_) layer->set_optimizer(optimizer_);
                layers_.push_back(layer);
            } else if (type == "Conv2D") {
                std::string act_str; int in_c, in_h, in_w, out_c, k, s, p;
                ifs >> act_str >> in_c >> in_h >> in_w >> out_c >> k >> s >> p;
                Activation act = activation_from_string(act_str);
                auto layer = Layer::Conv2D(in_c, in_h, in_w, out_c, k, s, p, act);
                size_t w_size; ifs >> w_size;
                std::vector<float> w(w_size);
                for (size_t j = 0; j < w_size; ++j) ifs >> w[j];
                size_t b_size; ifs >> b_size;
                std::vector<float> b(b_size);
                for (size_t j = 0; j < b_size; ++j) ifs >> b[j];
                auto conv = std::dynamic_pointer_cast<Conv2DLayer>(layer);
                conv->weights().upload(w);
                conv->bias().upload(b);
                if (optimizer_) layer->set_optimizer(optimizer_);
                layers_.push_back(layer);
            } else if (type == "RNN") {
                std::string act_str; int in_size, hidden, seq_len;
                ifs >> act_str >> in_size >> hidden >> seq_len;
                Activation act = activation_from_string(act_str);
                auto layer = Layer::RNN(in_size, hidden, seq_len, act);
                size_t wih_size; ifs >> wih_size;
                std::vector<float> wih(wih_size);
                for (size_t j = 0; j < wih_size; ++j) ifs >> wih[j];
                size_t whh_size; ifs >> whh_size;
                std::vector<float> whh(whh_size);
                for (size_t j = 0; j < whh_size; ++j) ifs >> whh[j];
                size_t b_size; ifs >> b_size;
                std::vector<float> b(b_size);
                for (size_t j = 0; j < b_size; ++j) ifs >> b[j];
                auto rnn = std::dynamic_pointer_cast<RNNLayer>(layer);
                rnn->W_ih().upload(wih);
                rnn->W_hh().upload(whh);
                rnn->bias().upload(b);
                if (optimizer_) layer->set_optimizer(optimizer_);
                layers_.push_back(layer);
            } else if (type == "RBM") {
                std::string act_str; int vis, hid, cd;
                ifs >> act_str >> vis >> hid >> cd;
                Activation act = activation_from_string(act_str);
                auto layer = Layer::RBM(vis, hid, cd, act);
                size_t w_size; ifs >> w_size;
                std::vector<float> w(w_size);
                for (size_t j = 0; j < w_size; ++j) ifs >> w[j];
                size_t vb_size; ifs >> vb_size;
                std::vector<float> vb(vb_size);
                for (size_t j = 0; j < vb_size; ++j) ifs >> vb[j];
                size_t hb_size; ifs >> hb_size;
                std::vector<float> hb(hb_size);
                for (size_t j = 0; j < hb_size; ++j) ifs >> hb[j];
                auto rbm = std::dynamic_pointer_cast<RBMLayer>(layer);
                rbm->weights().upload(w);
                rbm->visible_bias().upload(vb);
                rbm->hidden_bias().upload(hb);
                if (optimizer_) layer->set_optimizer(optimizer_);
                layers_.push_back(layer);
            } else {
                throw std::runtime_error("Unknown layer type: " + type);
            }
        }
    }

    inline void summary() {
        std::cout << "Network: " << name_ << std::endl;
        std::cout << "Layers:" << std::endl;

        size_t total_flops = 0;
        size_t batch_size = 1;

        std::cout << "+---------------+----------------------+---------------------"
                     "-+----------------------+---------------+"
                  << std::endl;
        std::cout << "| Layer         | Type                 | Activation          "
                     " | Initialization       | FLOPs         |"
                  << std::endl;
        std::cout << "+---------------+----------------------+---------------------"
                     "-+----------------------+---------------+"
                  << std::endl;

        for (size_t i = 0; i < layers_.size(); ++i) {
            auto &layer = layers_[i];
            std::string layer_name = layer->get_name();
            std::string activation_name = Thot::Activations::to_string(layer->get_activation());
            std::string init_name = Thot::Initializers::to_string(layer->get_initialization());

            size_t layer_flops = layer->get_flops(batch_size);
            total_flops += layer_flops;

            if (layer_name.length() > 20)
                layer_name = layer_name.substr(0, 17) + "...";
            if (activation_name.length() > 20)
                activation_name = activation_name.substr(0, 17) + "...";
            if (init_name.length() > 20)
                init_name = init_name.substr(0, 17) + "...";

            std::cout << "| " << std::left << std::setw(13) << i + 1 << " | "
                      << std::left << std::setw(20) << layer_name << " | "
                      << std::left << std::setw(20) << activation_name << " | "
                      << std::left << std::setw(20) << init_name << " | "
                      << std::right << std::setw(13) << layer_flops << " |"
                      << std::endl;
        }

        std::cout << "+---------------+----------------------+---------------------"
                     "-+----------------------+---------------+"
                  << std::endl;
        std::cout << "| Thot Model    |                                            "
                     "                                "
                  << std::right << std::setw(7) << total_flops << " |" << std::endl;
        std::cout << "+---------------+--------------------------------------------"
                     "----------------------------------------+"
                  << std::endl;

        std::cout << "\nTraining Configuration:" << std::endl;
        std::cout << "+----------------------+----------------------+--------------"
                     "--------+"
                  << std::endl;
        std::cout << "| Optimizer           | Parameters           | Loss Function "
                     "       |"
                  << std::endl;
        std::cout << "+----------------------+----------------------+--------------"
                     "--------+"
                  << std::endl;

        std::string optimizer_name = optimizer_ ? optimizer_->get_name() : "None";
        std::string optimizer_params =
            optimizer_ ? optimizer_->get_params() : "None";
        std::string loss_name =
            loss_function_ ? Thot::Losses::to_string(loss_function_->get_type())
                           : "None";
        std::string loss_params =
            loss_function_ ? loss_function_->get_params() : "None";

        if (optimizer_name.length() > 20)
            optimizer_name = optimizer_name.substr(0, 17) + "...";
        if (optimizer_params.length() > 20)
            optimizer_params = optimizer_params.substr(0, 17) + "...";
        if (loss_name.length() > 20)
            loss_name = loss_name.substr(0, 17) + "...";
        if (loss_params.length() > 20)
            loss_params = loss_params.substr(0, 17) + "...";

        std::cout << "| " << std::left << std::setw(20) << optimizer_name << " | "
                  << std::left << std::setw(20) << optimizer_params << " | "
                  << std::left << std::setw(20) << loss_name << " |" << std::endl;
        std::cout << "| " << std::left << std::setw(20) << ""
                  << " | " << std::left << std::setw(20) << ""
                  << " | " << std::left << std::setw(20) << loss_params << " |"
                  << std::endl;
        std::cout << "+----------------------+----------------------+--------------"
                     "--------+"
                  << std::endl;
    }
    template <typename BatchMethod, typename KFoldMethod>
    void train(const std::vector<std::vector<float>> &inputs,
            const std::vector<std::vector<float>> &targets,
            const BatchMethod &batch_method, const KFoldMethod &kfold_method,
            int log_interval = 100, bool verbose = true,
            bool restore_best_model = true) {

        if (!optimizer_) { // if not defined
            optimizer_ = Thot::Optimizer::SGD(0.01f);
            for (auto &L : layers_) {
                L->set_optimizer(optimizer_);
            }
        }

        auto total_start = std::chrono::high_resolution_clock::now();
        std::vector<float> epoch_times;
        std::vector<float> fold_losses;
        double best_val_loss = std::numeric_limits<double>::infinity();
        ModelParams best_params;

        int folds = kfold_method.get_folds();

        for (int fold = 0; fold < folds; ++fold) {
            if (folds > 1 && verbose) {
                std::cout << "\nTraining Fold " << fold + 1 << "/" << folds
                          << std::endl;
            }

            std::vector<std::vector<float>> train_inputs, train_targets, val_inputs,
                val_targets;
            kfold_method.split(inputs, targets, fold, train_inputs, train_targets,
                   val_inputs, val_targets);

            for (int epoch = 0; epoch < batch_method.get_epochs(); ++epoch) {
                auto epoch_start = std::chrono::high_resolution_clock::now();

                double epoch_loss = batch_method.template train_epoch<Network>(
                        *this, train_inputs, train_targets, log_interval, verbose,
                        epoch + 1, batch_method.get_epochs());

                auto epoch_end = std::chrono::high_resolution_clock::now();
                float epoch_time =
                    std::chrono::duration<float>(epoch_end - epoch_start).count();
                epoch_times.push_back(epoch_time);
                std::cout.unsetf(std::ios_base::floatfield);
                if (epoch % log_interval == 0 || epoch == batch_method.get_epochs() - 1) {
                    std::cout << "Epoch " << epoch << " - Average Loss: "
                              << epoch_loss;

                    if (folds > 1) {
                        double val_loss = 0.0;
                        for (size_t i = 0; i < val_inputs.size(); ++i) {
                            std::vector<float> output = forward(val_inputs[i], {1, static_cast<int>(val_inputs[i].size())});

                            Utils::Tensor prediction_tensor(
                                {1, static_cast<int>(output.size())});
                            prediction_tensor.upload(output);

                            Utils::Tensor target_tensor(
                                {1, static_cast<int>(val_targets[i].size())});
                            target_tensor.upload(val_targets[i]);

                            val_loss += loss_function_->compute(prediction_tensor, target_tensor);
                        }
                        val_loss /= val_inputs.size();
                        std::cout << " - Validation Loss: " << val_loss;
                        fold_losses.push_back(val_loss);

                        if (val_loss < best_val_loss) { //best state
                            best_val_loss = val_loss;
                            best_params = capture_parameters();
                        }
                    }
                    std::cout << std::endl;
                }
            }
        }

        auto total_end = std::chrono::high_resolution_clock::now();

        if (restore_best_model && !best_params.empty()) { // restore
            apply_parameters(best_params);
        }

        float total_time =
            std::chrono::duration<float>(total_end - total_start).count();

        float avg_epoch_time =
            std::accumulate(epoch_times.begin(), epoch_times.end(), 0.0f) / (batch_method.get_epochs() * folds);

        float min_epoch_time =
            *std::min_element(epoch_times.begin(), epoch_times.end());

        float max_epoch_time =
            *std::max_element(epoch_times.begin(), epoch_times.end());

        float samples_per_second = (inputs.size() * batch_method.get_epochs() * folds) / total_time;

        std::cout << std::fixed << std::setprecision(2);

        std::cout << "\nTraining Summary:\n";
        std::cout << "----------------\n";
        std::cout << "Total Epochs: " << batch_method.get_epochs() * folds << "\n";
        if (folds > 1) {
            float avg_fold_loss =
                std::accumulate(fold_losses.begin(), fold_losses.end(), 0.0f) /
                fold_losses.size();
            float min_fold_loss =
                *std::min_element(fold_losses.begin(), fold_losses.end());
            float max_fold_loss =
                *std::max_element(fold_losses.begin(), fold_losses.end());
            std::cout << "Average Validation Loss: " << avg_fold_loss << "\n";
            std::cout << "Min Validation Loss: " << min_fold_loss << "\n";
            std::cout << "Max Validation Loss: " << max_fold_loss << "\n";
        }
        std::cout << "Total Training Time: " << format_time(total_time) << "\n";
        std::cout << "Average Epoch Time: " << format_time(avg_epoch_time) << "\n";
        std::cout << "Min Epoch Time: " << format_time(min_epoch_time) << "\n";
        std::cout << "Max Epoch Time: " << format_time(max_epoch_time) << "\n";
        std::cout << "Throughput: " << format_samples_per_second(samples_per_second) << "\n";

        std::cout.unsetf(std::ios_base::floatfield);
    }
};
} // namespace Thot

