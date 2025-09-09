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
#include <nlohmann/json.hpp>


#include "layers/layers.hpp"
#include "tensor.hpp"

#include "activations/activations.hpp"
#include "evaluations/evaluation.hpp"
#include "losses/losses.hpp"
#include "metrics/metrics.hpp"
#include "optimizations/optimizations.hpp"
#include "attentions/attentions.hpp"

#include "utils/translators.hpp"

#include "LearningProcess/batch.hpp"
#include "LearningProcess/kfold.hpp"


class Layer;
class Optimizer;

namespace Thot {




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
    size_t total_flops = 0;
    size_t total_parm = 0;



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
                std::cout << "Layer dimension mismatch between layer n" << i+1
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
            } else if (auto rcnn = std::dynamic_pointer_cast<RCNNLayer>(layer)) {
                lp.push_back(rcnn->weights().download());
                lp.push_back(rcnn->bias().download());
            } else if (auto rnn = std::dynamic_pointer_cast<RNNLayer>(layer)) {
                lp.push_back(rnn->weights_ih_.download());
                lp.push_back(rnn->weights_hh_.download());
                lp.push_back(rnn->bias_.download());
            } else if (auto rbm = std::dynamic_pointer_cast<RBMLayer>(layer)) {
                lp.push_back(rbm->weights_.download());
                lp.push_back(rbm->visible_bias_.download());
                lp.push_back(rbm->hidden_bias_.download());
            } else if (auto vae = std::dynamic_pointer_cast<VAELayer>(layer)) {
                lp.push_back(vae->enc_mean_->weights_.download());
                lp.push_back(vae->enc_mean_->bias_.download());
                lp.push_back(vae->enc_logvar_->weights_.download());
                lp.push_back(vae->enc_logvar_->bias_.download());
                lp.push_back(vae->dec_->weights_.download());
                lp.push_back(vae->dec_->bias_.download());
            } else if (auto scae = std::dynamic_pointer_cast<SparseContractiveAELayer>(layer)) {
                lp.push_back(scae->enc_weights_.download());
                lp.push_back(scae->enc_bias_.download());
                lp.push_back(scae->dec_weights_.download());
                lp.push_back(scae->dec_bias_.download());
            } else if (auto mha = std::dynamic_pointer_cast<MHAAttentionLayer>(layer)) {
                lp.push_back(mha->W_q().download());
                lp.push_back(mha->b_q().download());
                lp.push_back(mha->W_k().download());
                lp.push_back(mha->b_k().download());
                lp.push_back(mha->W_v().download());
                lp.push_back(mha->b_v().download());
                lp.push_back(mha->W_o().download());
                lp.push_back(mha->b_o().download());
            } else if (auto mla = std::dynamic_pointer_cast<MLAAttentionLayer>(layer)) {
                lp.push_back(mla->W_DKV().download());
                lp.push_back(mla->b_DKV().download());
                lp.push_back(mla->W_UK().download());
                lp.push_back(mla->b_UK().download());
                lp.push_back(mla->W_UV().download());
                lp.push_back(mla->b_UV().download());
                lp.push_back(mla->W_Q().download());
                lp.push_back(mla->b_Q().download());
                lp.push_back(mla->W_O().download());
                lp.push_back(mla->b_O().download());
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
            } else if (auto rcnn = std::dynamic_pointer_cast<RCNNLayer>(layers_[i])) {
                if (lp.size() >= 2) {
                    rcnn->weights().upload(lp[0]);
                    rcnn->bias().upload(lp[1]);
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
            } else if (auto vae = std::dynamic_pointer_cast<VAELayer>(layers_[i])) {
                if (lp.size() >= 6) {
                    vae->enc_mean_->weights_.upload(lp[0]);
                    vae->enc_mean_->bias_.upload(lp[1]);
                    vae->enc_logvar_->weights_.upload(lp[2]);
                    vae->enc_logvar_->bias_.upload(lp[3]);
                    vae->dec_->weights_.upload(lp[4]);
                    vae->dec_->bias_.upload(lp[5]);
                }
            } else if (auto scae = std::dynamic_pointer_cast<SparseContractiveAELayer>(layers_[i])) {
                if (lp.size() >= 4) {
                    scae->enc_weights_.upload(lp[0]);
                    scae->enc_bias_.upload(lp[1]);
                    scae->dec_weights_.upload(lp[2]);
                    scae->dec_bias_.upload(lp[3]);
                }
            } else if (auto mha = std::dynamic_pointer_cast<MHAAttentionLayer>(layers_[i])) {
                if (lp.size() >= 8) {
                    mha->W_q().upload(lp[0]);
                    mha->b_q().upload(lp[1]);
                    mha->W_k().upload(lp[2]);
                    mha->b_k().upload(lp[3]);
                    mha->W_v().upload(lp[4]);
                    mha->b_v().upload(lp[5]);
                    mha->W_o().upload(lp[6]);
                    mha->b_o().upload(lp[7]);
                }
            } else if (auto mla = std::dynamic_pointer_cast<MLAAttentionLayer>(layers_[i])) {
                if (lp.size() >= 10) {
                    mla->W_DKV().upload(lp[0]);
                    mla->b_DKV().upload(lp[1]);
                    mla->W_UK().upload(lp[2]);
                    mla->b_UK().upload(lp[3]);
                    mla->W_UV().upload(lp[4]);
                    mla->b_UV().upload(lp[5]);
                    mla->W_Q().upload(lp[6]);
                    mla->b_Q().upload(lp[7]);
                    mla->W_O().upload(lp[8]);
                    mla->b_O().upload(lp[9]);
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
            if (auto rcnn = std::dynamic_pointer_cast<RCNNLayer>(L)) {
                rcnn->conv_.set_optimizer(optimizer);
            }
        }
    }

    inline std::shared_ptr<Optimizer> get_optimizer() const { return optimizer_; }

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

        if (Istraining_) { // spike neuron plasticity
            Istraining_ = false;
            for (auto &L : layers_) {
                L->set_training(false);
                if (auto spike = std::dynamic_pointer_cast<SpikeLayer>(L)) {
                    spike->reset_state();
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

        Evaluations::evaluate(predictions, targets, latencies, get_flops(), get_model_input_size(), get_model_output_size(), type, total_flops, total_parm, verbose);
    }

    void set_loss(Loss type, float epsilon = 1e-8f, float delta = 1.0f) {
        loss_function_ = std::make_shared<Losses>(type, epsilon, delta);
    }

    float compute_loss(const Utils::Tensor &predictions,
                       const Utils::Tensor &targets) {
        float base = loss_function_->compute(predictions, targets);
        float reg = 0.0f;
        for (const auto &L : layers_) { //TODO Impl Penalization
            reg += L->regularization_loss(); // Sparce Con AE
        }
        return base + reg;

    }

    Utils::Tensor compute_gradients(const Utils::Tensor &predictions,
                                    const Utils::Tensor &targets) {
        return loss_function_->compute_gradients(predictions, targets);
    }

        void save(const std::string &path) {
            namespace fs = std::filesystem;
            fs::path dir = fs::path(path) / name_;
            fs::create_directories(dir);

            // Save model parameters in binary form
            ModelParams params = capture_parameters();
            std::ofstream param_file(dir / "parameters.bin", std::ios::binary | std::ios::trunc);
            if (!param_file)
                throw std::runtime_error("Failed to open parameter file for saving");
            size_t layer_count = params.size();
            param_file.write(reinterpret_cast<const char*>(&layer_count), sizeof(size_t));
            for (const auto &lp : params) {
                size_t vec_count = lp.size();
                param_file.write(reinterpret_cast<const char*>(&vec_count), sizeof(size_t));
                for (const auto &vec : lp) {
                    size_t sz = vec.size();
                    param_file.write(reinterpret_cast<const char*>(&sz), sizeof(size_t));
                    param_file.write(reinterpret_cast<const char*>(vec.data()), sz * sizeof(float));
                }
            }
            // Build JSON architecture description
            nlohmann::json j;
            j["name"] = name_;

            // Optimizer information
            nlohmann::json jopt;
            if (optimizer_) {
                jopt["name"] = optimizer_->get_name();
                jopt["learning_rate"] = optimizer_->get_learning_rate();
                if (auto sgdm = std::dynamic_pointer_cast<SGDM>(optimizer_)) {
                    jopt["momentum"] = sgdm->get_momentum();
                } else if (auto adam = std::dynamic_pointer_cast<Adam>(optimizer_)) {
                    jopt["beta1"] = adam->get_beta1();
                    jopt["beta2"] = adam->get_beta2();
                    jopt["epsilon"] = adam->get_epsilon();
                } else if (auto muon = std::dynamic_pointer_cast<Muon>(optimizer_)) {
                    jopt["beta"] = muon->get_beta();
                    jopt["weight_decay"] = muon->get_weight_decay();
                } else if (auto adamuon = std::dynamic_pointer_cast<AdaMuon>(optimizer_)) {
                    jopt["beta1"] = adamuon->get_beta1();
                    jopt["beta2"] = adamuon->get_beta2();
                    jopt["weight_decay"] = adamuon->get_weight_decay();
                }
            } else {
                jopt["name"] = "None";
            }
            j["optimizer"] = jopt;

            // Loss information
            nlohmann::json jloss;
            if (loss_function_) {
                jloss["name"] = Losses::to_string(loss_function_->get_type());
                std::string params_str = loss_function_->get_params();
                float eps = 1e-8f, delta = 1.0f;
                auto pos = params_str.find("Eps=");
                if (pos != std::string::npos)
                    eps = std::stof(params_str.substr(pos + 4));
                pos = params_str.find("Delta=");
                if (pos != std::string::npos)
                    delta = std::stof(params_str.substr(pos + 6));
                jloss["epsilon"] = eps;
                jloss["delta"] = delta;
            } else {
                jloss["name"] = "None";
            }
            j["loss"] = jloss;
            // Layers architecture
            nlohmann::json jlayers = nlohmann::json::array();
            for (auto &layer : layers_) {
                nlohmann::json jl;
                if (auto fc = std::dynamic_pointer_cast<FCLayer>(layer)) {
                    jl["type"] = "FC";
                    jl["activation"] = Activations::to_string(layer->get_activation());
                    jl["initializer"] = Initializations::to_string(layer->get_initialization());
                    jl["params"] = { {"input_size", fc->get_input_size()}, {"output_size", fc->get_output_size()} };
                } else if (auto conv = std::dynamic_pointer_cast<Conv2DLayer>(layer)) {
                    jl["type"] = "Conv2D";
                    jl["activation"] = Activations::to_string(layer->get_activation());
                    jl["initializer"] = Initializations::to_string(layer->get_initialization());
                    jl["params"] = {
                        {"in_channels", conv->in_channels()},
                        {"in_height", conv->in_height()},
                        {"in_width", conv->in_width()},
                        {"out_channels", conv->out_channels()},
                        {"kernel_size", conv->kernel_size()},
                        {"stride", conv->stride()},
                        {"padding", conv->padding()}
                    };
                } else if (auto rnn = std::dynamic_pointer_cast<RNNLayer>(layer)) {
                    jl["type"] = "RNN";
                    jl["activation"] = Activations::to_string(layer->get_activation());
                    jl["initializer"] = Initializations::to_string(layer->get_initialization());
                    jl["params"] = {
                        {"input_size", rnn->get_input_size()},
                        {"output_size", rnn->get_output_size()},
                        {"seq_length", rnn->get_seq_length()}
                    };
                    } else if (auto rcnn = std::dynamic_pointer_cast<RCNNLayer>(layer)) {
                    jl["type"] = "RCNN";
                    jl["activation"] = Activations::to_string(layer->get_activation());
                    jl["initializer"] = Initializations::to_string(layer->get_initialization());
                    jl["params"] = {
                        {"in_channels", rcnn->conv_.in_channels()},
                        {"in_height", rcnn->conv_.in_height()},
                        {"in_width", rcnn->conv_.in_width()},
                        {"out_channels", rcnn->conv_.out_channels()},
                        {"kernel_size", rcnn->conv_.kernel_size()},
                        {"stride", rcnn->conv_.stride()},
                        {"padding", rcnn->conv_.padding()},
        {"pooled_h", rcnn->pooled_h_},
                        {"pooled_w", rcnn->pooled_w_}
                    };
                } else if (auto vae = std::dynamic_pointer_cast<VAELayer>(layer)) {
                    jl["type"] = "VAE";
                    jl["activation"] = Activations::to_string(layer->get_activation());
                    jl["initializer"] = Initializations::to_string(layer->get_initialization());
                    jl["params"] = {
                        {"input_size", vae->get_input_size()},
                        {"latent_size", vae->latent_size_}
                    };
                } else if (auto scae = std::dynamic_pointer_cast<SparseContractiveAELayer>(layer)) {
                    jl["type"] = "SparseAE";
                    jl["activation"] = Activations::to_string(layer->get_activation());
                    jl["initializer"] = Initializations::to_string(layer->get_initialization());
                    jl["params"] = {
                        {"input_size", scae->input_size_},
                        {"latent_size", scae->latent_size_},
                        {"use_sparsity", scae->use_sparsity_},
                        {"use_contractive", scae->use_contractive_},
                        {"sparsity_rho", scae->sparsity_rho_},
                        {"sparsity_beta", scae->sparsity_beta_},
                        {"contractive_lambda", scae->contractive_lambda_}
                    };
                } else if (auto mha = std::dynamic_pointer_cast<MHAAttentionLayer>(layer)) {
                    jl["type"] = "MHA";
                    jl["initializer"] = Initializations::to_string(layer->get_initialization());
                    jl["params"] = {
                        {"embed_dim", mha->get_input_size()},
                        {"num_heads", mha->num_heads()}
                    };
                } else if (auto mla = std::dynamic_pointer_cast<MLAAttentionLayer>(layer)) {
                    jl["type"] = "MLA";
                    jl["initializer"] = Initializations::to_string(layer->get_initialization());
                    jl["params"] = {
                        {"embed_dim", mla->get_input_size()},
                        {"num_heads", mla->num_heads()},
                        {"latent_dim", mla->latent_dim()}
                    };
                } else if (auto rbm = std::dynamic_pointer_cast<RBMLayer>(layer)) {
                    jl["type"] = "RBM";
                    jl["activation"] = Activations::to_string(layer->get_activation());
                    jl["initializer"] = Initializations::to_string(layer->get_initialization());
                    jl["params"] = {
                        {"input_size", rbm->get_input_size()},
                        {"output_size", rbm->get_output_size()},
                        {"cd_steps", rbm->get_cd_steps()}
                    };
                } else if (auto spike = std::dynamic_pointer_cast<SpikeLayer>(layer)) {
                    jl["type"] = "Spike";
                    jl["initializer"] = Initializations::to_string(layer->get_initialization());
                    jl["params"] = {
                        {"size", spike->get_input_size()},
                        {"threshold", spike->threshold_}
                    };
                } else if (auto pool = std::dynamic_pointer_cast<MaxPool2DLayer>(layer)) {
                    jl["type"] = "MaxPool2D";
                    jl["initializer"] = Initializations::to_string(layer->get_initialization());
                    jl["params"] = {
                        {"in_channels", pool->in_channels()},
                        {"in_height", pool->in_height()},
                        {"in_width", pool->in_width()},
                        {"kernel_size", pool->kernel_size()},
                        {"stride", pool->stride()}
                    };
                } else if (auto flat = std::dynamic_pointer_cast<FlattenLayer>(layer)) {
                    jl["type"] = "Flatten";
                    jl["initializer"] = Initializations::to_string(layer->get_initialization());
                    jl["params"] = {
                        {"in_channels", flat->in_channels()},
                        {"in_height", flat->in_height()},
                        {"in_width", flat->in_width()}
                    };
                }
                jlayers.push_back(jl);
            }
        j["layers"] = jlayers;

        std::ofstream json_file(dir / "architecture.json", std::ios::out | std::ios::trunc);
        if (!json_file)
            throw std::runtime_error("Failed to open architecture file for saving");
        json_file << j.dump(4);
        std::cout << "Model and Architecture saved in:" << path << std::endl;
    }




    void load(const std::string &path) {
        namespace fs = std::filesystem;
        fs::path dir(path);
        if (!fs::exists(dir / "architecture.json")) {
            fs::path alt = dir / name_;
            if (fs::exists(alt / "architecture.json"))
                dir = alt;
            else
                throw std::runtime_error("Architecture file not found");
        }

        // Parse JSON architecture
        nlohmann::json j;
        std::ifstream json_file(dir / "architecture.json");
        if (!json_file)
            throw std::runtime_error("Failed to open architecture file for loading");
        json_file >> j;

        name_ = j.value("name", name_);

        // Optimizer
        if (j.contains("optimizer")) {
            std::string opt_name = j["optimizer"].value("name", "None");
            if (opt_name == "SGD") {
                float lr = j["optimizer"].value("learning_rate", 0.01f);
                set_optimizer(Optimizer::SGD(lr));
            } else if (opt_name == "SGDM") {
                float lr = j["optimizer"].value("learning_rate", 0.01f);
                float momentum = j["optimizer"].value("momentum", 0.9f);
                set_optimizer(Optimizer::SGDM(lr, momentum));
            } else if (opt_name == "Adam") {
                float lr = j["optimizer"].value("learning_rate", 0.001f);
                float b1 = j["optimizer"].value("beta1", 0.9f);
                float b2 = j["optimizer"].value("beta2", 0.999f);
                float eps = j["optimizer"].value("epsilon", 1e-8f);
                set_optimizer(Optimizer::Adam(lr, b1, b2, eps));
            } else if (opt_name == "Muon") {
                float lr = j["optimizer"].value("learning_rate", 0.01f);
                float beta = j["optimizer"].value("beta", 0.9f);
                float wd = j["optimizer"].value("weight_decay", 0.0f);
                set_optimizer(Optimizer::Muon(lr, beta, wd));
            } else if (opt_name == "AdaMuon") {
                float lr = j["optimizer"].value("learning_rate", 0.001f);
                float b1 = j["optimizer"].value("beta1", 0.9f);
                float b2 = j["optimizer"].value("beta2", 0.999f);
                float wd = j["optimizer"].value("weight_decay", 0.0f);
                set_optimizer(Optimizer::AdaMuon(lr, b1, b2, wd));
            }
        }

        // Loss
        if (j.contains("loss")) {
            std::string loss_name = j["loss"].value("name", "MSE");
            float eps = j["loss"].value("epsilon", 1e-8f);
            float delta = j["loss"].value("delta", 1.0f);
            set_loss(loss_from_string(loss_name), eps, delta);
        }

        // Layers
        layers_.clear();
        if (j.contains("layers")) {
            for (const auto &jl : j["layers"]) {
                std::string type = jl.value("type", "");
                std::string act_str = jl.value("activation", "Linear");
                std::string init_str = jl.value("initialization", "Xavier");
                Activation act = activation_from_string(act_str);
                Initialization init = initialization_from_string(init_str);
                std::shared_ptr<Layer> layer;
                if (type == "FC") {
                    int in = jl["params"].value("input_size", 0);
                    int out = jl["params"].value("output_size", 0);
                    layer = Layer::FC(in, out, act, init);
                } else if (type == "Conv2D") {
                    int in_c = jl["params"].value("in_channels", 0);
                    int in_h = jl["params"].value("in_height", 0);
                    int in_w = jl["params"].value("in_width", 0);
                    int out_c = jl["params"].value("out_channels", 0);
                    int k = jl["params"].value("kernel_size", 0);
                    int s = jl["params"].value("stride", 0);
                    int p = jl["params"].value("padding", 0);
                    layer = Layer::Conv2D(in_c, in_h, in_w, out_c, k, s, p, act, init);
                } else if (type == "RNN") {
                    int in = jl["params"].value("input_size", 0);
                    int out = jl["params"].value("output_size", 0);
                    int seq = jl["params"].value("seq_length", 0);
                    layer = Layer::RNN(in, out, seq, act, init);
                    } else if (type == "RCNN") {
                    int in_c = jl["params"].value("in_channels", 0);
                    int in_h = jl["params"].value("in_height", 0);
                    int in_w = jl["params"].value("in_width", 0);
                    int out_c = jl["params"].value("out_channels", 0);
                    int k = jl["params"].value("kernel_size", 0);
                    int s = jl["params"].value("stride", 0);
                    int p = jl["params"].value("padding", 0);
                    int ph = jl["params"].value("pooled_h", 0);
                    int pw = jl["params"].value("pooled_w", 0);
                    layer = Layer::RCNN(in_c, in_h, in_w, out_c, k, s, p, ph, pw, act, init);
                } else if (type == "VAE") {
                    int in = jl["params"].value("input_size", 0);
                    int latent = jl["params"].value("latent_size", 0);
                    layer = Layer::VAE(in, latent, act, init);
                } else if (type == "SparseAE") {
                    int in = jl["params"].value("input_size", 0);
                    int latent = jl["params"].value("latent_size", 0);
                    bool use_s = jl["params"].value("use_sparsity", false);
                    bool use_c = jl["params"].value("use_contractive", false);
                    float rho = jl["params"].value("sparsity_rho", 0.05f);
                    float beta = jl["params"].value("sparsity_beta", 1e-3f);
                    float lambda = jl["params"].value("contractive_lambda", 1e-3f);
                    layer = Layer::SparseAE(in, latent, act, init, use_s, use_c, rho, beta, lambda);
                } else if (type == "MHA") {
                    int embed = jl["params"].value("embed_dim", 0);
                    int heads = jl["params"].value("num_heads", 1);
                    layer = Attention::MHA(embed, heads, init);
                } else if (type == "MLA") {
                    int embed = jl["params"].value("embed_dim", 0);
                    int heads = jl["params"].value("num_heads", 1);
                    int latent = jl["params"].value("latent_dim", 0);
                    layer = Attention::MLA(embed, heads, latent, init);
                } else if (type == "RBM") {
                    int vis = jl["params"].value("input_size", 0);
                    int hid = jl["params"].value("output_size", 0);
                    int cd = jl["params"].value("cd_steps", 0);
                    layer = Layer::RBM(vis, hid, cd, act, init);
                } else if (type == "Spike") {
                    int sz = jl["params"].value("size", 0);
                    float th = jl["params"].value("threshold", 1.0f);
                    layer = Layer::Spike(sz, th);
                } else if (type == "MaxPool2D") {
                    int in_c = jl["params"].value("in_channels", 0);
                    int in_h = jl["params"].value("in_height", 0);
                    int in_w = jl["params"].value("in_width", 0);
                    int k = jl["params"].value("kernel_size", 0);
                    int s = jl["params"].value("stride", 0);
                    layer = Layer::MaxPool2D(in_c, in_h, in_w, k, s);
                } else if (type == "Flatten") {
                    int in_c = jl["params"].value("in_channels", 0);
                    int in_h = jl["params"].value("in_height", 0);
                    int in_w = jl["params"].value("in_width", 0);
                    layer = Layer::Flatten(in_c, in_h, in_w);
                }
                if (layer) {
                    if (optimizer_) layer->set_optimizer(optimizer_);
                    layers_.push_back(layer);
                }
            }
        }

        // Load parameters
        std::ifstream param_file(dir / "parameters.bin", std::ios::binary);
        if (!param_file)
            throw std::runtime_error("Failed to open parameter file for loading");
        size_t layer_count = 0;
        param_file.read(reinterpret_cast<char*>(&layer_count), sizeof(size_t));
        ModelParams params(layer_count);
        for (size_t i = 0; i < layer_count; ++i) {
            size_t vec_count = 0;
            param_file.read(reinterpret_cast<char*>(&vec_count), sizeof(size_t));
            LayerParams lp(vec_count);
            for (size_t jv = 0; jv < vec_count; ++jv) {
                size_t sz = 0;
                param_file.read(reinterpret_cast<char*>(&sz), sizeof(size_t));
                lp[jv].resize(sz);
                param_file.read(reinterpret_cast<char*>(lp[jv].data()), sz * sizeof(float));
            }
            params[i] = std::move(lp);
        }
        apply_parameters(params);
        std::cout << "Model and Architecture loaded from:" << path << std::endl;
    }

    inline void initializaiton_summary() {
        float lat=0.0f;
        float total=0.0f;
        std::vector<float> lat_vec;
        std::vector<std::string> names_vec;
        for (auto &L : layers_) {
            lat = L->get_latency();
            lat*=1e-9;
            lat_vec.push_back(lat);
            total+= lat;
            names_vec.push_back(L->get_name());
        }
        std::cout << "\n\n------------------------------------" <<std::endl;

        std::cout << "Latency from Initializaiton:" << std::endl;
        for (int i = 0; i < lat_vec.size(); ++i) {
            std::cout << "[" << i+1 << "]-> " << names_vec[i] << ": " << Thot::format_time(lat_vec[i]) << std::endl;
        }
        std::cout << "------------------------------------" <<std::endl;
        std::cout << "[MODEL]-> " << name_ << ": "<<Thot::format_time(total) << std::endl;
        std::cout << "------------------------------------\n\n" <<std::endl;

    }

    inline void summary() {
        std::cout << "Network: " << name_ << std::endl;
        std::cout << "Layers:" << std::endl;


        size_t batch_size = 1;

        std::cout << "+---------------+----------------------+---------------------"
                     "-+----------------------+---------------+---------------+"
                  << std::endl;
        std::cout << "| Layer         | Type                 | Activation          "
                     " | Initialization       | Total FLOPs   | Parameters    |"
                  << std::endl;
        std::cout << "+---------------+----------------------+---------------------"
                     "-+----------------------+---------------+---------------+"
                  << std::endl;

        for (size_t i = 0; i < layers_.size(); ++i) {
            auto &layer = layers_[i];
            std::string layer_name = layer->get_name();
            std::string activation_name = Thot::Activations::to_string(layer->get_activation());
            std::string init_name = Thot::Initializations::to_string(layer->get_initialization());

            size_t layer_flops = layer->get_flops(batch_size);
            size_t layer_parm = layer->get_parameters();

            total_flops += layer_flops;
            total_parm += layer_parm;

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
                      << std::right << std::setw(13) << human_readable_size(layer_flops) << " | "
                        << std::right << std::setw(13) << human_readable_size(layer_parm) << " |"
                      << std::endl;
        }

        std::cout << "+---------------+----------------------+---------------------"
                     "-+----------------------+---------------+---------------+"
                  << std::endl;
        std::cout << "| Thot Model    |                                            "
                     "                                "
                  << std::right << std::setw(7) << human_readable_size(total_flops) << " | "<< std::setw(13) << human_readable_size(total_parm) << " |" << std::endl;
        std::cout << "+---------------+--------------------------------------------"
                     "----------------------------------------+---------------+"
                  << std::endl;

        std::cout << "\nTraining Configuration:" << std::endl;
        std::cout << "+----------------------+----------------------+--------------"
                     "--------+"
                  << std::endl;
        std::cout << "| Optimizer            | Parameters           | Loss Function "
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

        initializaiton_summary();
    }

    template <typename BatchMethod, typename KFoldMethod>
    void train(const std::vector<std::vector<float>> &inputs, const std::vector<std::vector<float>> &targets, const BatchMethod &batch_method, const KFoldMethod &kfold_method,
            int log_interval = 100, bool verbose = true, bool restore_best_model = true) {

        if (!layers_.empty()) {
            int NetworkOutput = layers_.back()->get_output_size();
            if (NetworkOutput > 0 && targets[0].size() != static_cast<size_t>(NetworkOutput)) {
                throw std::invalid_argument("Output size does not match network output layer size\n - [Input] Network: " + std::to_string(NetworkOutput) + "  ||  Target: " + std::to_string(targets[0].size()));
            }
        }

        if (!optimizer_) { // if not defined
            optimizer_ = Thot::Optimizer::SGD(0.01f);
            for (auto &L : layers_) {
                L->set_optimizer(optimizer_);
            }
        }

        if (!Istraining_) { // spiek neuron reset
            Istraining_ = true;
            for (auto &L : layers_) {
                L->set_training(true);
                if (auto spike = std::dynamic_pointer_cast<SpikeLayer>(L)) {
                    spike->reset_state();
                }
            }
        }

        auto total_start = std::chrono::high_resolution_clock::now();
        std::vector<float> epoch_times;
        std::vector<float> fold_losses;
        double best_val_loss = std::numeric_limits<double>::infinity();
        double best_val_acc = -std::numeric_limits<double>::infinity();
        ModelParams best_params;

        int folds = kfold_method.get_folds();
        bool new_best = false;
        int best_epoch = -1;
        int best_fold = -1;

        for (int fold = 0; fold < folds; ++fold) {
            if (folds > 1 && verbose) {
                std::cout << "\nTraining Fold " << fold + 1 << "/" << folds
                          << std::endl;
            }
            kfold_method.start_fold(*this, fold);
            std::vector<std::vector<float>> train_inputs, train_targets, val_inputs,
                val_targets;
            kfold_method.split(inputs, targets, fold, train_inputs, train_targets,
                   val_inputs, val_targets);

            for (int epoch = 0; epoch < batch_method.get_epochs(); ++epoch) {
                auto epoch_start = std::chrono::high_resolution_clock::now();

                double epoch_loss = batch_method.template train_epoch<Network>(
                        *this, train_inputs, train_targets, log_interval, verbose,
                        epoch + 1, batch_method.get_epochs(), fold);

                auto epoch_end = std::chrono::high_resolution_clock::now();
                float epoch_time =
                    std::chrono::duration<float>(epoch_end - epoch_start).count();
                epoch_times.push_back(epoch_time);
                std::cout.unsetf(std::ios_base::floatfield);

                double val_loss = 0.0;
                if (folds > 1) {
                    for (size_t i = 0; i < val_inputs.size(); ++i) {
                        std::vector<float> output =
                            forward(val_inputs[i],
                                    {1, static_cast<int>(val_inputs[i].size())});

                        Utils::Tensor prediction_tensor(
                            {1, static_cast<int>(output.size())});
                        prediction_tensor.upload(output);

                        Utils::Tensor target_tensor(
                            {1, static_cast<int>(val_targets[i].size())});
                        target_tensor.upload(val_targets[i]);

                        val_loss +=
                            loss_function_->compute(prediction_tensor, target_tensor);


                        int correct = 0;
                        for (size_t j = 0; j < output.size(); ++j) {
                            float pred = output[j] >= 0.5f ? 1.0f : 0.0f;
                            if (pred == val_targets[i][j])
                                ++correct;
                        }
                    }
                    val_loss /= val_inputs.size();

                    if (val_loss < best_val_loss) { // best state by accuracy
                        best_val_loss = val_loss;
                        best_params = capture_parameters();
                        best_epoch = epoch;
                        best_fold = fold;
                        new_best = true;
                    }
                    fold_losses.push_back(val_loss);
                }
                bool is_logging =
                    (epoch % log_interval == 0 ||
                     epoch == batch_method.get_epochs() - 1);
                if (is_logging) {
                    std::cout << "Epoch " << epoch + 1
                              << " - Average Loss: " << epoch_loss;
                    if (folds > 1) {
                        std::cout << " - Validation Loss: " << val_loss;
                        if (new_best)
                            std::cout << " *(" << epoch + 1 << ")*";

                    }
                    std::cout << std::endl;
                    new_best = false;
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
            std::cout << "------------ Best State ------------" << std::endl;
            std::cout << "Best Fold: " << best_fold + 1 << std::endl;
            std::cout << "Best Epoch: " << best_epoch + 1 << std::endl;
            std::cout << "Loss ["
                      << Thot::Losses::to_string(loss_function_->get_type())
                      << "]: " << best_val_loss << std::endl;
            std::cout << "------------------------------------" << std::endl;


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
