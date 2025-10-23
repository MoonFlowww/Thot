#ifndef THOT_POSITIONAL_ENCODING_HPP
#define THOT_POSITIONAL_ENCODING_HPP


#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <c10/util/Optional.h>

#include <torch/torch.h>

namespace Thot::Layer::Details {
    enum class PositionalEncodingType {
        None,
        Sinusoidal,
        Learned,
    };

    struct PositionalEncodingOptions {
        PositionalEncodingType type{PositionalEncodingType::None};
        double dropout{0.0};
        std::size_t max_length{2048};
        bool batch_first{true};
    };

    class SinusoidalPositionalEncodingImpl : public torch::nn::Module {
    public:
        explicit SinusoidalPositionalEncodingImpl(std::int64_t embedding_dim,
                                                  PositionalEncodingOptions options = {})
            : embedding_dim_(embedding_dim), options_(std::move(options))
        {
            if (embedding_dim_ <= 0) {
                throw std::invalid_argument("Sinusoidal positional encoding requires a positive embedding dimension.");
            }
            if (options_.max_length == 0) {
                throw std::invalid_argument("Sinusoidal positional encoding requires a positive maximum sequence length.");
            }

            dropout_ = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(options_.dropout)));
            positional_encoding_ = build_encoding(options_.max_length, embedding_dim_);
            register_buffer("positional_encoding", positional_encoding_);
        }

        [[nodiscard]] std::int64_t embedding_dim() const noexcept { return embedding_dim_; }
        [[nodiscard]] const PositionalEncodingOptions& options() const noexcept { return options_; }

        torch::Tensor forward(torch::Tensor input)
        {
            const auto seq_dim = options_.batch_first ? 1 : 0;
            const auto seq_len = input.size(seq_dim);
            if (seq_len > static_cast<std::int64_t>(options_.max_length)) {
                throw std::out_of_range("Input sequence length exceeds the configured maximum for sinusoidal positional encoding.");
            }

            auto positional = positional_encoding_.index({torch::indexing::Slice(0, seq_len)});
            positional = positional.to(input.device(), input.dtype());

            if (options_.batch_first) {
                positional = positional.unsqueeze(0);
            } else {
                positional = positional.unsqueeze(1);
            }

            input = input + positional;
            return dropout_->forward(input);
        }

    private:
        static torch::Tensor build_encoding(std::size_t length, std::int64_t embedding_dim)
        {
            auto position = torch::arange(static_cast<std::int64_t>(length), torch::TensorOptions().dtype(torch::kFloat32));
            auto div_term = torch::arange(0, embedding_dim, 2, torch::TensorOptions().dtype(torch::kFloat32));
            div_term = torch::exp(-div_term * (std::log(10000.0) / static_cast<double>(embedding_dim)));

            auto encoding = torch::zeros({static_cast<std::int64_t>(length), embedding_dim},
                                         torch::TensorOptions().dtype(torch::kFloat32));

            auto sin_terms = torch::sin(position.unsqueeze(1) * div_term);
            auto cos_terms = torch::cos(position.unsqueeze(1) * div_term);

            using torch::indexing::Slice;
            encoding.index_put_({Slice(), Slice(0, c10::nullopt, 2)}, sin_terms);
            if (embedding_dim > 1) {
                encoding.index_put_({Slice(), Slice(1, c10::nullopt, 2)}, cos_terms);
            }

            return encoding;
        }

        std::int64_t embedding_dim_{};
        PositionalEncodingOptions options_{};
        torch::Tensor positional_encoding_{};
        torch::nn::Dropout dropout_{nullptr};
    };

    TORCH_MODULE(SinusoidalPositionalEncoding);

    class LearnedPositionalEncodingImpl : public torch::nn::Module {
    public:
        explicit LearnedPositionalEncodingImpl(std::int64_t embedding_dim,
                                               PositionalEncodingOptions options = {})
            : embedding_dim_(embedding_dim), options_(std::move(options))
        {
            if (embedding_dim_ <= 0) {
                throw std::invalid_argument("Learned positional encoding requires a positive embedding dimension.");
            }
            if (options_.max_length == 0) {
                throw std::invalid_argument("Learned positional encoding requires a positive maximum sequence length.");
            }

            dropout_ = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(options_.dropout)));
            positional_embedding_ = torch::empty({static_cast<std::int64_t>(options_.max_length), embedding_dim_});
            register_parameter("positional_embedding", positional_embedding_);
            reset_parameters();
        }

        [[nodiscard]] std::int64_t embedding_dim() const noexcept { return embedding_dim_; }
        [[nodiscard]] const PositionalEncodingOptions& options() const noexcept { return options_; }

        torch::Tensor forward(torch::Tensor input)
        {
            const auto seq_dim = options_.batch_first ? 1 : 0;
            const auto seq_len = input.size(seq_dim);
            if (seq_len > static_cast<std::int64_t>(options_.max_length)) {
                throw std::out_of_range("Input sequence length exceeds the configured maximum for learned positional encoding.");
            }

            auto positional = positional_embedding_.index({torch::indexing::Slice(0, seq_len)});
            positional = positional.to(input.device(), input.dtype());

            if (options_.batch_first) {
                positional = positional.unsqueeze(0);
            } else {
                positional = positional.unsqueeze(1);
            }

            input = input + positional;
            return dropout_->forward(input);
        }

    private:
        void reset_parameters()
        {
            torch::nn::init::normal_(positional_embedding_, 0.0, 0.02);
        }

        std::int64_t embedding_dim_{};
        PositionalEncodingOptions options_{};
        torch::Tensor positional_embedding_{};
        torch::nn::Dropout dropout_{nullptr};
    };

    TORCH_MODULE(LearnedPositionalEncoding);
}

#endif //THOT_POSITIONAL_ENCODING_HPP