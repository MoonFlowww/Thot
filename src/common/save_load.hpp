#ifndef THOT_COMMON_SAVE_LOAD_HPP
#define THOT_COMMON_SAVE_LOAD_HPP
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "../activation/activation.hpp"
#include "../attention/attention.hpp"
#include "../block/block.hpp"
#include "../block/details/transformers/classic.hpp"
#include "../common/local.hpp"
#include "../initialization/initialization.hpp"
#include "../layer/layer.hpp"
#include "../optimizer/optimizer.hpp"
#include "../regularization/regularization.hpp"

namespace Thot::Common::SaveLoad {
    using PropertyTree = boost::property_tree::ptree;
    using ModuleDescriptor = std::variant<Layer::Descriptor, Block::Descriptor>;

    namespace Detail {

        inline std::string to_lower(std::string value)
        {
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
            return value;
        }

        template <class Numeric>
        Numeric get_numeric(const PropertyTree& tree, const std::string& key, const std::string& context)
        {
            static_assert(std::is_arithmetic_v<Numeric>, "Numeric type required for property tree extraction.");
            const auto value = tree.get_optional<Numeric>(key);
            if (!value) {
                std::ostringstream message;
                message << "Missing numeric field '" << key << "' in " << context;
                throw std::runtime_error(message.str());
            }
            return *value;
        }

        inline bool get_boolean(const PropertyTree& tree, const std::string& key, const std::string& context)
        {
            const auto value = tree.get_optional<bool>(key);
            if (!value) {
                std::ostringstream message;
                message << "Missing boolean field '" << key << "' in " << context;
                throw std::runtime_error(message.str());
            }
            return *value;
        }

        inline std::string get_string(const PropertyTree& tree, const std::string& key, const std::string& context)
        {
            const auto value = tree.get_optional<std::string>(key);
            if (!value) {
                std::ostringstream message;
                message << "Missing string field '" << key << "' in " << context;
                throw std::runtime_error(message.str());
            }
            return *value;
        }

        template <class T>
        std::vector<T> read_array(const PropertyTree& tree, const std::string& context)
        {
            std::vector<T> values;
            values.reserve(tree.size());
            for (const auto& child : tree) {
                try {
                    if constexpr (std::is_same_v<T, bool>) {
                        values.push_back(static_cast<bool>(child.second.get_value<int>()));
                    } else {
                        values.push_back(child.second.get_value<T>());
                    }
                } catch (const boost::property_tree::ptree_bad_data&) {
                    std::ostringstream message;
                    message << "Invalid array element in " << context;
                    throw std::runtime_error(message.str());
                }
            }
            return values;
        }

        template <class T>
        PropertyTree write_array(const std::vector<T>& values)
        {
            PropertyTree array;
            for (const auto& value : values) {
                PropertyTree element;
                element.put("", value);
                array.push_back({"", element});
            }
            return array;
        }

        inline std::string activation_type_to_string(Activation::Type type)
        {
            switch (type) {
                case Activation::Type::Identity: return "identity";
                case Activation::Type::ReLU: return "relu";
                case Activation::Type::Sigmoid: return "sigmoid";
                case Activation::Type::Tanh: return "tanh";
                case Activation::Type::LeakyReLU: return "leaky_relu";
                case Activation::Type::Softmax: return "softmax";
                case Activation::Type::SiLU: return "silu";
                case Activation::Type::GeLU: return "gelu";
                case Activation::Type::GLU: return "glu";
                case Activation::Type::SwiGLU: return "swiglu";
                case Activation::Type::dSiLU: return "dsilu";
                case Activation::Type::PSiLU: return "psilu";
                case Activation::Type::Mish: return "mish";
                case Activation::Type::Swish: return "swish";
            }
            throw std::runtime_error("Unsupported activation type during serialisation.");
        }

        inline Activation::Type activation_type_from_string(const std::string& value)
        {
            const auto lowered = to_lower(value);
            if (lowered == "identity") return Activation::Type::Identity;
            if (lowered == "relu") return Activation::Type::ReLU;
            if (lowered == "sigmoid") return Activation::Type::Sigmoid;
            if (lowered == "tanh") return Activation::Type::Tanh;
            if (lowered == "leaky_relu") return Activation::Type::LeakyReLU;
            if (lowered == "softmax") return Activation::Type::Softmax;
            if (lowered == "silu") return Activation::Type::SiLU;
            if (lowered == "gelu") return Activation::Type::GeLU;
            if (lowered == "glu") return Activation::Type::GLU;
            if (lowered == "swiglu") return Activation::Type::SwiGLU;
            if (lowered == "dsilu") return Activation::Type::dSiLU;
            if (lowered == "psilu") return Activation::Type::PSiLU;
            if (lowered == "mish") return Activation::Type::Mish;
            if (lowered == "swish") return Activation::Type::Swish;
            std::ostringstream message;
            message << "Unknown activation type '" << value << "'.";
            throw std::runtime_error(message.str());
        }

        inline std::string initialization_type_to_string(Initialization::Type type)
        {
            switch (type) {
                case Initialization::Type::Default: return "default";
                case Initialization::Type::XavierNormal: return "xavier_normal";
                case Initialization::Type::XavierUniform: return "xavier_uniform";
                case Initialization::Type::KaimingNormal: return "kaiming_normal";
                case Initialization::Type::KaimingUniform: return "kaiming_uniform";
                case Initialization::Type::ZeroBias: return "zero_bias";
                case Initialization::Type::Dirac: return "dirac";
                case Initialization::Type::Lyapunov: return "lyapunov";
            }
            throw std::runtime_error("Unsupported initialisation type during serialisation.");
        }

        inline Initialization::Type initialization_type_from_string(const std::string& value)
        {
            const auto lowered = to_lower(value);
            if (lowered == "default") return Initialization::Type::Default;
            if (lowered == "xavier_normal") return Initialization::Type::XavierNormal;
            if (lowered == "xavier_uniform") return Initialization::Type::XavierUniform;
            if (lowered == "kaiming_normal") return Initialization::Type::KaimingNormal;
            if (lowered == "kaiming_uniform") return Initialization::Type::KaimingUniform;
            if (lowered == "zero_bias") return Initialization::Type::ZeroBias;
            if (lowered == "dirac") return Initialization::Type::Dirac;
            if (lowered == "lyapunov") return Initialization::Type::Lyapunov;
            std::ostringstream message;
            message << "Unknown initialisation type '" << value << "'.";
            throw std::runtime_error(message.str());
        }

        inline std::string attention_variant_to_string(Attention::Variant variant)
        {
            switch (variant) {
                case Attention::Variant::Full: return "full";
                case Attention::Variant::Causal: return "causal";
            }
            throw std::runtime_error("Unsupported attention variant during serialisation.");
        }

        inline Attention::Variant attention_variant_from_string(const std::string& value)
        {
            const auto lowered = to_lower(value);
            if (lowered == "full") return Attention::Variant::Full;
            if (lowered == "causal") return Attention::Variant::Causal;
            std::ostringstream message;
            message << "Unknown attention variant '" << value << "'.";
            throw std::runtime_error(message.str());
        }

        namespace Classic = Block::Details::Transformer::Classic;

        inline std::string positional_encoding_type_to_string(Classic::PositionalEncodingType type)
        {
            switch (type) {
                case Classic::PositionalEncodingType::None: return "none";
                case Classic::PositionalEncodingType::Sinusoidal: return "sinusoidal";
                case Classic::PositionalEncodingType::Learned: return "learned";
            }
            throw std::runtime_error("Unsupported positional encoding type during serialisation.");
        }

        inline Classic::PositionalEncodingType positional_encoding_type_from_string(const std::string& value)
        {
            const auto lowered = to_lower(value);
            if (lowered == "none") return Classic::PositionalEncodingType::None;
            if (lowered == "sinusoidal") return Classic::PositionalEncodingType::Sinusoidal;
            if (lowered == "learned") return Classic::PositionalEncodingType::Learned;
            std::ostringstream message;
            message << "Unknown positional encoding type '" << value << "'.";
            throw std::runtime_error(message.str());
        }

        inline PropertyTree serialize_activation_descriptor(const Activation::Descriptor& descriptor)
        {
            PropertyTree tree;
            tree.put("type", activation_type_to_string(descriptor.type));
            return tree;
        }

        inline Activation::Descriptor deserialize_activation_descriptor(const PropertyTree& tree, const std::string& context)
        {
            Activation::Descriptor descriptor;
            descriptor.type = activation_type_from_string(get_string(tree, "type", context));
            return descriptor;
        }

        inline PropertyTree serialize_initialization_descriptor(const Initialization::Descriptor& descriptor)
        {
            PropertyTree tree;
            tree.put("type", initialization_type_to_string(descriptor.type));
            return tree;
        }

        inline Initialization::Descriptor deserialize_initialization_descriptor(const PropertyTree& tree,
                                                                                const std::string& context)
        {
            Initialization::Descriptor descriptor;
            descriptor.type = initialization_type_from_string(get_string(tree, "type", context));
            return descriptor;
        }

        inline PropertyTree serialize_classic_attention_options(const Classic::AttentionOptions& options)
        {
            PropertyTree tree;
            tree.put("embed_dim", options.embed_dim);
            tree.put("num_heads", options.num_heads);
            tree.put("harddropout", options.dropout);
            tree.put("bias", options.bias);
            tree.put("batch_first", options.batch_first);
            tree.put("variant", attention_variant_to_string(options.variant));
            return tree;
        }

        inline Classic::AttentionOptions deserialize_classic_attention_options(const PropertyTree& tree,
                                                                               const std::string& context)
        {
            Classic::AttentionOptions options;
            options.embed_dim = get_numeric<std::int64_t>(tree, "embed_dim", context);
            options.num_heads = get_numeric<std::int64_t>(tree, "num_heads", context);
            options.dropout = get_numeric<double>(tree, "harddropout", context);
            options.bias = get_boolean(tree, "bias", context);
            options.batch_first = get_boolean(tree, "batch_first", context);
            options.variant = attention_variant_from_string(get_string(tree, "variant", context));
            return options;
        }

        inline PropertyTree serialize_classic_feed_forward_options(const Classic::FeedForwardOptions& options)
        {
            PropertyTree tree;
            tree.put("embed_dim", options.embed_dim);
            tree.put("mlp_ratio", options.mlp_ratio);
            tree.put("bias", options.bias);
            tree.add_child("activation", serialize_activation_descriptor(options.activation));
            tree.add_child("initialization", serialize_initialization_descriptor(options.initialization));
            return tree;
        }

        inline Classic::FeedForwardOptions deserialize_classic_feed_forward_options(const PropertyTree& tree,
                                                                                    const std::string& context)
        {
            Classic::FeedForwardOptions options;
            options.embed_dim = get_numeric<std::int64_t>(tree, "embed_dim", context);
            options.mlp_ratio = get_numeric<double>(tree, "mlp_ratio", context);
            options.bias = get_boolean(tree, "bias", context);
            options.activation = deserialize_activation_descriptor(tree.get_child("activation"), context);
            options.initialization = deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            return options;
        }

        inline PropertyTree serialize_classic_layer_norm_options(const Classic::LayerNormOptions& options)
        {
            PropertyTree tree;
            tree.put("eps", options.eps);
            tree.put("elementwise_affine", options.elementwise_affine);
            return tree;
        }

        inline Classic::LayerNormOptions deserialize_classic_layer_norm_options(const PropertyTree& tree,
                                                                                const std::string& context)
        {
            Classic::LayerNormOptions options;
            options.eps = get_numeric<double>(tree, "eps", context);
            options.elementwise_affine = get_boolean(tree, "elementwise_affine", context);
            return options;
        }

        inline PropertyTree serialize_classic_positional_encoding_options(const Classic::PositionalEncodingOptions& options)
        {
            PropertyTree tree;
            tree.put("type", positional_encoding_type_to_string(options.type));
            tree.put("harddropout", options.dropout);
            tree.put("max_length", static_cast<std::uint64_t>(options.max_length));
            tree.put("batch_first", options.batch_first);
            return tree;
        }

        inline Classic::PositionalEncodingOptions deserialize_classic_positional_encoding_options(const PropertyTree& tree,
                                                                                                  const std::string& context)
        {
            Classic::PositionalEncodingOptions options;
            options.type = positional_encoding_type_from_string(get_string(tree, "type", context));
            options.dropout = get_numeric<double>(tree, "harddropout", context);
            options.max_length = static_cast<std::size_t>(get_numeric<std::uint64_t>(tree, "max_length", context));
            options.batch_first = get_boolean(tree, "batch_first", context);
            return options;
        }

        inline std::string pooling_variant_to_string(const Layer::Details::PoolingOptions& options)
        {
            return std::visit(
                [](const auto& concrete) -> std::string {
                    using OptionType = std::decay_t<decltype(concrete)>;
                    if constexpr (std::is_same_v<OptionType, Layer::Details::MaxPool1dOptions>) {
                        return "max1d";
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AvgPool1dOptions>) {
                        return "avg1d";
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveAvgPool1dOptions>) {
                        return "adaptive_avg1d";
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveMaxPool1dOptions>) {
                        return "adaptive_max1d";
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::MaxPool2dOptions>) {
                        return "max2d";
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AvgPool2dOptions>) {
                        return "avg2d";
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveAvgPool2dOptions>) {
                        return "adaptive_avg2d";
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveMaxPool2dOptions>) {
                        return "adaptive_max2d";
                    } else {
                        static_assert(sizeof(OptionType) == 0, "Unsupported pooling options variant.");
                    }
                },
                options);
        }

        inline Layer::Details::PoolingOptions pooling_variant_from_string(const std::string& value)
        {
            const auto lowered = to_lower(value);
            if (lowered == "max1d") return Layer::Details::PoolingOptions{Layer::Details::MaxPool1dOptions{}};
            if (lowered == "avg1d") return Layer::Details::PoolingOptions{Layer::Details::AvgPool1dOptions{}};
            if (lowered == "adaptive_avg1d") return Layer::Details::PoolingOptions{Layer::Details::AdaptiveAvgPool1dOptions{}};
            if (lowered == "adaptive_max1d") return Layer::Details::PoolingOptions{Layer::Details::AdaptiveMaxPool1dOptions{}};
            if (lowered == "max2d") return Layer::Details::PoolingOptions{Layer::Details::MaxPool2dOptions{}};
            if (lowered == "avg2d") return Layer::Details::PoolingOptions{Layer::Details::AvgPool2dOptions{}};
            if (lowered == "adaptive_avg2d") return Layer::Details::PoolingOptions{Layer::Details::AdaptiveAvgPool2dOptions{}};
            if (lowered == "adaptive_max2d") return Layer::Details::PoolingOptions{Layer::Details::AdaptiveMaxPool2dOptions{}};
            std::ostringstream message;
            message << "Unknown pooling variant '" << value << "'.";
            throw std::runtime_error(message.str());
        }

    } // namespace Detail

    inline PropertyTree serialize_optimizer(const Optimizer::Descriptor& descriptor)
    {
        PropertyTree tree;
        std::visit(
            [&](const auto& concrete) {
                using DescriptorType = std::decay_t<decltype(concrete)>;
                if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::SGDDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "sgd");
                    tree.put("options.learning_rate", options.learning_rate);
                    tree.put("options.momentum", options.momentum);
                    tree.put("options.dampening", options.dampening);
                    tree.put("options.weight_decay", options.weight_decay);
                    tree.put("options.nesterov", options.nesterov);
                    tree.put("options.maximize", options.maximize);
                } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdamWDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "adamw");
                    tree.put("options.learning_rate", options.learning_rate);
                    tree.put("options.beta1", options.beta1);
                    tree.put("options.beta2", options.beta2);
                    tree.put("options.eps", options.eps);
                    tree.put("options.weight_decay", options.weight_decay);
                    tree.put("options.amsgrad", options.amsgrad);
                } else {
                    static_assert(sizeof(DescriptorType) == 0, "Unsupported optimizer descriptor supplied.");
                }
            },
            descriptor);
        return tree;
    }

    inline Optimizer::Descriptor deserialize_optimizer(const PropertyTree& tree, const std::string& context)
    {
        const auto type = Detail::to_lower(Detail::get_string(tree, "type", context));
        if (type == "sgd") {
            Optimizer::Details::SGDOptions options;
            options.learning_rate = Detail::get_numeric<double>(tree, "options.learning_rate", context);
            options.momentum = Detail::get_numeric<double>(tree, "options.momentum", context);
            options.dampening = Detail::get_numeric<double>(tree, "options.dampening", context);
            options.weight_decay = Detail::get_numeric<double>(tree, "options.weight_decay", context);
            options.nesterov = Detail::get_boolean(tree, "options.nesterov", context);
            options.maximize = Detail::get_boolean(tree, "options.maximize", context);
            return Optimizer::Descriptor{Optimizer::Details::SGDDescriptor{options}};
        }
        if (type == "adamw") {
            Optimizer::Details::AdamWOptions options;
            options.learning_rate = Detail::get_numeric<double>(tree, "options.learning_rate", context);
            options.beta1 = Detail::get_numeric<double>(tree, "options.beta1", context);
            options.beta2 = Detail::get_numeric<double>(tree, "options.beta2", context);
            options.eps = Detail::get_numeric<double>(tree, "options.eps", context);
            options.weight_decay = Detail::get_numeric<double>(tree, "options.weight_decay", context);
            options.amsgrad = Detail::get_boolean(tree, "options.amsgrad", context);
            return Optimizer::Descriptor{Optimizer::Details::AdamWDescriptor{options}};
        }
        std::ostringstream message;
        message << "Unknown optimizer descriptor '" << type << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_regularization(const Regularization::Descriptor& descriptor)
    {
        PropertyTree tree;
        std::visit(
            [&](const auto& concrete) {
                using DescriptorType = std::decay_t<decltype(concrete)>;
                const auto& options = concrete.options;
                if constexpr (std::is_same_v<DescriptorType, Regularization::L2Descriptor>) {
                    tree.put("type", "l2");
                    tree.put("options.coefficient", options.coefficient);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::EWCDescriptor>) {
                    tree.put("type", "ewc");
                    tree.put("options.strength", options.strength);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::MASDescriptor>) {
                    tree.put("type", "mas");
                    tree.put("options.strength", options.strength);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::SIDescriptor>) {
                    tree.put("type", "si");
                    tree.put("options.strength", options.strength);
                    tree.put("options.damping", options.damping);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::NuclearNormDescriptor>) {
                    tree.put("type", "nuclear_norm");
                    tree.put("options.strength", options.strength);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::SWADescriptor>) {
                    tree.put("type", "swa");
                    tree.put("options.coefficient", options.coefficient);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::SWAGDescriptor>) {
                    tree.put("type", "swag");
                    tree.put("options.coefficient", options.coefficient);
                    tree.put("options.variance_epsilon", options.variance_epsilon);
                    tree.put("options.start_step", static_cast<std::uint64_t>(options.start_step));
                    tree.put("options.accumulation_stride", static_cast<std::uint64_t>(options.accumulation_stride));
                    tree.put("options.max_snapshots", static_cast<std::uint64_t>(options.max_snapshots));
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::FGEDescriptor>) {
                    tree.put("type", "fge");
                    tree.put("options.coefficient", options.coefficient);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::SFGEDescriptor>) {
                    tree.put("type", "sfge");
                    tree.put("options.coefficient", options.coefficient);
                } else {
                    static_assert(sizeof(DescriptorType) == 0, "Unsupported regularisation descriptor supplied.");
                }
            },
            descriptor);
        return tree;
    }

    inline Regularization::Descriptor deserialize_regularization(const PropertyTree& tree, const std::string& context)
    {
        const auto type = Detail::to_lower(Detail::get_string(tree, "type", context));
        if (type == "l2") {
            Regularization::Details::L2Options options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::L2Descriptor{options}};
        }
        if (type == "ewc") {
            Regularization::Details::EWCOptions options;
            options.strength = Detail::get_numeric<double>(tree, "options.strength", context);
            return Regularization::Descriptor{Regularization::Details::EWCDescriptor{options}};
        }
        if (type == "mas") {
            Regularization::Details::MASOptions options;
            options.strength = Detail::get_numeric<double>(tree, "options.strength", context);
            return Regularization::Descriptor{Regularization::Details::MASDescriptor{options}};
        }
        if (type == "si") {
            Regularization::Details::SIOptions options;
            options.strength = Detail::get_numeric<double>(tree, "options.strength", context);
            options.damping = Detail::get_numeric<double>(tree, "options.damping", context);
            return Regularization::Descriptor{Regularization::Details::SIDescriptor{options}};
        }
        if (type == "nuclear_norm") {
            Regularization::Details::NuclearNormOptions options;
            options.strength = Detail::get_numeric<double>(tree, "options.strength", context);
            return Regularization::Descriptor{Regularization::Details::NuclearNormDescriptor{options}};
        }
        if (type == "swa") {
            Regularization::Details::SWAOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::SWADescriptor{options}};
        }
        if (type == "swag") {
            Regularization::Details::SWAGOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            options.variance_epsilon = Detail::get_numeric<double>(tree, "options.variance_epsilon", context);
            options.start_step = static_cast<std::size_t>(Detail::get_numeric<std::uint64_t>(tree, "options.start_step", context));
            options.accumulation_stride = static_cast<std::size_t>(
                Detail::get_numeric<std::uint64_t>(tree, "options.accumulation_stride", context));
            options.max_snapshots = static_cast<std::size_t>(
                Detail::get_numeric<std::uint64_t>(tree, "options.max_snapshots", context));
            return Regularization::Descriptor{Regularization::Details::SWAGDescriptor{options}};
        }
        if (type == "fge") {
            Regularization::Details::FGEOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::FGEDescriptor{options}};
        }
        if (type == "sfge") {
            Regularization::Details::SFGEOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::SFGEDescriptor{options}};
        }
        std::ostringstream message;
        message << "Unknown regularisation descriptor '" << type << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_local_config(const LocalConfig& config)
    {
        PropertyTree tree;
        if (config.optimizer) {
            tree.add_child("optimizer", serialize_optimizer(*config.optimizer));
        }
        PropertyTree regularization;
        for (const auto& descriptor : config.regularization) {
            regularization.push_back({"", serialize_regularization(descriptor)});
        }
        tree.add_child("regularization", regularization);
        return tree;
    }

    inline LocalConfig deserialize_local_config(const PropertyTree& tree, const std::string& context)
    {
        LocalConfig config;
        if (const auto optimizer = tree.get_child_optional("optimizer")) {
            config.optimizer = deserialize_optimizer(*optimizer, context + " optimizer");
        }
        if (const auto regularization = tree.get_child_optional("regularization")) {
            for (const auto& node : *regularization) {
                config.regularization.push_back(deserialize_regularization(node.second, context + " regularization"));
            }
        }
        return config;
    }

    inline PropertyTree serialize_attention(const Attention::Descriptor& descriptor)
    {
        PropertyTree tree;
        std::visit(
            [&](const auto& concrete) {
                using DescriptorType = std::decay_t<decltype(concrete)>;
                if constexpr (std::is_same_v<DescriptorType, Attention::MultiHeadDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "multi_head");
                    tree.put("options.embed_dim", options.embed_dim);
                    tree.put("options.num_heads", options.num_heads);
                    tree.put("options.dropout", options.dropout);
                    tree.put("options.bias", options.bias);
                    tree.put("options.add_bias_kv", options.add_bias_kv);
                    tree.put("options.add_zero_attn", options.add_zero_attn);
                    tree.put("options.batch_first", options.batch_first);
                    tree.put("options.variant", Detail::attention_variant_to_string(options.variant));
                } else {
                    static_assert(sizeof(DescriptorType) == 0, "Unsupported attention descriptor supplied.");
                }
            },
            descriptor);
        return tree;
    }

    inline Attention::Descriptor deserialize_attention(const PropertyTree& tree, const std::string& context)
    {
        const auto type = Detail::to_lower(Detail::get_string(tree, "type", context));
        if (type != "multi_head") {
            std::ostringstream message;
            message << "Unsupported attention descriptor '" << type << "' in " << context;
            throw std::runtime_error(message.str());
        }
        Attention::MultiHeadOptions options;
        options.embed_dim = Detail::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
        options.num_heads = Detail::get_numeric<std::int64_t>(tree, "options.num_heads", context);
        options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
        options.bias = Detail::get_boolean(tree, "options.bias", context);
        options.add_bias_kv = Detail::get_boolean(tree, "options.add_bias_kv", context);
        options.add_zero_attn = Detail::get_boolean(tree, "options.add_zero_attn", context);
        options.batch_first = Detail::get_boolean(tree, "options.batch_first", context);
        options.variant = Detail::attention_variant_from_string(Detail::get_string(tree, "options.variant", context));
        return Attention::Descriptor{Attention::MultiHeadDescriptor{options}};
    }

    inline PropertyTree serialize_layer_descriptor(const Layer::Descriptor& descriptor)
    {
        PropertyTree tree;
        std::visit(
            [&](const auto& concrete) {
                using DescriptorType = std::decay_t<decltype(concrete)>;
                if constexpr (std::is_same_v<DescriptorType, Layer::FCDescriptor>) {
                    tree.put("type", "fc");
                    tree.put("options.in_features", concrete.options.in_features);
                    tree.put("options.out_features", concrete.options.out_features);
                    tree.put("options.bias", concrete.options.bias);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::Conv1dDescriptor>) {
                    tree.put("type", "conv1d");
                    tree.put("options.in_channels", concrete.options.in_channels);
                    tree.put("options.out_channels", concrete.options.out_channels);
                    tree.add_child("options.kernel_size", Detail::write_array(concrete.options.kernel_size));
                    tree.add_child("options.stride", Detail::write_array(concrete.options.stride));
                    tree.add_child("options.padding", Detail::write_array(concrete.options.padding));
                    tree.add_child("options.dilation", Detail::write_array(concrete.options.dilation));
                    tree.put("options.groups", concrete.options.groups);
                    tree.put("options.bias", concrete.options.bias);
                    tree.put("options.padding_mode", concrete.options.padding_mode);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::Conv2dDescriptor>) {
                    tree.put("type", "conv2d");
                    tree.put("options.in_channels", concrete.options.in_channels);
                    tree.put("options.out_channels", concrete.options.out_channels);
                    tree.add_child("options.kernel_size", Detail::write_array(concrete.options.kernel_size));
                    tree.add_child("options.stride", Detail::write_array(concrete.options.stride));
                    tree.add_child("options.padding", Detail::write_array(concrete.options.padding));
                    tree.add_child("options.dilation", Detail::write_array(concrete.options.dilation));
                    tree.put("options.groups", concrete.options.groups);
                    tree.put("options.bias", concrete.options.bias);
                    tree.put("options.padding_mode", concrete.options.padding_mode);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::BatchNorm2dDescriptor>) {
                    tree.put("type", "batch_norm2d");
                    tree.put("options.num_features", concrete.options.num_features);
                    tree.put("options.eps", concrete.options.eps);
                    tree.put("options.momentum", concrete.options.momentum);
                    tree.put("options.affine", concrete.options.affine);
                    tree.put("options.track_running_stats", concrete.options.track_running_stats);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::PoolingDescriptor>) {
                    tree.put("type", "pooling");
                    tree.put("options.variant", Detail::pooling_variant_to_string(concrete.options));
                    std::visit(
                        [&](const auto& options) {
                            using OptionType = std::decay_t<decltype(options)>;
                            if constexpr (std::is_same_v<OptionType, Layer::Details::MaxPool1dOptions>) {
                                tree.add_child("options.kernel_size", Detail::write_array(options.kernel_size));
                                tree.add_child("options.stride", Detail::write_array(options.stride));
                                tree.add_child("options.padding", Detail::write_array(options.padding));
                                tree.add_child("options.dilation", Detail::write_array(options.dilation));
                                tree.put("options.ceil_mode", options.ceil_mode);
                            } else if constexpr (std::is_same_v<OptionType, Layer::Details::AvgPool1dOptions>) {
                                tree.add_child("options.kernel_size", Detail::write_array(options.kernel_size));
                                tree.add_child("options.stride", Detail::write_array(options.stride));
                                tree.add_child("options.padding", Detail::write_array(options.padding));
                                tree.put("options.ceil_mode", options.ceil_mode);
                                tree.put("options.count_include_pad", options.count_include_pad);
                            } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveAvgPool1dOptions>) {
                                tree.add_child("options.output_size", Detail::write_array(options.output_size));
                            } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveMaxPool1dOptions>) {
                                tree.add_child("options.output_size", Detail::write_array(options.output_size));
                            } else if constexpr (std::is_same_v<OptionType, Layer::Details::MaxPool2dOptions>) {
                                tree.add_child("options.kernel_size", Detail::write_array(options.kernel_size));
                                tree.add_child("options.stride", Detail::write_array(options.stride));
                                tree.add_child("options.padding", Detail::write_array(options.padding));
                                tree.add_child("options.dilation", Detail::write_array(options.dilation));
                                tree.put("options.ceil_mode", options.ceil_mode);
                            } else if constexpr (std::is_same_v<OptionType, Layer::Details::AvgPool2dOptions>) {
                                tree.add_child("options.kernel_size", Detail::write_array(options.kernel_size));
                                tree.add_child("options.stride", Detail::write_array(options.stride));
                                tree.add_child("options.padding", Detail::write_array(options.padding));
                                tree.put("options.ceil_mode", options.ceil_mode);
                                tree.put("options.count_include_pad", options.count_include_pad);
                            } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveAvgPool2dOptions>) {
                                tree.add_child("options.output_size", Detail::write_array(options.output_size));
                            } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveMaxPool2dOptions>) {
                                tree.add_child("options.output_size", Detail::write_array(options.output_size));
                            }
                        },
                        concrete.options);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::HardDropoutDescriptor>) {
                    tree.put("type", "harddropout");
                    tree.put("options.probability", concrete.options.probability);
                    tree.put("options.inplace", concrete.options.inplace);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::SoftDropoutDescriptor>) {
                    tree.put("type", "softdropout");
                    tree.put("options.probability", concrete.options.probability);
                    tree.put("options.inplace", concrete.options.inplace);
                    tree.put("options.noise_mean", concrete.options.noise_mean);
                    tree.put("options.noise_std", concrete.options.noise_std);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::FlattenDescriptor>) {
                    tree.put("type", "flatten");
                    tree.put("options.start_dim", concrete.options.start_dim);
                    tree.put("options.end_dim", concrete.options.end_dim);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("local", serialize_local_config(concrete.local));
                    } else if constexpr (std::is_same_v<DescriptorType, Layer::RNNDescriptor>) {
                    tree.put("type", "rnn");
                    tree.put("options.input_size", concrete.options.input_size);
                    tree.put("options.hidden_size", concrete.options.hidden_size);
                    tree.put("options.num_layers", concrete.options.num_layers);
                    tree.put("options.dropout", concrete.options.dropout);
                    tree.put("options.batch_first", concrete.options.batch_first);
                    tree.put("options.bidirectional", concrete.options.bidirectional);
                    tree.put("options.nonlinearity", concrete.options.nonlinearity);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::LSTMDescriptor>) {
                    tree.put("type", "lstm");
                    tree.put("options.input_size", concrete.options.input_size);
                    tree.put("options.hidden_size", concrete.options.hidden_size);
                    tree.put("options.num_layers", concrete.options.num_layers);
                    tree.put("options.dropout", concrete.options.dropout);
                    tree.put("options.batch_first", concrete.options.batch_first);
                    tree.put("options.bidirectional", concrete.options.bidirectional);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::GRUDescriptor>) {
                    tree.put("type", "gru");
                    tree.put("options.input_size", concrete.options.input_size);
                    tree.put("options.hidden_size", concrete.options.hidden_size);
                    tree.put("options.num_layers", concrete.options.num_layers);
                    tree.put("options.dropout", concrete.options.dropout);
                    tree.put("options.batch_first", concrete.options.batch_first);
                    tree.put("options.bidirectional", concrete.options.bidirectional);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::StateSpaceDescriptor>) {
                    tree.put("type", "statespace");
                    tree.put("options.input_size", concrete.options.input_size);
                    tree.put("options.hidden_size", concrete.options.hidden_size);
                    tree.put("options.output_size", concrete.options.output_size);
                    tree.put("options.num_layers", concrete.options.num_layers);
                    tree.put("options.dropout", concrete.options.dropout);
                    tree.put("options.batch_first", concrete.options.batch_first);
                    tree.put("options.bidirectional", concrete.options.bidirectional);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else {
                    static_assert(sizeof(DescriptorType) == 0, "Unsupported layer descriptor supplied.");
                }
            },
            descriptor);
        return tree;
    }

    inline Layer::Descriptor deserialize_layer_descriptor(const PropertyTree& tree, const std::string& context)
    {
        const auto type = Detail::to_lower(Detail::get_string(tree, "type", context));
        if (type == "fc") {
            Layer::Details::FCDescriptor descriptor;
            descriptor.options.in_features = Detail::get_numeric<std::int64_t>(tree, "options.in_features", context);
            descriptor.options.out_features = Detail::get_numeric<std::int64_t>(tree, "options.out_features", context);
            descriptor.options.bias = Detail::get_boolean(tree, "options.bias", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "conv1d") {
            Layer::Details::Conv1dDescriptor descriptor;
            descriptor.options.in_channels = Detail::get_numeric<std::int64_t>(tree, "options.in_channels", context);
            descriptor.options.out_channels = Detail::get_numeric<std::int64_t>(tree, "options.out_channels", context);
            descriptor.options.kernel_size = Detail::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context);
            descriptor.options.stride = Detail::read_array<std::int64_t>(tree.get_child("options.stride"), context);
            descriptor.options.padding = Detail::read_array<std::int64_t>(tree.get_child("options.padding"), context);
            descriptor.options.dilation = Detail::read_array<std::int64_t>(tree.get_child("options.dilation"), context);
            descriptor.options.groups = Detail::get_numeric<std::int64_t>(tree, "options.groups", context);
            descriptor.options.bias = Detail::get_boolean(tree, "options.bias", context);
            descriptor.options.padding_mode = Detail::get_string(tree, "options.padding_mode", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "conv2d") {
            Layer::Details::Conv2dDescriptor descriptor;
            descriptor.options.in_channels = Detail::get_numeric<std::int64_t>(tree, "options.in_channels", context);
            descriptor.options.out_channels = Detail::get_numeric<std::int64_t>(tree, "options.out_channels", context);
            descriptor.options.kernel_size = Detail::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context);
            descriptor.options.stride = Detail::read_array<std::int64_t>(tree.get_child("options.stride"), context);
            descriptor.options.padding = Detail::read_array<std::int64_t>(tree.get_child("options.padding"), context);
            descriptor.options.dilation = Detail::read_array<std::int64_t>(tree.get_child("options.dilation"), context);
            descriptor.options.groups = Detail::get_numeric<std::int64_t>(tree, "options.groups", context);
            descriptor.options.bias = Detail::get_boolean(tree, "options.bias", context);
            descriptor.options.padding_mode = Detail::get_string(tree, "options.padding_mode", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "batch_norm2d") {
            Layer::Details::BatchNorm2dDescriptor descriptor;
            descriptor.options.num_features = Detail::get_numeric<std::int64_t>(tree, "options.num_features", context);
            descriptor.options.eps = Detail::get_numeric<double>(tree, "options.eps", context);
            descriptor.options.momentum = Detail::get_numeric<double>(tree, "options.momentum", context);
            descriptor.options.affine = Detail::get_boolean(tree, "options.affine", context);
            descriptor.options.track_running_stats = Detail::get_boolean(tree, "options.track_running_stats", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "pooling") {
            Layer::Details::PoolingDescriptor descriptor;
            descriptor.options = Detail::pooling_variant_from_string(Detail::get_string(tree, "options.variant", context));
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            std::visit(
                [&](auto& options) {
                    using OptionType = std::decay_t<decltype(options)>;
                    if constexpr (std::is_same_v<OptionType, Layer::Details::MaxPool1dOptions>) {
                        options.kernel_size = Detail::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context);
                        options.stride = Detail::read_array<std::int64_t>(tree.get_child("options.stride"), context);
                        options.padding = Detail::read_array<std::int64_t>(tree.get_child("options.padding"), context);
                        options.dilation = Detail::read_array<std::int64_t>(tree.get_child("options.dilation"), context);
                        options.ceil_mode = Detail::get_boolean(tree, "options.ceil_mode", context);
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AvgPool1dOptions>) {
                        options.kernel_size = Detail::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context);
                        options.stride = Detail::read_array<std::int64_t>(tree.get_child("options.stride"), context);
                        options.padding = Detail::read_array<std::int64_t>(tree.get_child("options.padding"), context);
                        options.ceil_mode = Detail::get_boolean(tree, "options.ceil_mode", context);
                        options.count_include_pad = Detail::get_boolean(tree, "options.count_include_pad", context);
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveAvgPool1dOptions>) {
                        options.output_size = Detail::read_array<std::int64_t>(tree.get_child("options.output_size"), context);
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveMaxPool1dOptions>) {
                        options.output_size = Detail::read_array<std::int64_t>(tree.get_child("options.output_size"), context);
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::MaxPool2dOptions>) {
                        options.kernel_size = Detail::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context);
                        options.stride = Detail::read_array<std::int64_t>(tree.get_child("options.stride"), context);
                        options.padding = Detail::read_array<std::int64_t>(tree.get_child("options.padding"), context);
                        options.dilation = Detail::read_array<std::int64_t>(tree.get_child("options.dilation"), context);
                        options.ceil_mode = Detail::get_boolean(tree, "options.ceil_mode", context);
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AvgPool2dOptions>) {
                        options.kernel_size = Detail::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context);
                        options.stride = Detail::read_array<std::int64_t>(tree.get_child("options.stride"), context);
                        options.padding = Detail::read_array<std::int64_t>(tree.get_child("options.padding"), context);
                        options.ceil_mode = Detail::get_boolean(tree, "options.ceil_mode", context);
                        options.count_include_pad = Detail::get_boolean(tree, "options.count_include_pad", context);
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveAvgPool2dOptions>) {
                        options.output_size = Detail::read_array<std::int64_t>(tree.get_child("options.output_size"), context);
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveMaxPool2dOptions>) {
                        options.output_size = Detail::read_array<std::int64_t>(tree.get_child("options.output_size"), context);
                    }
                },
                descriptor.options);
            return Layer::Descriptor{descriptor};
        }
        if (type == "harddropout") {
            Layer::Details::HardDropoutDescriptor descriptor;
            descriptor.options.probability = Detail::get_numeric<double>(tree, "options.probability", context);
            descriptor.options.inplace = Detail::get_boolean(tree, "options.inplace", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "softdropout") {
            Layer::Details::SoftDropoutDescriptor descriptor;
            descriptor.options.probability = Detail::get_numeric<double>(tree, "options.probability", context);
            descriptor.options.inplace = Detail::get_boolean(tree, "options.inplace", context);
            descriptor.options.noise_mean = Detail::get_numeric<double>(tree, "options.noise_mean", context);
            descriptor.options.noise_std = Detail::get_numeric<double>(tree, "options.noise_std", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "flatten") {
            Layer::Details::FlattenDescriptor descriptor;
            descriptor.options.start_dim = Detail::get_numeric<std::int64_t>(tree, "options.start_dim", context);
            descriptor.options.end_dim = Detail::get_numeric<std::int64_t>(tree, "options.end_dim", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "rnn") {
            Layer::Details::RNNDescriptor descriptor;
            descriptor.options.input_size = Detail::get_numeric<std::int64_t>(tree, "options.input_size", context);
            descriptor.options.hidden_size = Detail::get_numeric<std::int64_t>(tree, "options.hidden_size", context);
            descriptor.options.num_layers = Detail::get_numeric<std::int64_t>(tree, "options.num_layers", context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            descriptor.options.batch_first = Detail::get_boolean(tree, "options.batch_first", context);
            descriptor.options.bidirectional = Detail::get_boolean(tree, "options.bidirectional", context);
            descriptor.options.nonlinearity = Detail::get_string(tree, "options.nonlinearity", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "lstm") {
            Layer::Details::LSTMDescriptor descriptor;
            descriptor.options.input_size = Detail::get_numeric<std::int64_t>(tree, "options.input_size", context);
            descriptor.options.hidden_size = Detail::get_numeric<std::int64_t>(tree, "options.hidden_size", context);
            descriptor.options.num_layers = Detail::get_numeric<std::int64_t>(tree, "options.num_layers", context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            descriptor.options.batch_first = Detail::get_boolean(tree, "options.batch_first", context);
            descriptor.options.bidirectional = Detail::get_boolean(tree, "options.bidirectional", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "gru") {
            Layer::Details::GRUDescriptor descriptor;
            descriptor.options.input_size = Detail::get_numeric<std::int64_t>(tree, "options.input_size", context);
            descriptor.options.hidden_size = Detail::get_numeric<std::int64_t>(tree, "options.hidden_size", context);
            descriptor.options.num_layers = Detail::get_numeric<std::int64_t>(tree, "options.num_layers", context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            descriptor.options.batch_first = Detail::get_boolean(tree, "options.batch_first", context);
            descriptor.options.bidirectional = Detail::get_boolean(tree, "options.bidirectional", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "statespace") {
            Layer::Details::StateSpaceDescriptor descriptor;
            descriptor.options.input_size = Detail::get_numeric<std::int64_t>(tree, "options.input_size", context);
            descriptor.options.hidden_size = Detail::get_numeric<std::int64_t>(tree, "options.hidden_size", context);
            descriptor.options.output_size = Detail::get_numeric<std::int64_t>(tree, "options.output_size", context);
            descriptor.options.num_layers = Detail::get_numeric<std::int64_t>(tree, "options.num_layers", context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            descriptor.options.batch_first = Detail::get_boolean(tree, "options.batch_first", context);
            descriptor.options.bidirectional = Detail::get_boolean(tree, "options.bidirectional", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        std::ostringstream message;
        message << "Unknown layer descriptor '" << type << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_block_descriptor(const Block::Descriptor& descriptor)
    {
        PropertyTree tree;
        std::visit(
            [&](const auto& concrete) {
                using DescriptorType = std::decay_t<decltype(concrete)>;
                if constexpr (std::is_same_v<DescriptorType, Block::Details::SequentialDescriptor>) {
                    tree.put("type", "sequential");
                    PropertyTree layers;
                    for (const auto& layer : concrete.layers) {
                        layers.push_back({"", serialize_layer_descriptor(layer)});
                    }
                    tree.add_child("layers", layers);
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Block::Details::ResidualDescriptor>) {
                    tree.put("type", "residual");
                    PropertyTree layers;
                    for (const auto& layer : concrete.layers) {
                        layers.push_back({"", serialize_layer_descriptor(layer)});
                    }
                    tree.add_child("layers", layers);
                    tree.put("repeats", static_cast<std::uint64_t>(concrete.repeats));
                    tree.put("skip.use_projection", concrete.skip.use_projection);
                    if (concrete.skip.projection) {
                        tree.add_child("skip.projection", serialize_layer_descriptor(*concrete.skip.projection));
                    }
                    tree.add_child("output.final_activation",
                                   Detail::serialize_activation_descriptor(concrete.output.final_activation));
                    tree.put("output.dropout", concrete.output.dropout);
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Detail::Classic::EncoderDescriptor>) {
                    tree.put("type", "transformer_encoder");
                    tree.add_child("options.attention", Detail::serialize_classic_attention_options(concrete.options.attention));
                    tree.add_child("options.feed_forward",
                                   Detail::serialize_classic_feed_forward_options(concrete.options.feed_forward));
                    tree.add_child("options.layer_norm",
                                   Detail::serialize_classic_layer_norm_options(concrete.options.layer_norm));
                    tree.add_child("options.positional_encoding", Detail::serialize_classic_positional_encoding_options(
                                                                      concrete.options.positional_encoding));
                    tree.put("options.layers", static_cast<std::uint64_t>(concrete.options.layers));
                    tree.put("options.embed_dim", concrete.options.embed_dim);
                    tree.put("options.dropout", concrete.options.dropout);
                    PropertyTree layers;
                    for (const auto& layer : concrete.layers) {
                        PropertyTree entry;
                        entry.add_child("attention", serialize_attention(layer.attention));
                        entry.add_child("attention_dropout", serialize_layer_descriptor(layer.attention_dropout));
                        PropertyTree feed_forward_layers;
                        for (const auto& ff : layer.feed_forward) {
                            feed_forward_layers.push_back({"", serialize_layer_descriptor(ff)});
                        }
                        entry.add_child("feed_forward", feed_forward_layers);
                        entry.add_child("feed_forward_dropout", serialize_layer_descriptor(layer.feed_forward_dropout));
                        layers.push_back({"", entry});
                    }
                    tree.add_child("layers", layers);
                } else if constexpr (std::is_same_v<DescriptorType, Detail::Classic::DecoderDescriptor>) {
                    tree.put("type", "transformer_decoder");
                    tree.add_child("options.self_attention",
                                   Detail::serialize_classic_attention_options(concrete.options.self_attention));
                    tree.add_child("options.cross_attention",
                                   Detail::serialize_classic_attention_options(concrete.options.cross_attention));
                    tree.add_child("options.feed_forward",
                                   Detail::serialize_classic_feed_forward_options(concrete.options.feed_forward));
                    tree.add_child("options.layer_norm",
                                   Detail::serialize_classic_layer_norm_options(concrete.options.layer_norm));
                    tree.add_child("options.positional_encoding", Detail::serialize_classic_positional_encoding_options(
                                                                      concrete.options.positional_encoding));
                    tree.put("options.layers", static_cast<std::uint64_t>(concrete.options.layers));
                    tree.put("options.embed_dim", concrete.options.embed_dim);
                    tree.put("options.dropout", concrete.options.dropout);
                    PropertyTree layers;
                    for (const auto& layer : concrete.layers) {
                        PropertyTree entry;
                        entry.add_child("self_attention", serialize_attention(layer.self_attention));
                        entry.add_child("self_attention_dropout", serialize_layer_descriptor(layer.self_attention_dropout));
                        entry.add_child("cross_attention", serialize_attention(layer.cross_attention));
                        entry.add_child("cross_attention_dropout", serialize_layer_descriptor(layer.cross_attention_dropout));
                        PropertyTree feed_forward_layers;
                        for (const auto& ff : layer.feed_forward) {
                            feed_forward_layers.push_back({"", serialize_layer_descriptor(ff)});
                        }
                        entry.add_child("feed_forward", feed_forward_layers);
                        entry.add_child("feed_forward_dropout", serialize_layer_descriptor(layer.feed_forward_dropout));
                        layers.push_back({"", entry});
                    }
                    tree.add_child("layers", layers);
                } else {
                    static_assert(sizeof(DescriptorType) == 0, "Unsupported block descriptor supplied.");
                }
            },
            descriptor);
        return tree;
    }

    inline Block::Descriptor deserialize_block_descriptor(const PropertyTree& tree, const std::string& context)
    {
        const auto type = Detail::to_lower(Detail::get_string(tree, "type", context));
        if (type == "sequential") {
            Block::Details::SequentialDescriptor descriptor;
            for (const auto& node : tree.get_child("layers")) {
                descriptor.layers.push_back(deserialize_layer_descriptor(node.second, context + " sequential layer"));
            }
            descriptor.local = deserialize_local_config(tree.get_child("local"), context + " sequential local");
            return Block::Descriptor{descriptor};
        }
        if (type == "residual") {
            Block::Details::ResidualDescriptor descriptor;
            for (const auto& node : tree.get_child("layers")) {
                descriptor.layers.push_back(deserialize_layer_descriptor(node.second, context + " residual layer"));
            }
            descriptor.repeats = static_cast<std::size_t>(Detail::get_numeric<std::uint64_t>(tree, "repeats", context));
            descriptor.skip.use_projection = Detail::get_boolean(tree, "skip.use_projection", context);
            if (const auto projection = tree.get_child_optional("skip.projection")) {
                descriptor.skip.projection = deserialize_layer_descriptor(*projection, context + " residual projection");
            }
            descriptor.output.final_activation =
                Detail::deserialize_activation_descriptor(tree.get_child("output.final_activation"), context);
            descriptor.output.dropout = Detail::get_numeric<double>(tree, "output.dropout", context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context + " residual local");
            return Block::Descriptor{descriptor};
        }
        if (type == "transformer_encoder") {
            Detail::Classic::EncoderDescriptor descriptor;
            descriptor.options.attention =
                Detail::deserialize_classic_attention_options(tree.get_child("options.attention"), context);
            descriptor.options.feed_forward =
                Detail::deserialize_classic_feed_forward_options(tree.get_child("options.feed_forward"), context);
            descriptor.options.layer_norm =
                Detail::deserialize_classic_layer_norm_options(tree.get_child("options.layer_norm"), context);
            descriptor.options.positional_encoding = Detail::deserialize_classic_positional_encoding_options(
                tree.get_child("options.positional_encoding"), context);
            descriptor.options.layers =
                static_cast<std::size_t>(Detail::get_numeric<std::uint64_t>(tree, "options.layers", context));
            descriptor.options.embed_dim = Detail::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            for (const auto& node : tree.get_child("layers")) {
                Detail::Classic::EncoderLayerDescriptor layer;
                layer.attention = deserialize_attention(node.second.get_child("attention"), context + " encoder attention");
                layer.attention_dropout = deserialize_layer_descriptor(node.second.get_child("attention_dropout"),
                                                                       context + " encoder attention dropout");
                for (const auto& feed_forward : node.second.get_child("feed_forward")) {
                    layer.feed_forward.push_back(
                        deserialize_layer_descriptor(feed_forward.second, context + " encoder feed-forward"));
                }
                layer.feed_forward_dropout = deserialize_layer_descriptor(node.second.get_child("feed_forward_dropout"),
                                                                           context + " encoder feed-forward dropout");
                descriptor.layers.push_back(std::move(layer));
            }
            return Block::Descriptor{descriptor};
        }
        if (type == "transformer_decoder") {
            Detail::Classic::DecoderDescriptor descriptor;
            descriptor.options.self_attention =
                Detail::deserialize_classic_attention_options(tree.get_child("options.self_attention"), context);
            descriptor.options.cross_attention =
                Detail::deserialize_classic_attention_options(tree.get_child("options.cross_attention"), context);
            descriptor.options.feed_forward =
                Detail::deserialize_classic_feed_forward_options(tree.get_child("options.feed_forward"), context);
            descriptor.options.layer_norm =
                Detail::deserialize_classic_layer_norm_options(tree.get_child("options.layer_norm"), context);
            descriptor.options.positional_encoding = Detail::deserialize_classic_positional_encoding_options(
                tree.get_child("options.positional_encoding"), context);
            descriptor.options.layers =
                static_cast<std::size_t>(Detail::get_numeric<std::uint64_t>(tree, "options.layers", context));
            descriptor.options.embed_dim = Detail::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            for (const auto& node : tree.get_child("layers")) {
                Detail::Classic::DecoderLayerDescriptor layer;
                layer.self_attention = deserialize_attention(node.second.get_child("self_attention"),
                                                             context + " decoder self-attention");
                layer.self_attention_dropout = deserialize_layer_descriptor(node.second.get_child("self_attention_dropout"),
                                                                            context + " decoder self-attention dropout");
                layer.cross_attention = deserialize_attention(node.second.get_child("cross_attention"),
                                                              context + " decoder cross-attention");
                layer.cross_attention_dropout = deserialize_layer_descriptor(node.second.get_child("cross_attention_dropout"),
                                                                             context + " decoder cross-attention dropout");
                for (const auto& feed_forward : node.second.get_child("feed_forward")) {
                    layer.feed_forward.push_back(deserialize_layer_descriptor(feed_forward.second, context + " decoder feed-forward"));
                }
                layer.feed_forward_dropout = deserialize_layer_descriptor(node.second.get_child("feed_forward_dropout"),
                                                                           context + " decoder feed-forward dropout");
                descriptor.layers.push_back(std::move(layer));
            }
            return Block::Descriptor{descriptor};
        }
        std::ostringstream message;
        message << "Unknown block descriptor '" << type << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_module_descriptor(const ModuleDescriptor& descriptor)
    {
        PropertyTree tree;
        if (std::holds_alternative<Layer::Descriptor>(descriptor)) {
            tree.put("kind", "layer");
            tree.add_child("descriptor", serialize_layer_descriptor(std::get<Layer::Descriptor>(descriptor)));
        } else {
            tree.put("kind", "block");
            tree.add_child("descriptor", serialize_block_descriptor(std::get<Block::Descriptor>(descriptor)));
        }
        return tree;
    }

    inline ModuleDescriptor deserialize_module_descriptor(const PropertyTree& tree, const std::string& context)
    {
        const auto kind = Detail::to_lower(Detail::get_string(tree, "kind", context));
        if (kind == "layer") {
            return ModuleDescriptor{deserialize_layer_descriptor(tree.get_child("descriptor"), context + " layer")};
        }
        if (kind == "block") {
            return ModuleDescriptor{deserialize_block_descriptor(tree.get_child("descriptor"), context + " block")};
        }
        std::ostringstream message;
        message << "Unknown module kind '" << kind << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_module_list(const std::vector<ModuleDescriptor>& descriptors)
    {
        PropertyTree tree;
        for (const auto& descriptor : descriptors) {
            tree.push_back({"", serialize_module_descriptor(descriptor)});
        }
        return tree;
    }

    inline std::vector<ModuleDescriptor> deserialize_module_list(const PropertyTree& tree, const std::string& context)
    {
        std::vector<ModuleDescriptor> descriptors;
        descriptors.reserve(tree.size());
        for (const auto& node : tree) {
            descriptors.push_back(deserialize_module_descriptor(node.second, context));
        }
        return descriptors;
    }

    inline void write_json_file(const std::filesystem::path& path, const PropertyTree& tree)
    {
        std::ofstream stream(path);
        if (!stream) {
            std::ostringstream message;
            message << "Failed to open '" << path.string() << "' for writing.";
            throw std::runtime_error(message.str());
        }
        boost::property_tree::write_json(stream, tree, true);
    }

    inline PropertyTree read_json_file(const std::filesystem::path& path)
    {
        PropertyTree tree;
        boost::property_tree::read_json(path.string(), tree);
        return tree;
    }
}
#endif // THOT_COMMON_SAVE_LOAD_HPP