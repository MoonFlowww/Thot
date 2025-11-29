#ifndef OMNI_COMMON_GRAPH_HPP
#define OMNI_COMMON_GRAPH_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace Omni::Layer::Details {
    struct RegisteredLayer;
}

namespace Omni {
    enum class MergePolicy {
        Strict,
        Broadcast,
        Stack
    };

    struct Port {
        enum class Kind {
            Input,
            Output,
            Module,
            Join
        };

        Kind kind{Kind::Module};
        std::string identifier{};
        std::string attribute{};
        std::string representation{};
        MergePolicy merge_policy{MergePolicy::Strict};
        std::optional<std::size_t> node_index{};
        std::optional<std::size_t> join_index{};
        std::vector<std::string> join_members{};
        std::optional<int64_t> join_dimension{};

        Port() = default;

        Port(Kind kind, std::string identifier, std::string attribute, MergePolicy merge)
            : kind(kind), identifier(std::move(identifier)), attribute(std::move(attribute)), representation(build_representation()), merge_policy(merge) {}

        [[nodiscard]] bool is_input() const noexcept { return kind == Kind::Input; }
        [[nodiscard]] bool is_output() const noexcept { return kind == Kind::Output; }
        [[nodiscard]] bool is_module() const noexcept { return kind == Kind::Module; }
        [[nodiscard]] bool is_join() const noexcept { return kind == Kind::Join; }
        [[nodiscard]] const std::string& describe() const noexcept { return representation; }
        [[nodiscard]] std::string storage_key() const
        {
            if (!join_members.empty()) {
                std::string key = identifier;
                if (!attribute.empty()) {
                    key.append(":");
                    key.append(attribute);
                }
                return key;
            }
            if (attribute.empty()) {
                return identifier;
            }
            return identifier + ":" + attribute;
        }

        void assign_node(std::size_t value) { node_index = value; }
        void assign_join(std::size_t value) { join_index = value; }

        static Port Module(std::string_view specification)
        {
            const auto trimmed = trim(specification);
            if (trimmed.empty()) {
                throw std::invalid_argument("Port::Module requires a non-empty specification.");
            }

            auto [token, attribute] = split_token(trimmed);
            token = trim(token);
            attribute = trim(attribute);

            Port port{};
            port.attribute.assign(attribute.begin(), attribute.end());
            port.representation.assign(trimmed.begin(), trimmed.end());

            if (token.empty()) {
                throw std::invalid_argument(
                    "Port::Module encountered an empty identifier in '" + port.representation + "'.");
            }


            port.kind = Kind::Module;
            port.identifier.assign(token.begin(), token.end());


            if (port.identifier.empty()) {
                throw std::invalid_argument(
                    "Port specification '" + port.representation + "' is missing an identifier.");
            }

            port.merge_policy = MergePolicy::Strict;
            return port;
        }
        static Port Join(std::string_view name, MergePolicy policy = MergePolicy::Strict)
        {
            const auto trimmed = trim(name);
            if (trimmed.empty()) {
                throw std::invalid_argument("Port::join requires a non-empty name.");
            }

            auto [token, attribute] = split_token(trimmed);
            token = trim(token);
            attribute = trim(attribute);

            if (token.empty()) {
                throw std::invalid_argument("Join port specification cannot be empty.");
            }

            Port port{};
            port.kind = Kind::Join;
            port.identifier.assign(token.begin(), token.end());
            port.attribute.assign(attribute.begin(), attribute.end());
            port.merge_policy = policy;
            std::string repr{"join("};
            repr.append(trimmed.begin(), trimmed.end());
            repr.push_back(')');
            port.representation = std::move(repr);
            return port;
        }

        static Port Join(
            std::initializer_list<std::string_view> names,
            MergePolicy policy = MergePolicy::Strict)
        {
            return Join(names, policy, std::nullopt);
        }

        static Port Join(
            std::initializer_list<std::string_view> names,
            MergePolicy policy,
            int64_t concat_dimension)
        {
            return Join(names, policy, std::optional<int64_t>{concat_dimension});
        }

        static Port Join(
            std::initializer_list<std::string_view> names,
            MergePolicy policy,
            std::optional<int64_t> concat_dimension)
        {
            if (names.size() == 0) {
                throw std::invalid_argument(
                    "Port::join requires at least one module name when using the aggregate form.");
            }

            std::vector<std::string> original_order;
            original_order.reserve(names.size());
            std::vector<std::string> canonical;
            canonical.reserve(names.size());

            for (auto name : names) {
                const auto trimmed = trim(name);
                if (trimmed.empty()) {
                    throw std::invalid_argument("Join specification contains an empty module name.");
                }
                original_order.emplace_back(trimmed.begin(), trimmed.end());
                canonical.push_back(original_order.back());
            }

            std::sort(canonical.begin(), canonical.end());
            canonical.erase(std::unique(canonical.begin(), canonical.end()), canonical.end());

            std::string identifier{"@join["};
            for (std::size_t index = 0; index < canonical.size(); ++index) {
                if (index > 0) {
                    identifier.append("|");
                }
                identifier.append(canonical[index]);
            }
            identifier.push_back(']');

            Port port{};
            port.kind = Kind::Join;
            port.identifier = std::move(identifier);
            port.merge_policy = policy;
            port.join_members = std::move(canonical);
            port.join_dimension = concat_dimension;

            if (concat_dimension) {
                port.attribute = "dim=" + std::to_string(*concat_dimension);
            }

            std::string repr{"join("};
            for (std::size_t index = 0; index < original_order.size(); ++index) {
                if (index > 0) {
                    repr.append(", ");
                }
                repr.append(original_order[index]);
            }
            if (concat_dimension) {
                repr.append("; dim=");
                repr.append(std::to_string(*concat_dimension));
            }
            repr.push_back(')');
            port.representation = std::move(repr);

            return port;
        }

        static Port Input(std::string_view name = "@input") {
            const auto trimmed = trim(name);
            if (trimmed.empty()) {
                throw std::invalid_argument("Port::Input requires a non-empty name.");
            }
            Port p{};
            p.kind = Kind::Input;
            if (!trimmed.empty() && trimmed.front()=='@' && trimmed != "@input") {
                throw std::invalid_argument(
                    "Unsupported input sentinel '" + std::string(trimmed) + "'. Use '@input' or an alias.");
            }
            p.identifier.assign(trimmed.begin(), trimmed.end());
            p.representation.assign(trimmed.begin(), trimmed.end());
            p.merge_policy = MergePolicy::Strict;
            return p;
        }

        static Port Output(std::string_view name = "@output") {
            const auto trimmed = trim(name);
            if (trimmed.empty()) {
                throw std::invalid_argument("Port::Output requires a non-empty name.");
            }
            Port p{};
            p.kind = Kind::Output;
            if (!trimmed.empty() && trimmed.front()=='@' && trimmed != "@output") {
                throw std::invalid_argument(
                    "Unsupported output sentinel '" + std::string(trimmed) + "'. Use '@output' or an alias.");
            }
            p.identifier.assign(trimmed.begin(), trimmed.end());
            p.representation.assign(trimmed.begin(), trimmed.end());
            p.merge_policy = MergePolicy::Strict;
            return p;
        }


    private:
        [[nodiscard]] std::string build_representation() const
        {
            if (!representation.empty()) {
                return representation;
            }
            std::string repr = identifier;
            if (!attribute.empty()) {
                repr.append(1, ':');
                repr.append(attribute);
            }
            return repr;
        }

        static std::pair<std::string_view, std::string_view> split_token(std::string_view token)
        {
            const auto position = token.find_first_of(".:");
            if (position == std::string_view::npos) {
                return {token, std::string_view{}};
            }
            return {token.substr(0, position), token.substr(position + 1)};
        }

        static std::string_view trim(std::string_view token)
        {
            while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front()))) {
                token.remove_prefix(1);
            }
            while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back()))) {
                token.remove_suffix(1);
            }
            return token;
        }
    };

    struct LinkSpec {
        Port source{};
        Port target{};

        LinkSpec() = default;
        LinkSpec(Port source, Port target) : source(std::move(source)), target(std::move(target)) {}
    };

    struct ModuleNameBinding {
        std::size_t entry{std::numeric_limits<std::size_t>::max()};
        std::size_t exit{std::numeric_limits<std::size_t>::max()};
        std::vector<std::size_t> layers{};

        [[nodiscard]] bool has_entry() const noexcept
        {
            return entry != std::numeric_limits<std::size_t>::max();
        }
    };

    struct CompiledNode {
        enum class Kind {
            Input,
            Module,
            Join,
            Output
        };

        Kind kind{Kind::Module};
        std::size_t index{std::numeric_limits<std::size_t>::max()};
        MergePolicy merge{MergePolicy::Strict};
        std::string label{};
        std::vector<std::size_t> inputs{};
        std::vector<std::size_t> outputs{};
    };

    struct CompiledStep {
        std::size_t node_index{std::numeric_limits<std::size_t>::max()};
        std::vector<std::size_t> dependencies{};
    };

    struct ExecutionStep {
        enum class Kind {
            Module,
            Join,
            Output
        };

        Kind kind{Kind::Module};
        std::size_t activation_index{std::numeric_limits<std::size_t>::max()};

        struct ModuleData {
            ::Omni::Layer::Details::RegisteredLayer* layer{nullptr};
            std::size_t input_index{std::numeric_limits<std::size_t>::max()};
        };

        struct JoinData {
            MergePolicy policy{MergePolicy::Strict};
            std::vector<std::size_t> producers{};
            std::size_t workspace_index{std::numeric_limits<std::size_t>::max()};
            std::optional<int64_t> concat_dimension{};
        };

        struct OutputData {
            std::size_t input_index{std::numeric_limits<std::size_t>::max()};
        };

        ModuleData module{};
        JoinData join{};
        OutputData output{};
    };

    struct JoinBuffer {
        std::size_t node_index{std::numeric_limits<std::size_t>::max()};
        MergePolicy policy{MergePolicy::Strict};
        std::vector<std::size_t> producers{};
        std::optional<int64_t> concat_dimension{};
    };

    struct GraphExecutionWorkspace {
        torch::Tensor input{};
        torch::Tensor output{};
        std::vector<torch::Tensor> node_buffers{};
        std::vector<std::vector<torch::Tensor>> join_scratch{};

        void invalidate() noexcept
        {
            input = torch::Tensor{};
            output = torch::Tensor{};
            node_buffers.clear();
            join_scratch.clear();
        }

        void ensure_node_capacity(std::size_t count)
        {
            if (node_buffers.size() != count) {
                node_buffers.resize(count);
            }
        }

        void ensure_join_scratch(const std::vector<JoinBuffer>& join_buffers)
        {
            if (join_scratch.size() != join_buffers.size()) {
                join_scratch.resize(join_buffers.size());
            }

            for (std::size_t index = 0; index < join_scratch.size(); ++index) {
                auto& scratch = join_scratch[index];
                scratch.reserve(join_buffers[index].producers.size());
            }
        }

        void bind_input(std::size_t index)
        {
            if (index >= node_buffers.size()) {
                ensure_node_capacity(index + 1);
            }
            node_buffers[index] = input;
        }

        void bind_output(std::size_t index)
        {
            if (index >= node_buffers.size()) {
                ensure_node_capacity(index + 1);
            }
            node_buffers[index] = output;
        }
    };
}

#endif