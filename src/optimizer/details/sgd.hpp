#ifndef THOT_SGD_HPP
#define THOT_SGD_HPP
#include <torch/optim/sgd.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace thot::optimizer::details {
    // Option tags --------------------------------------------------------------

    template <double Value>
    struct learning_rate {
        static constexpr double value = Value;
    };

    template <double Value>
    struct momentum {
        static constexpr double value = Value;
    };

    template <double Value>
    struct weight_decay {
        static constexpr double value = Value;
    };

    struct disable_weight_decay final {
    };

    struct enable_nesterov final {
    };

    struct disable_nesterov final {
    };

    // Option parsing -----------------------------------------------------------

    namespace detail {

        template <typename T>
        struct is_learning_rate : std::false_type {
        };

        template <double V>
        struct is_learning_rate<learning_rate<V>> : std::true_type {
        };

        template <typename T>
        inline constexpr bool is_learning_rate_v = is_learning_rate<T>::value;


        template <typename T>
        struct is_momentum : std::false_type {
        };

        template <double V>
        struct is_momentum<momentum<V>> : std::true_type {
        };

        template <typename T>
        inline constexpr bool is_momentum_v = is_momentum<T>::value;


        template <typename T>
        struct is_weight_decay : std::false_type {
        };

        template <double V>
        struct is_weight_decay<weight_decay<V>> : std::true_type {
        };

        template <typename T>
        inline constexpr bool is_weight_decay_v = is_weight_decay<T>::value;


        template <typename T>
        inline constexpr bool is_disable_weight_decay_v = std::is_same_v<T, disable_weight_decay>;

        template <typename T>
        inline constexpr bool is_enable_nesterov_v = std::is_same_v<T, enable_nesterov>;

        template <typename T>
        inline constexpr bool is_disable_nesterov_v = std::is_same_v<T, disable_nesterov>;


        template <typename Option>
        constexpr void assign_learning_rate(double& value) {
            if constexpr (is_learning_rate_v<Option>) {
                value = Option::value;
            }
        }

        template <typename Option>
        constexpr void assign_momentum(double& value) {
            if constexpr (is_momentum_v<Option>) {
                value = Option::value;
            }
        }

        template <typename Option>
        constexpr void assign_weight_decay(double& value) {
            if constexpr (is_weight_decay_v<Option>) {
                value = Option::value;
            }
        }



        template <typename... Options>
        struct sgd_option_state {
            static constexpr double default_lr = 1e-3;
            static constexpr double default_momentum = 0.0;

            static constexpr std::size_t learning_rate_count = (std::size_t(is_learning_rate_v<Options>) + ... + 0);
            static constexpr std::size_t momentum_count = (std::size_t(is_momentum_v<Options>) + ... + 0);
            static constexpr std::size_t weight_decay_count = (std::size_t(is_weight_decay_v<Options>) + ... + 0);
            static constexpr std::size_t disable_weight_decay_count = (std::size_t(is_disable_weight_decay_v<Options>) + ... + 0);
            static constexpr std::size_t enable_nesterov_count = (std::size_t(is_enable_nesterov_v<Options>) + ... + 0);
            static constexpr std::size_t disable_nesterov_count = (std::size_t(is_disable_nesterov_v<Options>) + ... + 0);

            static_assert(learning_rate_count <= 1, "Duplicate learning_rate options detected");
            static_assert(momentum_count <= 1, "Duplicate momentum options detected");
            static_assert(weight_decay_count <= 1, "Duplicate weight_decay options detected");
            static_assert(disable_weight_decay_count <= 1, "Duplicate disable_weight_decay flags detected");
            static_assert(enable_nesterov_count <= 1, "Duplicate enable_nesterov flags detected");
            static_assert(disable_nesterov_count <= 1, "Duplicate disable_nesterov flags detected");
            static_assert(!(weight_decay_count > 0 && disable_weight_decay_count > 0),
                          "Conflicting weight_decay configuration flags");
            static_assert(!(enable_nesterov_count > 0 && disable_nesterov_count > 0),
                          "Conflicting Nesterov configuration flags");

            static constexpr bool has_learning_rate = learning_rate_count == 1;
            static constexpr bool has_momentum = momentum_count == 1;
            static constexpr bool has_weight_decay = weight_decay_count == 1;
            static constexpr bool has_disable_weight_decay = disable_weight_decay_count == 1;
            static constexpr bool has_enable_nesterov = enable_nesterov_count == 1;
            static constexpr bool has_disable_nesterov = disable_nesterov_count == 1;

            static constexpr double learning_rate = [] {
                double value = default_lr;
                (assign_learning_rate<Options>(value), ...);
                return value;
            }();

            static_assert(learning_rate > 0.0, "SGD requires a strictly positive learning rate");

            static constexpr double momentum = [] {
                double value = default_momentum;
                (assign_momentum<Options>(value), ...);
                return value;
            }();

            static_assert(momentum >= 0.0, "Momentum must be non-negative");

            static constexpr bool use_weight_decay = has_weight_decay && !has_disable_weight_decay;

            static constexpr double weight_decay = [] {
                double value = 0.0;
                (assign_weight_decay<Options>(value), ...);
                return value;
            }();

            static_assert(!use_weight_decay || weight_decay >= 0.0, "Weight decay must be non-negative");

            static constexpr bool use_nesterov = has_enable_nesterov ? true : false;

            static_assert(!(momentum_count > 0 && !(has_enable_nesterov || has_disable_nesterov)),
                          "Momentum requires an explicit Nesterov flag (enable_nesterov or disable_nesterov)");
            static_assert(!(use_nesterov && momentum == 0.0),
                          "Nesterov momentum requires a non-zero momentum value");
        };

    }  // namespace detail

    // Descriptor ---------------------------------------------------------------

    template <typename... Options>
    class sgd_descriptor {
    public:
        using option_state = detail::sgd_option_state<Options...>;
        using optimizer_type = torch::optim::SGD;

        static constexpr double learning_rate = option_state::learning_rate;
        static constexpr double momentum = option_state::momentum;
        static constexpr bool use_nesterov = option_state::use_nesterov;
        static constexpr bool use_weight_decay = option_state::use_weight_decay;
        static constexpr double weight_decay = option_state::weight_decay;

        template <typename ParameterContainer>
        explicit sgd_descriptor(ParameterContainer& parameters)
            : optimizer_(parameters, build_options()) {}

        optimizer_type& native() noexcept { return optimizer_; }
        const optimizer_type& native() const noexcept { return optimizer_; }

        void zero_grad() { optimizer_.zero_grad(); }
        void step() { optimizer_.step(); }

    private:
        static torch::optim::SGDOptions build_options() {
            torch::optim::SGDOptions opts(learning_rate);
            opts.momentum(momentum);
            opts.nesterov(use_nesterov);
            if (use_weight_decay) {
                opts.weight_decay(weight_decay);
            } else {
                opts.weight_decay(0.0);
            }
            return opts;
        }

        optimizer_type optimizer_;
    };

    // Factory -----------------------------------------------------------------

    template <typename... Options>
    struct sgd_factory {
        using descriptor_type = sgd_descriptor<Options...>;

        template <typename ParameterContainer>
        descriptor_type operator()(ParameterContainer& parameters) const {
            return descriptor_type{parameters};
        }
    };
}
#endif //THOT_SGD_HPP