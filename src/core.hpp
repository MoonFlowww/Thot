#ifndef THOT_CORE_HPP
#define THOT_CORE_HPP
/*
 * This file correspond to the Core/Brain of the Wrapper, it will collect request from main.cpp, send orders to module-factories, and send function pointers to network.hpp.
 * In Order the keep network.hpp pure, with only what's needed to forward/backword the network
 * We must keep the runtime fast, by keeping "Do we use ..." in constexpr
 */

#include <tuple>

#include "layer/layer.hpp"
#include "network.hpp"

namespace thot::core {
    template <typename... Layers>
    [[nodiscard]] inline auto instantiate_layers() {
        if constexpr (sizeof...(Layers) == 0) {
            return std::tuple<>{};
        } else {
            return std::tuple<Layers...>{Layers{}...};
        }
    }

    template <typename... Layers>
    [[nodiscard]] inline auto make_network() {
        if constexpr (sizeof...(Layers) == 0) {
            return network::Sequential<>{};
        } else {
            return network::Sequential<Layers...>{instantiate_layers<Layers...>()};
        }
    }

    template <typename Architecture>
    struct builder;

    template <typename... Layers>
    struct builder<std::tuple<Layers...>> {
        using network_type = network::Sequential<Layers...>;

        static network_type build() {
            if constexpr (sizeof...(Layers) == 0) {
                return network_type{};
            } else {
                return network_type{instantiate_layers<Layers...>()};
            }
        }
    };
}
#endif //THOT_CORE_HPP