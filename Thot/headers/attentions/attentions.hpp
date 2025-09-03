#pragma once

#include <memory>
#include <string>

#include "../layers/layers.hpp"
#include "../initializations/initializations.hpp"

namespace Thot {
    class Layer;

    namespace Attention {
        // Factory function returning a shared_ptr to a multi-head attention layer
        std::shared_ptr<Layer> MHA(int embed_dim,
                                   int num_heads,
                                   Initialization init,
                                   const std::string &name = "MHA");
    }
}

#include "details/mha.hpp"

namespace Thot {
    namespace Attention {
        inline std::shared_ptr<Layer> MHA(int embed_dim,
                                          int num_heads,
                                          Initialization init,
                                          const std::string &name) {
            return std::make_shared<MHAAttentionLayer>(embed_dim, num_heads, init, name);
        }
    }
}