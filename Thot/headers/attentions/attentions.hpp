#pragma once

#include <memory>
#include <string>

#include "../layers/layers.hpp"
#include "../initializations/initializations.hpp"

namespace Thot {
    class Layer;

    namespace Attention {
        std::shared_ptr<Layer> MHA(int embed_dim, int num_heads, Initialization init, const std::string &name = "MHA");
        std::shared_ptr<Layer> MLA(int embed_dim, int num_heads, int latent_dim, Initialization init, const std::string &name = "MLA");
    }
}

#include "details/mha.hpp"
#include "details/mla.hpp"

namespace Thot {
    namespace Attention {
        inline std::shared_ptr<Layer> MHA(int embed_dim, int num_heads, Initialization init, const std::string &name)
        { return std::make_shared<MHAAttentionLayer>(embed_dim, num_heads, init, name);}
        inline std::shared_ptr<Layer> MLA(int embed_dim, int num_heads, int latent_dim, Initialization init, const std::string &name)
        { return std::make_shared<MLAAttentionLayer>(embed_dim, num_heads, latent_dim, init, name); }
    }
}


