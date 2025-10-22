#ifndef THOT_BLOCK_DETAILS_SEQUENTIAL_HPP
#define THOT_BLOCK_DETAILS_SEQUENTIAL_HPP

#include <vector>

#include "../../layer/layer.hpp"

namespace Thot::Block::Details {

    struct SequentialDescriptor {
        std::vector<::Thot::Layer::Descriptor> layers{};
    };

}  // namespace Thot::Block::Details

#endif // THOT_BLOCK_DETAILS_SEQUENTIAL_HPP