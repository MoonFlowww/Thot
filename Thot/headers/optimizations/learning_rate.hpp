#pragma once
#include <functional>

namespace Thot {

    enum class LearningRate {
        Constant,
        Schedule,
        Function
    };

    using LrFn = std::function<float(int epoch, int fold)>;

} // namespace Thot