#pragma once
#include <functional>

namespace Thot {

    using LrFn = std::function<float(int epoch, int fold)>;
    using LrSchedule = std::function<LrFn(float initial_lr)>;
} // namespace Thot