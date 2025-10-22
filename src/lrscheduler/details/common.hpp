#ifndef THOT_LRSCHEDULER_COMMON_HPP
#define THOT_LRSCHEDULER_COMMON_HPP

namespace Thot::LrScheduler::Details {

    class Scheduler {
    public:
        virtual ~Scheduler() = default;
        virtual void step() = 0;
    };

}  // namespace Thot::LrScheduler::Details

#endif //THOT_LRSCHEDULER_COMMON_HPP