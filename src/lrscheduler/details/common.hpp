#ifndef OMNI_LRSCHEDULER_COMMON_HPP
#define OMNI_LRSCHEDULER_COMMON_HPP

namespace Omni::LrScheduler::Details {

    class Scheduler {
    public:
        virtual ~Scheduler() = default;
        virtual void step() = 0;
    };

}  // namespace Omni::LrScheduler::Details

#endif //OMNI_LRSCHEDULER_COMMON_HPP