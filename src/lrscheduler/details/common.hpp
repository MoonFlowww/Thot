#ifndef Nott_LRSCHEDULER_COMMON_HPP
#define Nott_LRSCHEDULER_COMMON_HPP

namespace Nott::LrScheduler::Details {

    class Scheduler {
    public:
        virtual ~Scheduler() = default;
        virtual void step() = 0;
    };

}  // namespace Nott::LrScheduler::Details

#endif //Nott_LRSCHEDULER_COMMON_HPP