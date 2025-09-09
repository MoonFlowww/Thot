#pragma once
#include <functional>
#include <vector>
#include <cmath>
#include <algorithm>
#include "learning_rate.hpp"

namespace Thot {
namespace LrScheduler {

    using namespace std;
    using Thot::LrFn;
    using Thot::LrSchedule;

    inline LrSchedule StepDecay(int step_size, float gamma) {
        return [step_size, gamma](float lr) {
            return [lr, step_size, gamma](int epoch, int) {
                return lr * std::pow(gamma, epoch / step_size);
            };
        };
    }

    inline LrSchedule MultiStepDecay(vector<int> milestones, float gamma) {
        std::sort(milestones.begin(), milestones.end());
        return [milestones, gamma](float lr) {
            return [lr, milestones, gamma](int epoch, int) {
                float current = lr;
                for (int m : milestones) {
                    if (epoch >= m) current *= gamma;
                }
                return current;
            };
        };
    }

    inline LrSchedule ExponentialDecay(float gamma) {
        return [gamma](float lr) {
            return [lr, gamma](int epoch, int) {
                return lr * std::pow(gamma, epoch);
            };
        };
    }

    inline LrSchedule CosineAnnealingDecay(int T_max, float eta_min = 0.0f) {
        const float PI = 3.14159265358979323846f;
        return [T_max, eta_min, PI](float lr) {
            return [lr, T_max, eta_min, PI](int epoch, int) {
                if (epoch >= T_max) return eta_min;
                return eta_min + 0.5f * (lr - eta_min) * (1.0f + std::cos(PI * epoch / T_max));
            };
        };
    }

    inline LrSchedule CosineDecay(int T_max) {
        const float PI = 3.14159265358979323846f;
        return [T_max, PI](float lr) {
            return [lr, T_max, PI](int epoch, int) {
                return 0.5f * lr * (1.0f + std::cos(PI * epoch / T_max));
            };
        };
    }

    inline LrSchedule CosineDecayRestarts(int T_0, float T_mult = 2.0f) {
        const float PI = 3.14159265358979323846f;
        return [T_0, T_mult, PI](float lr) {
            return [lr, T_0, T_mult, PI](int epoch, int) mutable {
                static int T_curr = T_0;
                static int last_restart = 0;
                if (epoch - last_restart >= T_curr) {
                    last_restart = epoch;
                    T_curr = static_cast<int>(T_curr * T_mult);
                }
                float cos_inner = PI * (epoch - last_restart) / T_curr;
                return lr * 0.5f * (1.0f + std::cos(cos_inner));
            };
        };
    }

    inline LrSchedule PolynomialDecay(int total_steps, float power) {
        return [total_steps, power](float lr) {
            return [lr, total_steps, power](int epoch, int) {
                float step = static_cast<float>(std::min(epoch, total_steps));
                return lr * std::pow(1.0f - step / total_steps, power);
            };
        };
    }

    inline LrSchedule CyclicDecay(float max_lr, int step_size) {
        return [max_lr, step_size](float base_lr) {
            return [base_lr, max_lr, step_size](int epoch, int) {
                int cycle = std::floor(1 + epoch / (2.0f * step_size));
                float x = std::fabs(epoch / float(step_size) - 2 * cycle + 1);
                float scale = std::max(0.0f, 1.0f - x);
                return base_lr + (max_lr - base_lr) * scale;
            };
        };
    }

    inline LrSchedule OneCycleDecay(float max_lr, int total_steps, float pct_start = 0.3f) {
        const float PI = 3.14159265358979323846f;
        return [max_lr, total_steps, pct_start, PI](float base_lr) {
            return [base_lr, max_lr, total_steps, pct_start, PI](int epoch, int) {
                int up_steps = static_cast<int>(pct_start * total_steps);
                if (epoch < up_steps) {
                    float pct = float(epoch) / up_steps;
                    return base_lr + pct * (max_lr - base_lr);
                } else if (epoch <= total_steps) {
                    float pct = float(epoch - up_steps) / (total_steps - up_steps);
                    return max_lr - (max_lr - base_lr) * (0.5f * (1.0f + std::cos(PI * pct)));
                } else {
                    return base_lr;
                }
            };
        };
    }

    inline LrSchedule OnPlateauDecay(float factor, int patience) {
        return [factor, patience](float lr) {
            return [lr, factor, patience](int epoch, int) {
                return lr * std::pow(factor, epoch / patience);
            };
        };
    }

} // namespace LrScheduler
} // namespace Thot