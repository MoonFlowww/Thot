#ifndef OMNI_PLOT_DETAILS_STATISTICS_HPP
#define OMNI_PLOT_DETAILS_STATISTICS_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace Omni::Plot::Details {
    inline double compute_kolmogorov_smirnov(const std::vector<double>& probs,
                                             const std::vector<int>& outcomes)
    {
        if (probs.empty() || probs.size() != outcomes.size()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        std::vector<double> positive;
        std::vector<double> negative;
        positive.reserve(probs.size());
        negative.reserve(probs.size());

        for (std::size_t i = 0; i < probs.size(); ++i) {
            if (outcomes[i] == 1) {
                positive.push_back(probs[i]);
            } else {
                negative.push_back(probs[i]);
            }
        }

        if (positive.empty() || negative.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        std::sort(positive.begin(), positive.end());
        std::sort(negative.begin(), negative.end());

        std::size_t i = 0;
        std::size_t j = 0;
        double max_diff = 0.0;

        while (i < positive.size() || j < negative.size()) {
            double threshold = 0.0;
            if (j >= negative.size() || (i < positive.size() && positive[i] <= negative[j])) {
                threshold = positive[i];
                while (i < positive.size() && positive[i] <= threshold) {
                    ++i;
                }
            } else {
                threshold = negative[j];
                while (j < negative.size() && negative[j] <= threshold) {
                    ++j;
                }
            }

            const double cdf_pos = static_cast<double>(i) / static_cast<double>(positive.size());
            const double cdf_neg = static_cast<double>(j) / static_cast<double>(negative.size());
            max_diff = std::max(max_diff, std::abs(cdf_pos - cdf_neg));
        }

        return max_diff;
    }
}

#endif // OMNI_PLOT_DETAILS_STATISTICS_HPP