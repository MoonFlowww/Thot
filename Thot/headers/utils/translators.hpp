//
// Created by moonfloww on 28/08/2025.
//

#ifndef THOT_TRANSLATORS_H
#define THOT_TRANSLATORS_H
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>

namespace Thot {

    std::string format_time(float seconds) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);

        if (seconds < 1e-6) {
            oss << seconds * 1e9 << " ns";
        } else if (seconds < 1e-3) {
            oss << seconds * 1e6 << " us";
        } else if (seconds < 1.0) {
            oss << seconds * 1e3 << " ms";
        } else if (seconds < 60.0) {
            oss << seconds << " s";
        } else if (seconds < 3600.0) {
            int minutes = static_cast<int>(seconds / 60);
            float remaining_seconds = seconds - (minutes * 60);
            oss << minutes << " m " << remaining_seconds << " s";
        } else {
            int hours = static_cast<int>(seconds / 3600);
            int minutes = static_cast<int>((seconds - (hours * 3600)) / 60);
            float remaining_seconds = seconds - (hours * 3600) - (minutes * 60);
            oss << hours << " h " << minutes << " m " << remaining_seconds << " s";
        }
        std::cout.unsetf(std::ios_base::floatfield);
        std::cout << std::setprecision(6);
        return oss.str();
    }

    std::string format_samples_per_second(float samples_per_second) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);

        if (samples_per_second < 1e3) {
            oss << samples_per_second << " samples/s";
        } else if (samples_per_second < 1e6) {
            oss << samples_per_second / 1e3 << "K samples/s";
        } else if (samples_per_second < 1e9) {
            oss << samples_per_second / 1e6 << "M samples/s";
        } else {
            oss << samples_per_second / 1e9 << "B samples/s";
        }
        std::cout.unsetf(std::ios_base::floatfield);
        std::cout << std::setprecision(6);
        return oss.str();
    }

    std::string formatBytes(float bytes) {
        static const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unitIndex = 0;

        double value = bytes;
        while (value >= 1024.0 && unitIndex < 4) {
            value /= 1024.0;
            ++unitIndex;
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << value << " " << units[unitIndex];
        std::cout.unsetf(std::ios_base::floatfield);
        return oss.str();
    }

}
#endif //THOT_TRANSLATORS_H