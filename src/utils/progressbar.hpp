#ifndef LIBOMNI_PROGRESSBAR_HPP
#define LIBOMNI_PROGRESSBAR_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace Omni::Utils {
    class ProgressBar {
    public:
        ProgressBar(std::int64_t total, std::string label, std::size_t width = 30)
            : total_(std::max<std::int64_t>(total, 0)),
              label_(std::move(label)),
              width_(std::max<std::size_t>(width, static_cast<std::size_t>(1))),
              last_units_(-1),
              finished_(false) {}

        void update(std::int64_t current) {
            if (finished_ || total_ <= 0) {
                return;
            }
            if (current < 0) {
                current = 0;
            }
            if (current > total_) {
                current = total_;
            }

            const double ratio = static_cast<double>(current) /
                                 static_cast<double>(std::max<std::int64_t>(total_, 1));
            auto scaled_units = static_cast<std::int64_t>(std::round(ratio * width_ * 8.0));
            const std::int64_t max_units = static_cast<std::int64_t>(width_) * 8;
            if (scaled_units > max_units) {
                scaled_units = max_units;
            }

            if (scaled_units == last_units_ && current != total_) {
                return;
            }
            last_units_ = scaled_units;

            const std::size_t full_cells = static_cast<std::size_t>(scaled_units / 8);
            std::size_t partial_index = static_cast<std::size_t>(scaled_units % 8);

            std::ostringstream stream;
            stream << '\r' << label_ << " [";
            for (std::size_t i = 0; i < full_cells && i < width_; ++i) {
                stream << "\xE2\x96\x88";
            }

            const bool has_partial_cell = partial_index > 0 && full_cells < width_;
            if (has_partial_cell) {
                stream << PartialBlock(partial_index);
            }

            const std::size_t printed_cells = full_cells + (has_partial_cell ? 1 : 0);
            if (printed_cells < width_) {
                stream << std::string(width_ - printed_cells, ' ');
            }

            stream << "] ";
            stream << std::setw(3) << static_cast<int>(std::round(ratio * 100.0)) << "% ";
            stream << '(' << current << '/' << total_ << ')';

            std::cout << stream.str() << std::flush;

            if (current == total_) {
                finish();
            }
        }

        void complete() {
            update(total_);
        }

        [[nodiscard]] std::int64_t total() const {
            return total_;
        }

    private:
        static const char* PartialBlock(std::size_t index) { // smooth pBar
            static constexpr const char* blocks[] = {
                "",
                "\xE2\x96\x8F",
                "\xE2\x96\x8E",
                "\xE2\x96\x8D",
                "\xE2\x96\x8C",
                "\xE2\x96\x8B",
                "\xE2\x96\x8A",
                "\xE2\x96\x89"
            };
            if (index >= (sizeof(blocks) / sizeof(blocks[0]))) {
                return blocks[(sizeof(blocks) / sizeof(blocks[0])) - 1];
            }
            return blocks[index];
        }

        void finish() {
            if (finished_) {
                return;
            }
            finished_ = true;
            std::cout << std::endl;
        }

        std::int64_t total_;
        std::string label_;
        std::size_t width_;
        std::int64_t last_units_;
        bool finished_;
    };
}

#endif // LIBOMNI_PROGRESSBAR_HPP