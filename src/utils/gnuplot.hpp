#ifndef THOT_GNUPLOT_HPP
#define THOT_GNUPLOT_HPP

#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <iomanip>
namespace Thot::Utils {
    class Gnuplot {
    public:
        enum class PlotMode {
            Lines,
            Points,
            LinesPoints,
            Impulses,
            Dots,
            Boxes,
            FilledCurves
        };

        enum class AxisScale {
            Linear,       // identity
            Log,          // classic log(value)
            LogOneMinus,  // -log_b(1 - value) for [0,1)
            Logit         // log(value/(1-value)) for (0,1)
        };
        struct PlotStyle {
            PlotMode mode;
            std::optional<int> lineType;
            std::optional<double> lineWidth;
            std::optional<std::string> lineColor;
            std::optional<int> pointType;
            std::optional<double> pointSize;
            std::optional<std::string> extra;

            constexpr PlotStyle(
                PlotMode mode = PlotMode::Lines,
                std::optional<int> lineType = std::nullopt,
                std::optional<double> lineWidth = std::nullopt,
                std::optional<std::string> lineColor = std::nullopt,
                std::optional<int> pointType = std::nullopt,
                std::optional<double> pointSize = std::nullopt,
                std::optional<std::string> extra = std::nullopt)
                : mode(mode),
                  lineType(std::move(lineType)),
                  lineWidth(std::move(lineWidth)),
                  lineColor(std::move(lineColor)),
                  pointType(std::move(pointType)),
                  pointSize(std::move(pointSize)),
                  extra(std::move(extra)) {}
        };

        struct DataSet2D {
            std::vector<double> x{};
            std::vector<double> y{};
            std::string title{};
            PlotStyle style{};
        };

        struct DataSet3D {
            std::vector<double> x{};
            std::vector<double> y{};
            std::vector<double> z{};
            std::string title{};
            PlotStyle style{};
        };

        explicit Gnuplot(std::string command = "gnuplot")
            : command_(EnsurePersist(std::move(command))),
              pipe_(popen(command_.c_str(), "w")) {
            if (pipe_ == nullptr) {
                throw std::runtime_error("Failed to open pipe to gnuplot");
            }
        }

        ~Gnuplot() {
            close();
        }

        Gnuplot(const Gnuplot&) = delete;
        Gnuplot& operator=(const Gnuplot&) = delete;

        Gnuplot(Gnuplot&& other) noexcept
            : command_(std::move(other.command_)),
              pipe_(other.pipe_) {
            other.pipe_ = nullptr;
        }

        Gnuplot& operator=(Gnuplot&& other) noexcept {
            if (this != &other) {
                close();
                command_ = std::move(other.command_);
                pipe_ = other.pipe_;
                other.pipe_ = nullptr;
            }
            return *this;
        }

        [[nodiscard]] bool valid() const noexcept {
            return pipe_ != nullptr;
        }

        // Set printf-style numeric format for an axis (e.g. "%.3f")
        void setFormat(char axis, const std::string& fmt) {
            command(std::string("set format ") + axis + " '" + EscapeSingleQuotes(fmt) + "'");
        }

        // Main scaler: applies Linear, Log, LogOneMinus, or Logit to axis ('x'|'y'|'z')
        // For LogOneMinus, 'base' controls the logarithm base (default 10).
        void setAxisScale(char axis, AxisScale scale, double base = 10.0) {
            switch (scale) {
                case AxisScale::Linear:
                    unsetLogScale(axis);
                    unsetNonlinear(axis);
                    break;

                case AxisScale::Log: {
                    unsetNonlinear(axis);
                    setLogScale(axis, true, base);
                    break;
                }

                case AxisScale::LogOneMinus: {
                    // y' = -log_b(1 - y)  <=>  via (-log(1-ax)/log(base))
                    // inverse: y = 1 - exp(-y'*log(base))
                    unsetLogScale(axis);
                    const std::string ax(1, axis);
                    std::ostringstream fwd, inv;
                    fwd  << "(-log(1.0-" << ax << ")/log(" << base << "))";
                    inv  << "(1.0-exp(-"   << ax << "*log(" << base << ")))";
                    setNonlinear(axis, fwd.str(), inv.str());
                    break;
                }

                case AxisScale::Logit: {
                    // y' = log(y/(1-y)), inverse: y = 1/(1+exp(-y'))
                    unsetLogScale(axis);
                    const std::string ax(1, axis);
                    const std::string fwd = "(log(" + ax + "/(1.0-" + ax + ")))";
                    const std::string inv = "(1.0/(1.0+exp(-" + ax + ")))";
                    setNonlinear(axis, fwd, inv);
                    break;
                }
            }
        }

        // Convenience
        void setAxisScaleX(AxisScale s, double base = 10.0) { setAxisScale('x', s, base); }
        void setAxisScaleY(AxisScale s, double base = 10.0) { setAxisScale('y', s, base); }
        void setAxisScaleZ(AxisScale s, double base = 10.0) { setAxisScale('z', s, base); }

        // Tick helpers in DATA units (gnuplot maps them through the nonlinear transform)
        void unsetTics(char axis) { command(std::string("unset ") + axis + "tics"); }

        // Set ticks at given data values with default text = formatted value (fmt like "%.3f")
        void setTics(char axis, const std::vector<double>& values, const std::string& fmt = "%.3f") {
            std::ostringstream s; s << "set " << axis << "tics (";
            char buf[128]; bool first = true;
            for (double v : values) {
                if (!first) s << ", ";
                first = false;
                std::snprintf(buf, sizeof(buf), fmt.c_str(), v);
                s << "'" << EscapeSingleQuotes(buf) << "' " << v;
            }
            s << ")";
            command(s.str());
        }

        // Labeled ticks explicitly
        void setLabeledTics(char axis, const std::vector<std::pair<double,std::string>>& tics) {
            std::ostringstream s; s << "set " << axis << "tics (";
            for (std::size_t i = 0; i < tics.size(); ++i) {
                if (i) s << ", ";
                s << "'" << EscapeSingleQuotes(tics[i].second) << "' " << tics[i].first;
            }
            s << ")";
            command(s.str());
        }

        void command(const std::string& cmd) {
            ensureValid();
            writeLine(cmd);
            flush();
        }

        void setTerminal(const std::string& terminal) {
            command("set terminal " + terminal);
        }

        void setOutput(const std::string& file) {
            command("set output '" + EscapeSingleQuotes(file) + "'");
        }

        void unsetOutput() {
            command("unset output");
        }

        void setTitle(const std::string& title) {
            command("set title '" + EscapeSingleQuotes(title) + "'");
        }

        void setXLabel(const std::string& label) {
            command("set xlabel '" + EscapeSingleQuotes(label) + "'");
        }

        void setYLabel(const std::string& label) {
            command("set ylabel '" + EscapeSingleQuotes(label) + "'");
        }

        void setZLabel(const std::string& label) {
            command("set zlabel '" + EscapeSingleQuotes(label) + "'");
        }

        void setRange(const std::string& axis, double min, double max) {
            ensureValid();
            std::ostringstream stream;
            stream << "set " << axis << "range [" << min << ':' << max << ']';
            writeLine(stream.str());
            flush();
        }

        void setRange(char axis, double min, double max) {
            setRange(std::string(1, axis), min, max);
        }

        void unsetRange(const std::string& axis) {
            ensureValid();
            std::string cmd = std::string("unset ") + axis + "range";
            writeLine(cmd);
            flush();
        }

        void unsetRange(char axis) {
            unsetRange(std::string(1, axis));
        }

        void setLogScale(const std::string& axis, bool enable = true, double base = 10.0) {
            ensureValid();
            std::ostringstream stream;
            if (enable) {
                stream << "set logscale " << axis;
                if (base != 10.0) {
                    stream << ' ' << base;
                }
            } else {
                stream << "unset logscale " << axis;
            }
            writeLine(stream.str());
            flush();
        }

        void setLogScale(char axis, bool enable = true, double base = 10.0) {
            setLogScale(std::string(1, axis), enable, base);
        }

        void unsetLogScale(const std::string& axis) {
            setLogScale(axis, false);
        }

        void unsetLogScale(char axis) {
            unsetLogScale(std::string(1, axis));
        }

        void setNonlinear(const std::string& axis,
                  const std::string& forward,
                  const std::string& inverse)
        {
            ensureValid();
            std::ostringstream stream;
            stream << "set nonlinear " << axis << " via " << forward << " inverse " << inverse;
            writeLine(stream.str());
            flush();
        }

        void setNonlinear(char axis, const std::string& forward, const std::string& inverse)
        {
            setNonlinear(std::string(1, axis), forward, inverse);
        }

        void unsetNonlinear(const std::string& axis)
        {
            command("unset nonlinear " + axis);
        }

        void unsetNonlinear(char axis)
        {
            unsetNonlinear(std::string(1, axis));
        }

        void setGrid(bool enable = true) {
            command(std::string(enable ? "set" : "unset") + " grid");
        }

        void setKey(const std::string& options) {
            command("set key " + options);
        }

        void unsetKey() {
            command("unset key");
        }

        void setAutoscale(bool enable = true) {
            command(std::string(enable ? "set" : "unset") + " autoscale");
        }

        void setSamples(int count) {
            if (count <= 0) {
                throw std::invalid_argument("Sample count must be positive");
            }
            command("set samples " + std::to_string(count));
        }

        void beginMultiplot(const std::string& options = {}) {
            if (options.empty()) {
                command("set multiplot");
            } else {
                command("set multiplot " + options);
            }
        }

        void endMultiplot() {
            command("unset multiplot");
        }

        void setPalette(const std::string& options) {
            command("set palette " + options);
        }

        void unsetPalette() {
            command("unset palette");
        }

        void plot(const DataSet2D& dataSet) {
            plot(std::vector<DataSet2D>{dataSet});
        }

        void plot(const std::vector<DataSet2D>& dataSets) {
            plotInternal(dataSets, "plot");
        }

        void plot(std::initializer_list<DataSet2D> dataSets) {
            plot(std::vector<DataSet2D>(dataSets));
        }
        void plotRaw(const std::string& header,
                      const std::function<void(std::FILE*)>& writer) {
            ensureValid();
            if (!writer) {
                throw std::invalid_argument("plotRaw requires a valid writer");
            }
            writeLine(header);
            writer(pipe_);
            writeLine("e");
            flush();
        }

        void plotRaw(const std::string& header,
                      const std::vector<std::function<void(std::FILE*)>>& writers) {
            ensureValid();
            if (writers.empty()) {
                throw std::invalid_argument("plotRaw requires at least one writer");
            }
            writeLine(header);
            for (const auto& writer : writers) {
                if (!writer) {
                    throw std::invalid_argument("plotRaw requires a valid writer");
                }
                writer(pipe_);
                writeLine("e");
            }
            flush();
        }


        void plot(const std::vector<double>& y,
                  const std::string& title = {},
                  PlotStyle style = PlotStyle{}) {
            std::vector<double> x(y.size());
            for (std::size_t i = 0; i < y.size(); ++i) {
                x[i] = static_cast<double>(i);
            }
            plot(x, y, title, style);
        }

        void plot(const std::vector<double>& x,
                  const std::vector<double>& y,
                  const std::string& title = {},
                  PlotStyle style = PlotStyle{}) {
            if (x.size() != y.size()) {
                throw std::invalid_argument("X and Y data vectors must have the same size");
            }
            DataSet2D dataSet{
                std::vector<double>(x.begin(), x.end()),
                std::vector<double>(y.begin(), y.end()),
                title,
                style
            };
            plot(dataSet);
        }

        void splot(const DataSet3D& dataSet) {
            splot(std::vector<DataSet3D>{dataSet});
        }

        void splot(const std::vector<DataSet3D>& dataSets) {
            plotInternal(dataSets, "splot");
        }

        void splot(std::initializer_list<DataSet3D> dataSets) {
            splot(std::vector<DataSet3D>(dataSets));
        }

        void splot(const std::vector<double>& x,
                   const std::vector<double>& y,
                   const std::vector<double>& z,
                   const std::string& title = {},
                   PlotStyle style = PlotStyle{}) {
            if (x.size() != y.size() || x.size() != z.size()) {
                throw std::invalid_argument("X, Y, and Z data vectors must have the same size");
            }
            DataSet3D dataSet{
                std::vector<double>(x.begin(), x.end()),
                std::vector<double>(y.begin(), y.end()),
                std::vector<double>(z.begin(), z.end()),
                title,
                style
            };
            splot(dataSet);
        }

        void plotEquation(const std::string& expression,
                          const std::string& title = {},
                          PlotStyle style = PlotStyle{}) {
            ensureValid();
            std::ostringstream stream;
            stream << "plot " << expression << ' ';
            appendTitleAndStyle(stream, title, style);
            writeLine(stream.str());
            flush();
        }

        void replot() {
            command("replot");
        }

        void clear() {
            command("clear");
        }

    private:
        std::string command_;
        std::FILE* pipe_;

        void close() {
            if (pipe_ != nullptr) {
                pclose(pipe_);
                pipe_ = nullptr;
            }
        }

        void ensureValid() {
            if (!valid()) {
                throw std::runtime_error("gnuplot process is not available");
            }
        }

        void writeLine(const std::string& line) {
            if (std::fputs((line + '\n').c_str(), pipe_) < 0) {
                throw std::runtime_error("Failed to write to gnuplot");
            }
        }

        void flush() {
            if (std::fflush(pipe_) != 0) {
                throw std::runtime_error("Failed to flush gnuplot pipe");
            }
        }


        static std::string EnsurePersist(std::string command) {
            if (command.find("-persist") == std::string::npos) {
                if (!command.empty() && command.back() != ' ') {
                    command += ' ';
                }
                command += "-persist";
            }
            return command;
        }

        static std::string EscapeSingleQuotes(const std::string& input) {
            std::string escaped;
            escaped.reserve(input.size());
            for (char ch : input) {
                if (ch == static_cast<char>(39)) {
                    escaped += "\\'";
                } else {
                    escaped += ch;
                }
            }
            return escaped;
        }

        static std::string ModeToString(PlotMode mode) {
            switch (mode) {
                case PlotMode::Lines:
                    return "lines";
                case PlotMode::Points:
                    return "points";
                case PlotMode::LinesPoints:
                    return "linespoints";
                case PlotMode::Impulses:
                    return "impulses";
                case PlotMode::Dots:
                    return "dots";
                case PlotMode::Boxes:
                    return "boxes";
                case PlotMode::FilledCurves:
                    return "filledcurves";
            }
            return "lines";
        }

        static void appendStyle(std::ostringstream& stream, const PlotStyle& style) {
            stream << "with " << ModeToString(style.mode);
            if (style.lineType) {
                stream << " lt " << *style.lineType;
            }
            if (style.lineWidth) {
                stream << " lw " << *style.lineWidth;
            }
            if (style.lineColor && !style.lineColor->empty()) {
                stream << " lc rgb '" << EscapeSingleQuotes(*style.lineColor) << "'";
            }
            if (style.pointType) {
                stream << " pt " << *style.pointType;
            }
            if (style.pointSize) {
                stream << " ps " << *style.pointSize;
            }
            if (style.extra && !style.extra->empty()) {
                stream << ' ' << *style.extra;
            }
        }

        static void appendTitleAndStyle(std::ostringstream& stream,
                                        const std::string& title,
                                        const PlotStyle& style) {
            if (title.empty()) {
                stream << "notitle ";
            } else {
                stream << "title '" << EscapeSingleQuotes(title) << "' ";
            }
            appendStyle(stream, style);
        }

        template <typename DataSet>
        void plotInternal(const std::vector<DataSet>& dataSets, const char* keyword) {
            ensureValid();
            if (dataSets.empty()) {
                throw std::invalid_argument("No datasets provided");
            }

            std::ostringstream header;
            header << keyword << ' ';
            for (std::size_t i = 0; i < dataSets.size(); ++i) {
                if (i > 0) {
                    header << ", ";
                }
                header << "'-' ";
                appendTitleAndStyle(header, dataSets[i].title, dataSets[i].style);
            }
            writeLine(header.str());

            for (const auto& dataSet : dataSets) {
                writeDataSet(dataSet);
                writeLine("e");
            }
            flush();
        }

        void writeDataSet(const DataSet2D& dataSet) {
            const std::size_t size = std::min(dataSet.x.size(), dataSet.y.size());
            for (std::size_t i = 0; i < size; ++i) {
                if (std::fprintf(pipe_, "%.*g %.*g\n", 15, dataSet.x[i], 15, dataSet.y[i]) < 0) {
                    throw std::runtime_error("Failed to write 2D data to gnuplot");
                }
            }
        }

        void writeDataSet(const DataSet3D& dataSet) {
            const std::size_t size = std::min({dataSet.x.size(), dataSet.y.size(), dataSet.z.size()});
            for (std::size_t i = 0; i < size; ++i) {
                if (std::fprintf(pipe_, "%.*g %.*g %.*g\n",
                                  15, dataSet.x[i],
                                  15, dataSet.y[i],
                                  15, dataSet.z[i]) < 0) {
                    throw std::runtime_error("Failed to write 3D data to gnuplot");
                }
            }
        }
    };
}

#endif // THOT_GNUPLOT_HPP