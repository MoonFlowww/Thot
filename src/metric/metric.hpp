#ifndef THOT_METRIC_HPP
#define THOT_METRIC_HPP
// This file is an factory, must exempt it from any logical-code. For functions look into "/details"
namespace Thot::Metric::Classification {
    enum class Kind {
        Accuracy,
        Precision,
        Recall,
        F1,
        TruePositiveRate,
        TrueNegativeRate,
        Top1Accuracy,
        ExpectedCalibrationError,
        MaximumCalibrationError,
        CohensKappa,
        LogLoss,
        BrierScore,
    };

    struct Descriptor {
        Kind kind;
    };

    [[nodiscard]] constexpr auto Make(Kind kind) noexcept -> Descriptor { return Descriptor{kind}; }

    inline constexpr Descriptor Accuracy{Kind::Accuracy};
    inline constexpr Descriptor Precision{Kind::Precision};
    inline constexpr Descriptor Recall{Kind::Recall};
    inline constexpr Descriptor F1{Kind::F1};
    inline constexpr Descriptor TruePositiveRate{Kind::TruePositiveRate};
    inline constexpr Descriptor TrueNegativeRate{Kind::TrueNegativeRate};
    inline constexpr Descriptor Top1Accuracy{Kind::Top1Accuracy};
    inline constexpr Descriptor ExpectedCalibrationError{Kind::ExpectedCalibrationError};
    inline constexpr Descriptor MaximumCalibrationError{Kind::MaximumCalibrationError};
    inline constexpr Descriptor CohensKappa{Kind::CohensKappa};
    inline constexpr Descriptor LogLoss{Kind::LogLoss};
    inline constexpr Descriptor BrierScore{Kind::BrierScore};
}

#endif //THOT_METRIC_HPP