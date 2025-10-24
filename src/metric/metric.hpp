#ifndef THOT_METRIC_HPP
#define THOT_METRIC_HPP
// This file is an factory, must exempt it from any logical-code. For functions look into "/details"
namespace Thot::Metric::Classification {
    enum class Kind {
        Accuracy,
        AUCROC,
        BalancedAccuracy,
        BalancedErrorRate,
        F1,
        FBeta0Point5,
        FBeta2,
        FalseDiscoveryRate,
        FalseNegativeRate,
        FalseOmissionRate,
        FalsePositiveRate,
        FowlkesMallows,
        HammingLoss,
        Informedness,
        JaccardIndexMicro,
        JaccardIndexMacro,
        Markness,
        Matthews,
        NegativeLikelihoodRatio,
        NegativePredictiveValue,
        PositiveLikelihoodRatio,
        PositivePredictiveValue,
        Precision,
        Prevalence,
        Recall,
        Top1Error,
        Top3Error,
        Top5Error,
        Top1Accuracy,
        Top3Accuracy,
        Top5Accuracy,
        Specificity,
        ThreatScore,
        TrueNegativeRate,
        TruePositiveRate,
        YoudenIndex,
        LogLoss,
        BrierScore,
        BrierSkillScore,
        ExpectedCalibrationError,
        MaximumCalibrationError,
        CalibrationSlope,
        CalibrationIntercept,
        HosmerLemeshowPValue,
        KolmogorovSmirnovStatistic,
        CohensKappa,
        ConfusionEntropy,
        CoverageError,
        LabelRankingAveragePrecision,
        SubsetAccuracy,
        AUPRC,
        AUPRG,
        GiniCoefficient,
    };

    struct Descriptor {
        Kind kind;
    };

    [[nodiscard]] constexpr auto Make(Kind kind) noexcept -> Descriptor { return Descriptor{kind}; }

    inline constexpr Descriptor Accuracy{Kind::Accuracy};
    inline constexpr Descriptor AUCROC{Kind::AUCROC};
    inline constexpr Descriptor BalancedAccuracy{Kind::BalancedAccuracy};
    inline constexpr Descriptor BalancedErrorRate{Kind::BalancedErrorRate};
    inline constexpr Descriptor F1{Kind::F1};
    inline constexpr Descriptor FBeta0Point5{Kind::FBeta0Point5};
    inline constexpr Descriptor FBeta2{Kind::FBeta2};
    inline constexpr Descriptor FalseDiscoveryRate{Kind::FalseDiscoveryRate};
    inline constexpr Descriptor FalseNegativeRate{Kind::FalseNegativeRate};
    inline constexpr Descriptor FalseOmissionRate{Kind::FalseOmissionRate};
    inline constexpr Descriptor FalsePositiveRate{Kind::FalsePositiveRate};
    inline constexpr Descriptor FowlkesMallows{Kind::FowlkesMallows};
    inline constexpr Descriptor HammingLoss{Kind::HammingLoss};
    inline constexpr Descriptor Informedness{Kind::Informedness};
    inline constexpr Descriptor JaccardIndexMicro{Kind::JaccardIndexMicro};
    inline constexpr Descriptor JaccardIndexMacro{Kind::JaccardIndexMacro};
    inline constexpr Descriptor Markness{Kind::Markness};
    inline constexpr Descriptor Matthews{Kind::Matthews};
    inline constexpr Descriptor NegativeLikelihoodRatio{Kind::NegativeLikelihoodRatio};
    inline constexpr Descriptor NegativePredictiveValue{Kind::NegativePredictiveValue};
    inline constexpr Descriptor PositiveLikelihoodRatio{Kind::PositiveLikelihoodRatio};
    inline constexpr Descriptor PositivePredictiveValue{Kind::PositivePredictiveValue};
    inline constexpr Descriptor Precision{Kind::Precision};
    inline constexpr Descriptor Prevalence{Kind::Prevalence};
    inline constexpr Descriptor Recall{Kind::Recall};
    inline constexpr Descriptor Top1Error{Kind::Top1Error};
    inline constexpr Descriptor Top3Error{Kind::Top3Error};
    inline constexpr Descriptor Top5Error{Kind::Top5Error};
    inline constexpr Descriptor Top1Accuracy{Kind::Top1Accuracy};
    inline constexpr Descriptor Top3Accuracy{Kind::Top3Accuracy};
    inline constexpr Descriptor Top5Accuracy{Kind::Top5Accuracy};
    inline constexpr Descriptor Specificity{Kind::Specificity};
    inline constexpr Descriptor ThreatScore{Kind::ThreatScore};
    inline constexpr Descriptor TrueNegativeRate{Kind::TrueNegativeRate};
    inline constexpr Descriptor TruePositiveRate{Kind::TruePositiveRate};
    inline constexpr Descriptor YoudenIndex{Kind::YoudenIndex};
    inline constexpr Descriptor LogLoss{Kind::LogLoss};
    inline constexpr Descriptor BrierScore{Kind::BrierScore};
    inline constexpr Descriptor BrierSkillScore{Kind::BrierSkillScore};
    inline constexpr Descriptor ExpectedCalibrationError{Kind::ExpectedCalibrationError};
    inline constexpr Descriptor MaximumCalibrationError{Kind::MaximumCalibrationError};
    inline constexpr Descriptor CalibrationSlope{Kind::CalibrationSlope};
    inline constexpr Descriptor CalibrationIntercept{Kind::CalibrationIntercept};
    inline constexpr Descriptor HosmerLemeshowPValue{Kind::HosmerLemeshowPValue};
    inline constexpr Descriptor KolmogorovSmirnovStatistic{Kind::KolmogorovSmirnovStatistic};
    inline constexpr Descriptor CohensKappa{Kind::CohensKappa};
inline constexpr Descriptor ConfusionEntropy{Kind::ConfusionEntropy};
    inline constexpr Descriptor CoverageError{Kind::CoverageError};
    inline constexpr Descriptor LabelRankingAveragePrecision{Kind::LabelRankingAveragePrecision};
    inline constexpr Descriptor SubsetAccuracy{Kind::SubsetAccuracy};
    inline constexpr Descriptor AUPRC{Kind::AUPRC};
    inline constexpr Descriptor AUPRG{Kind::AUPRG};
    inline constexpr Descriptor GiniCoefficient{Kind::GiniCoefficient};
}

namespace Thot::Metric::Timeseries {
    enum class Kind {
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        MeanBiasError,
        MeanSquaredError,
        MedianAbsoluteError,
        R2Score,
        RootMeanSquaredError,
        SymmetricMeanAbsolutePercentageError,
        WeightedAbsolutePercentageError,
        MeanPercentageError,
        ExplainedVariance,
        TheilsU1,
        TheilsU2,
        MeanAbsoluteScaledError,
        RootMeanSquaredScaledError,
        MedianRelativeAbsoluteError,
        GeometricMeanRelativeAbsoluteError,
        OverallWeightedAverage,
        DynamicTimeWarpingDistance,
        TimeWarpEditDistance,
        SpectralDistance,
        CosineSimilarity,
        NegativeLogLikelihood,
        ContinuousRankedProbabilityScore,
        EnergyScore,
        PinballLossAverage,
        BrierScore,
        PredictionIntervalCoverageProbability,
        MeanPredictionIntervalWidth,
        WinklerScore,
        ConditionalCoverageError,
        QuantileCrossingRate,
        AutocorrelationOfResiduals,
        PartialAutocorrelationOfResiduals,
        LjungBoxStatistic,
        BoxPierceStatistic,
        DurbinWatsonStatistic,
        JarqueBeraStatistic,
        AndersonDarlingStatistic,
        BreuschPaganStatistic,
        WhiteStatistic,
        PopulationStabilityIndex,
        KullbackLeiblerDivergence,
        JensenShannonDivergence,
        WassersteinDistance,
        MaximumMeanDiscrepancy,
        LossDriftSlope,
        LossCusumStatistic,
        ResidualChangePointScore,
        QLIKE,
        LogVarianceMeanSquaredError,
        SqrtVarianceMeanSquaredError,
        msIC,
        msIR,
    };

    struct Descriptor {
        Kind kind;
    };

    [[nodiscard]] constexpr auto Make(Kind kind) noexcept -> Descriptor { return Descriptor{kind}; }
}


#endif //THOT_METRIC_HPP