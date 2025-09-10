#pragma once

#include "../regularizer.hpp"
#include <cublas_v2.h>
#include <memory>

namespace Thot {
namespace Regularizers {

class L1Regularizer : public Regularizer {
    float lambda_;
public:
    explicit L1Regularizer(float lambda = 1e-4f) : lambda_(lambda) {}
    float compute(const Utils::Tensor& params) override {
        cublasHandle_t handle; cublasCreate(&handle);
        float result = 0.0f;
        cublasSasum(handle, static_cast<int>(params.size()), params.data(), 1, &result);
        cublasDestroy(handle);
        return lambda_ * result;
    }
};

class L2Regularizer : public Regularizer {
    float lambda_;
public:
    explicit L2Regularizer(float lambda = 1e-4f) : lambda_(lambda) {}
    float compute(const Utils::Tensor& params) override {
        cublasHandle_t handle; cublasCreate(&handle);
        float nrm = 0.0f;
        cublasSnrm2(handle, static_cast<int>(params.size()), params.data(), 1, &nrm);
        cublasDestroy(handle);
        return lambda_ * nrm * nrm;
    }
};

class ElasticNetRegularizer : public Regularizer {
    float l1_;
    float l2_;
public:
    ElasticNetRegularizer(float l1 = 1e-4f, float l2 = 1e-4f)
        : l1_(l1), l2_(l2) {}
    float compute(const Utils::Tensor& params) override {
        cublasHandle_t handle; cublasCreate(&handle);
        float l1v = 0.0f; float l2n = 0.0f;
        cublasSasum(handle, static_cast<int>(params.size()), params.data(), 1, &l1v);
        cublasSnrm2(handle, static_cast<int>(params.size()), params.data(), 1, &l2n);
        cublasDestroy(handle);
        return l1_ * l1v + l2_ * l2n * l2n;
    }
};

inline bool register_basic_regularizers() {
    register_regularizer("l1", [](){ return std::make_shared<L1Regularizer>(); });
    register_regularizer("l2", [](){ return std::make_shared<L2Regularizer>(); });
    register_regularizer("elastic_net", [](){ return std::make_shared<ElasticNetRegularizer>(); });
    return true;
}

static inline bool basic_regularizers_registered = register_basic_regularizers();

} // namespace Regularizers
} // namespace Thot