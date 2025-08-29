#include <cassert>
#include <cmath>
#include <iostream>
#include "losses/losses.hpp"
#include "tensor.hpp"

int main() {
    using namespace Thot;
    using namespace cuda::losses;

    // MSE
    Utils::Tensor pred_mse({2}); pred_mse.upload({1.0f, 2.0f});
    Utils::Tensor tgt_mse({2});  tgt_mse.upload({0.0f, 2.0f});
    Utils::Tensor loss_mse({2});
    launchMSE(static_cast<float*>(pred_mse.data()), static_cast<float*>(tgt_mse.data()), static_cast<float*>(loss_mse.data()), 2);
    auto h_mse = loss_mse.download();
    assert(std::fabs(h_mse[0] - 0.5f) < 1e-5);
    assert(std::fabs(h_mse[1] - 0.0f) < 1e-5);

    // MAE
    Utils::Tensor loss_mae({2});
    launchMAE(static_cast<float*>(pred_mse.data()), static_cast<float*>(tgt_mse.data()), static_cast<float*>(loss_mae.data()), 2);
    auto h_mae = loss_mae.download();
    assert(std::fabs(h_mae[0] - 1.0f) < 1e-5);
    assert(std::fabs(h_mae[1] - 0.0f) < 1e-5);

    // Binary Cross-Entropy
    Utils::Tensor pred_bce({1}); pred_bce.upload({0.9f});
    Utils::Tensor tgt_bce({1});  tgt_bce.upload({1.0f});
    Utils::Tensor loss_bce({1});
    launchBinaryCrossEntropy(static_cast<float*>(pred_bce.data()), static_cast<float*>(tgt_bce.data()), static_cast<float*>(loss_bce.data()), 1, 1e-8f);
    auto h_bce = loss_bce.download();
    assert(std::fabs(h_bce[0] - (-std::log(0.9f))) < 1e-5);

    // Categorical Cross-Entropy
    Utils::Tensor pred_cce({1,3}); pred_cce.upload({0.7f, 0.2f, 0.1f});
    Utils::Tensor tgt_cce({1,3});  tgt_cce.upload({1.0f, 0.0f, 0.0f});
    Utils::Tensor loss_cce({1});
    launchCategoricalCrossEntropy(static_cast<float*>(pred_cce.data()), static_cast<float*>(tgt_cce.data()), static_cast<float*>(loss_cce.data()), 1, 3, 1e-8f);
    auto h_cce = loss_cce.download();
    assert(std::fabs(h_cce[0] - (-std::log(0.7f))) < 1e-5);

    // Sparse Categorical Cross-Entropy
    Utils::Tensor tgt_scce({1}); tgt_scce.upload({0.0f});
    Utils::Tensor loss_scce({1});
    launchSparseCategoricalCrossEntropy(static_cast<float*>(pred_cce.data()), static_cast<float*>(tgt_scce.data()), static_cast<float*>(loss_scce.data()), 1, 3, 1e-8f);
    auto h_scce = loss_scce.download();
    assert(std::fabs(h_scce[0] - (-std::log(0.7f))) < 1e-5);

    // Hinge
    Utils::Tensor pred_hinge({1}); pred_hinge.upload({0.5f});
    Utils::Tensor tgt_hinge({1});  tgt_hinge.upload({1.0f});
    Utils::Tensor loss_hinge({1});
    launchHinge(static_cast<float*>(pred_hinge.data()), static_cast<float*>(tgt_hinge.data()), static_cast<float*>(loss_hinge.data()), 1);
    auto h_hinge = loss_hinge.download();
    assert(std::fabs(h_hinge[0] - 0.5f) < 1e-5);

    // Huber
    Utils::Tensor pred_huber({1}); pred_huber.upload({2.0f});
    Utils::Tensor tgt_huber({1});  tgt_huber.upload({0.0f});
    Utils::Tensor loss_huber({1});
    launchHuber(static_cast<float*>(pred_huber.data()), static_cast<float*>(tgt_huber.data()), static_cast<float*>(loss_huber.data()), 1, 1.0f);
    auto h_huber = loss_huber.download();
    assert(std::fabs(h_huber[0] - 1.5f) < 1e-5);

    // KL Divergence
    Utils::Tensor pred_kl({1}); pred_kl.upload({0.5f});
    Utils::Tensor tgt_kl({1});  tgt_kl.upload({0.25f});
    Utils::Tensor loss_kl({1});
    launchKLDivergence(static_cast<float*>(pred_kl.data()), static_cast<float*>(tgt_kl.data()), static_cast<float*>(loss_kl.data()), 1, 1e-8f);
    auto h_kl = loss_kl.download();
    assert(std::fabs(h_kl[0] - (0.25f*std::log(0.25f/0.5f))) < 1e-5);

    std::cout << "Loss tests passed." << std::endl;
    return 0;
}
