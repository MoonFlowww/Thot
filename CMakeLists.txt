cmake_minimum_required(VERSION 3.18)
project(Thot)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

set(_libtorch_root "/opt/libtorch")
if (EXISTS "${_libtorch_root}/share/cmake/Torch/TorchConfig.cmake")
    list(PREPEND CMAKE_PREFIX_PATH "${_libtorch_root}")
    set(Torch_DIR "${_libtorch_root}/share/cmake/Torch")
endif()

find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

add_library(Thot SHARED
        src/Thot.cpp
        src/core.hpp
        src/network.hpp
        src/layer/layer.hpp
        src/loss/loss.hpp
        src/lrscheduler/lrscheduler.hpp
        src/metric/metric.hpp
        src/optimizer/optimizer.hpp
        src/regularization/regularization.hpp
        src/common/stresstest.hpp
        src/utils/terminal.hpp
        src/utils/gnuplot.hpp
        src/initialization/initialization.hpp
        src/evaluation/evaluation.hpp
        src/evaluation/details/classification.hpp
        src/evaluation/details/timeserie.hpp
        src/data/data.hpp
        src/data/details/generation.hpp
        src/data/details/load.hpp
        src/data/details/manipulation.hpp
        src/data/details/dimreduction.hpp
        src/calibration/calibration.hpp
        src/block/block.hpp
        src/block/details/transformers/classic.hpp
        src/block/details/transformers/mamba.hpp
        src/block/details/transformers/plusplus.hpp
        src/block/details/transformers/titan.hpp
        src/block/details/transformers/atlas.hpp
        src/block/details/transformers/ebt.hpp
        src/attention/attention.hpp
        src/attention/details/kernel.hpp
        src/attention/details/head.hpp
        src/activation/activation.hpp
        src/layer/details/fc.hpp
        src/layer/details/conv.hpp
        src/layer/details/batchnorm.hpp
        src/layer/details/recurrent.hpp
        src/layer/details/flatten.hpp
        src/layer/details/pooling.hpp
        src/layer/details/positional_encoding.hpp
        src/layer/details/dropout.hpp
        src/lrscheduler/details/cosineannealing.hpp
        src/lrscheduler/details/exponential.hpp
        src/optimizer/details/muon.hpp
        src/optimizer/details/adam.hpp
        src/optimizer/details/sophia.hpp
        src/regularization/details/ewc.hpp
        src/regularization/details/si.hpp
        src/regularization/details/mas.hpp
        src/regularization/details/l2.hpp
        src/utils/progressbar.hpp
        src/training/kfold.hpp
        src/activation/details/relu.hpp
        src/loss/details/mse.hpp
        src/loss/details/ce.hpp
        src/loss/details/cce.hpp
        src/optimizer/details/sgd.hpp
        src/activation/apply.hpp
        src/activation/details/dsilu.hpp
        src/activation/details/gelu.hpp
        src/activation/details/glu.hpp
        src/activation/details/leaky_relu.hpp
        src/activation/details/mish.hpp
        src/activation/details/psilu.hpp
        src/activation/details/sigmoid.hpp
        src/activation/details/silu.hpp
        src/activation/details/softmax.hpp
        src/activation/details/swiglu.hpp
        src/activation/details/swish.hpp
        src/activation/details/tanh.hpp
        src/initialization/apply.hpp
        src/loss/details/reduction.hpp
        src/layer/registry.hpp
        src/optimizer/registry.hpp)



target_link_libraries(Thot "${TORCH_LIBRARIES}" Boost::filesystem)
target_include_directories(Thot PUBLIC ${CMAKE_SOURCE_DIR}/include ${TORCH_INCLUDE_DIRS})

#Test---------------
add_executable(test_Thot test/etth/etth.cpp
        src/block/details/blocks/residual.hpp
        src/block/details/blocks/sequential.hpp
        src/lrscheduler/details/common.hpp
        src/lrscheduler/registry.hpp
        src/common/local.hpp
        src/regularization/apply.hpp
        src/regularization/details/nuclearnorm.hpp
        src/regularization/details/fge.hpp
        src/regularization/details/sfge.hpp
        src/regularization/details/swa.hpp
        src/regularization/details/swag.hpp
        src/calibration/details/temperature_scaling.hpp
        src/common/save_load.hpp
        src/layer/details/s4.hpp
        src/plot/details/reliability.hpp
        src/plot/details/training.hpp
        src/plot/plot.hpp
        src/plot/details/statistics.hpp
        src/plot/details/reliability/reliability_curve_utils.hpp
        src/plot/details/reliability/reliability_det.hpp
        src/plot/details/reliability/reliability_pr.hpp
        src/plot/details/reliability/reliability_youdens.hpp
        src/plot/details/reliability/reliability_roc.hpp
        src/regularization/details/centeringvariance.hpp
        src/regularization/details/decov.hpp
        src/regularization/details/elasticnet.hpp
        src/regularization/details/grouplasso.hpp
        src/regularization/details/jacobian.hpp
        src/regularization/details/klsparsity.hpp
        src/regularization/details/l0hardconcrete.hpp
        src/regularization/details/l1.hpp
        src/regularization/details/maxnorm.hpp
        src/regularization/details/orthogonality.hpp
        src/regularization/details/r1.hpp
        src/regularization/details/r2.hpp
        src/regularization/details/spectralnorm.hpp
        src/regularization/details/structuredl2.hpp
        src/regularization/details/trades.hpp
        src/regularization/details/vat.hpp
        src/regularization/details/wgangp.hpp
        src/regularization/details/common.hpp
        src/common/graph.hpp
        src/common/streaming.hpp
        src/loss/details/bce.hpp
        src/loss/details/cosine_embedding.hpp
        src/loss/details/kl.hpp
        src/loss/details/mae.hpp
        src/loss/details/margin_ranking.hpp
        src/loss/details/nll.hpp
        src/loss/details/smooth_l1.hpp
        src/loss/details/helper.hpp
        src/optimizer/details/adafactor.hpp
        src/optimizer/details/adagrad.hpp
        src/optimizer/details/lamb.hpp
        src/optimizer/details/lion.hpp
        src/optimizer/details/rmsprop.hpp
        src/block/details/transformers/longformer_xl.hpp
        src/block/details/transformers/perceiver.hpp
        src/block/details/transformers/bert.hpp
        src/block/details/transformers/vision.hpp
        src/layer/details/reduce.hpp
        test/etth/wavelet.hpp
        src/layer/details/instancenorm.hpp
        src/data/details/repair.hpp
        src/plot/details/data.hpp
        src/data/details/manipulation/normalization/zscore.hpp
        src/data/details/manipulation/normalization/common.hpp
        src/data/details/manipulation/normalization/power.hpp
        src/data/details/manipulation/normalization/ehlers.hpp
)
target_link_libraries(test_Thot PRIVATE Thot)
