cmake_minimum_required(VERSION 3.18)
project(Thot)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)

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
        src/utils/stresstest.hpp
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
        src/layer/details/statespace.hpp
        src/layer/details/batchnorm.hpp
        src/layer/details/residual.hpp
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
        src/activation/details/raw.hpp
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
add_executable(test_Thot test/main.cpp
        test/semi_lazy.cpp
        src/block/details/blocks/residual.hpp
        src/block/details/blocks/sequential.hpp
        src/lrscheduler/details/common.hpp
        src/lrscheduler/registry.hpp
        src/common/local.hpp
)
target_link_libraries(test_Thot PRIVATE Thot)
