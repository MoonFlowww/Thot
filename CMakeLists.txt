cmake_minimum_required(VERSION 3.22)
project(Thot LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT UNIX OR APPLE)
    message(FATAL_ERROR "Linux-only build.")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS 11.8)
    message(FATAL_ERROR "Need CUDA 11.8+ for Ada (SM 89). Found ${CUDAToolkit_VERSION}.")
endif()

# target all codegen to Ada (RTX 4070 Ti)
set(CMAKE_CUDA_ARCHITECTURES 89)

enable_testing()

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/Thot/cuda/cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/Thot/headers
        ${CUDAToolkit_INCLUDE_DIRS}
)

set(KERNEL_SOURCES
        Thot/cuda/cu/activations/activations.cu
        Thot/cuda/cu/layers/conv2d.cu
        Thot/cuda/cu/layers/fc.cu
        Thot/cuda/cu/layers/rbm.cu
        Thot/cuda/cu/layers/rnn.cu
        Thot/cuda/cu/losses/loss.cu
        Thot/cuda/cu/LowRankCuda/lowrank.cu
        Thot/cuda/cu/optimizations/adam.cu
        Thot/cuda/cu/optimizations/sgd.cu
        Thot/cuda/cu/optimizations/sgdm.cu
)

set(HEADERS
        Thot/headers/activations/activations.hpp
        Thot/headers/initializations/initializations.hpp
        Thot/headers/layers/details/conv2d.hpp
        Thot/headers/layers/details/fc.hpp
        Thot/headers/layers/details/rbm.hpp
        Thot/headers/layers/details/rnn.hpp
        Thot/headers/layers/layers.hpp
        Thot/headers/losses/losses.hpp
        Thot/headers/metrics/metrics.hpp
        Thot/headers/network.hpp
        Thot/headers/optimizations/details/adam.hpp
        Thot/headers/optimizations/details/sgd.hpp
        Thot/headers/optimizations/details/sgdm.hpp
        Thot/headers/optimizations/optimizations.hpp
        Thot/headers/tensor.hpp
        Thot/Thot.hpp
        Thot/headers/utils/progressbar.h
        Thot/headers/utils/translators.h
        Thot/headers/LearningProcess/batch.hpp
        Thot/headers/LearningProcess/kfold.hpp
)

add_library(thot_kernels STATIC ${KERNEL_SOURCES})
set_target_properties(thot_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

add_executable(${PROJECT_NAME} main.cpp ${HEADERS})
target_link_libraries(${PROJECT_NAME} PRIVATE
        -Wl,--whole-archive thot_kernels -Wl,--no-whole-archive
        CUDA::cudart
        pthread dl
)
target_compile_definitions(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:_DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
)
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

######################TESTs######################
#RNN tests
add_executable(rnn_test tests/rnn_test.cpp)
target_link_libraries(rnn_test PRIVATE
        -Wl,--whole-archive thot_kernels -Wl,--no-whole-archive
        CUDA::cudart
        pthread dl
)
set_target_properties(rnn_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME rnn_test COMMAND rnn_test)
