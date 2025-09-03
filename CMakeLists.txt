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

# -------- cuDNN --------
option(THOT_USE_CUDNN "Enable cuDNN acceleration" ON)

if(THOT_USE_CUDNN)
    # Allow users to point at a non-default install with CUDNN_ROOT
    # Typical layouts:
    #   $CUDNN_ROOT/include/cudnn.h
    #   $CUDNN_ROOT/lib64/libcudnn.so
    find_path(CUDNN_INCLUDE_DIR
            NAMES cudnn.h
            HINTS ENV CUDNN_ROOT ${CUDAToolkit_INCLUDE_DIRS}
            PATH_SUFFIXES include)
    find_library(CUDNN_LIBRARY
            NAMES cudnn
            HINTS ENV CUDNN_ROOT ${CUDAToolkit_LIBRARY_DIR}
            PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

    if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
        message(FATAL_ERROR "cuDNN not found. Set CUDNN_ROOT or CUDNN_INCLUDE_DIR/CUDNN_LIBRARY.")
    endif()

    add_library(cudnn::cudnn UNKNOWN IMPORTED)
    set_target_properties(cudnn::cudnn PROPERTIES
            IMPORTED_LOCATION "${CUDNN_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}")

    # Improve runtime linking in-build (no install step)
    get_filename_component(_CUDNN_LIBDIR "${CUDNN_LIBRARY}" DIRECTORY)
    list(APPEND CMAKE_BUILD_RPATH "${_CUDNN_LIBDIR}")
    set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" CACHE STRING "" FORCE)

    message(STATUS "cuDNN: lib='${CUDNN_LIBRARY}' include='${CUDNN_INCLUDE_DIR}'")
endif()
# -----------------------

# Ada (RTX 4070 Ti)
set(CMAKE_CUDA_ARCHITECTURES 89)

enable_testing()

add_compile_definitions(THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/Thot/cuda/cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/Thot/headers
        ${CUDAToolkit_INCLUDE_DIRS}
        $<$<BOOL:${THOT_USE_CUDNN}>:${CUDNN_INCLUDE_DIR}>
)

set(KERNEL_SOURCES
        Thot/cuda/cu/activations/activations.cu
        Thot/cuda/cu/layers/conv2d.cu
        Thot/cuda/cu/layers/flatten.cu
        Thot/cuda/cu/layers/maxpool2d.cu
        Thot/cuda/cu/layers/fc.cu
        Thot/cuda/cu/layers/rbm.cu
        Thot/cuda/cu/layers/rnn.cu
        Thot/cuda/cu/losses/loss.cu
        Thot/cuda/cu/LowRankCuda/lowrank.cu
        Thot/cuda/cu/optimizations/adam.cu
        Thot/cuda/cu/optimizations/sgd.cu
        Thot/cuda/cu/optimizations/sgdm.cu
        Thot/cuda/cu/layers/flatten.cu
        Thot/cuda/cu/layers/maxpool2d.cu

)        #Thot/cuda/cu/layers/conv2d_old.cu

set(HEADERS
        Thot/headers/activations/activations.hpp
        Thot/headers/initializations/initializations.hpp
        Thot/headers/layers/details/conv2d.hpp
        Thot/headers/layers/details/fc.hpp
        Thot/headers/layers/details/flatten.hpp
        Thot/headers/layers/details/maxpool2d.hpp
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
        Thot/headers/utils/translators.h
        Thot/headers/LearningProcess/batch.hpp
        Thot/headers/LearningProcess/kfold.hpp
        Thot/cuda/cuh/layers/flatten.cuh
        Thot/cuda/cuh/layers/maxpool2d.cuh
        Thot/headers/layers/details/flatten.hpp
        Thot/headers/layers/details/maxpool2d.hpp
        Thot/headers/data/details/cifar.hpp
        Thot/cuda/cu/attentions/mha.cu
        Thot/cuda/cuh/attentions/mha.cuh
        Thot/headers/attentions/details/mha.hpp
)

add_library(thot_kernels STATIC ${KERNEL_SOURCES})
target_link_libraries(thot_kernels PUBLIC CUDA::cublas)
set_target_properties(thot_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

add_executable(${PROJECT_NAME} main.cpp ${HEADERS})
target_link_libraries(${PROJECT_NAME} PRIVATE
        -Wl,--whole-archive thot_kernels -Wl,--no-whole-archive
        CUDA::cudart
        $<$<BOOL:${THOT_USE_CUDNN}>:cudnn::cudnn>
        pthread dl
)
target_compile_definitions(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:_DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
        $<$<BOOL:${THOT_USE_CUDNN}>:THOT_WITH_CUDNN=1>
)
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

###################### TESTs ######################

add_executable(cuda_tests tests/cuda_tests.cpp)
target_link_libraries(cuda_tests PRIVATE
        -Wl,--whole-archive thot_kernels -Wl,--no-whole-archive
        CUDA::cudart
        $<$<BOOL:${THOT_USE_CUDNN}>:cudnn::cudnn>
        pthread dl
)
set_target_properties(cuda_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
target_compile_definitions(cuda_tests PRIVATE
        $<$<BOOL:${THOT_USE_CUDNN}>:THOT_WITH_CUDNN=1>
)

add_test(NAME cuda_tests COMMAND cuda_tests)
