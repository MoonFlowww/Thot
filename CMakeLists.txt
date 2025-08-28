cmake_minimum_required(VERSION 3.22)
project(Thot LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()



enable_testing()



include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Thot/cuda/cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/Thot/headers
)

set(SOURCES
    main.cpp
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

# Headers (for IDEs)
set(HEADERS
    Thot/cuda/cuh/activations/activations.cuh
    Thot/cuda/cuh/layers/conv2d.cuh
    Thot/cuda/cuh/layers/fc.cuh
    Thot/cuda/cuh/layers/rbm.cuh
    Thot/cuda/cuh/layers/rnn.cuh
    Thot/cuda/cuh/losses/loss.cuh
    Thot/cuda/cuh/LowRankCuda/lowrank.cuh
    Thot/cuda/cuh/optimizations/adam.cuh
    Thot/cuda/cuh/optimizations/sgd.cuh
    Thot/cuda/cuh/optimizations/sgdm.cuh
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

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_compile_definitions(${PROJECT_NAME}
    PRIVATE
        $<$<CONFIG:Debug>:WIN32;WIN64;_DEBUG;_CONSOLE>
        $<$<CONFIG:Release>:WIN32;WIN64;NDEBUG;_CONSOLE>
)

if(WIN32)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            cudart_static
            kernel32
            user32
            gdi32
            winspool
            comdlg32
            advapi32
            shell32
            ole32
            oleaut32
            uuid
            odbc32
            odbccp32
    )
else() # Linux
    find_package(CUDAToolkit REQUIRED)

    target_include_directories(${PROJECT_NAME}
            PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(${PROJECT_NAME}
            PRIVATE
            CUDA::cudart
            pthread
            dl
    )
endif()

# CUDA
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 60-real;70-real;75-real;80-real;86-real;89-real;90
)




add_executable(rnn_sequence_test
        tests/rnn_sequence_test.cpp
        Thot/cuda/cu/layers/rnn.cu)

target_include_directories(rnn_sequence_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Thot/cuda/cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/Thot/headers
)

if(WIN32)
    target_link_libraries(rnn_sequence_test PRIVATE
            cudart_static)
else()
    target_include_directories(rnn_sequence_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(rnn_sequence_test PRIVATE
            CUDA::cudart
            pthread
            dl)
endif()

set_target_properties(rnn_sequence_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 60-real;70-real;75-real;80-real;86-real;89-real;90)

add_test(NAME rnn_sequence_test COMMAND rnn_sequence_test)