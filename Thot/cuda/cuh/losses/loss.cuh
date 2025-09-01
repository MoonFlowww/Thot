#pragma once

#include <cuda_runtime.h>


namespace cuda {
	namespace losses {

	    extern __device__ bool verbose;

		__global__ void mse(const float* predictions, const float* targets, float* loss, int size);
		__global__ void mseGradient(const float* predictions, const float* targets, float* gradients, int size);

		__global__ void mae(const float* predictions, const float* targets, float* loss, int size);
		__global__ void maeGradient(const float* predictions, const float* targets, float* gradients, int size);

	    __global__ void binaryCrossEntropy(const float* predictions, const float* targets, float* loss, int size, float epsilon);
	    __global__ void binaryCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon);

	    __global__ void crossEntropy(const float* predictions, const float* targets, float* loss, int size, float epsilon);
	    __global__ void crossEntropyGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon);

		__global__ void categoricalCrossEntropy(const float* predictions, const float* targets, float* loss, int batch_size, int num_classes, float epsilon);
		__global__ void categoricalCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int batch_size, int num_classes, float epsilon);

		__global__ void sparseCategoricalCrossEntropy(const float* predictions, const float* targets, float* loss, int batch_size, int num_classes, float epsilon);
		__global__ void sparseCategoricalCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int batch_size, int num_classes, float epsilon);

		__global__ void hinge(const float* predictions, const float* targets, float* loss, int size);
		__global__ void hingeGradient(const float* predictions, const float* targets, float* gradients, int size);

		__global__ void huber(const float* predictions, const float* targets, float* loss, int size, float delta);
		__global__ void huberGradient(const float* predictions, const float* targets, float* gradients, int size, float delta);

		__global__ void klDivergence(const float* predictions, const float* targets, float* loss, int size, float epsilon);
		__global__ void klDivergenceGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon);


		void launchMSE(const float* predictions, const float* targets, float* loss, int size, cudaStream_t stream = nullptr);
		void launchMSEGradient(const float* predictions, const float* targets, float* gradients, int size, cudaStream_t stream = nullptr);

		void launchMAE(const float* predictions, const float* targets, float* loss, int size, cudaStream_t stream = nullptr);
		void launchMAEGradient(const float* predictions, const float* targets, float* gradients, int size, cudaStream_t stream = nullptr);

	    void launchBinaryCrossEntropy(const float* predictions, const float* targets, float* loss, int size, float epsilon, cudaStream_t stream = nullptr);
	    void launchBinaryCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon, cudaStream_t stream = nullptr);

	    void launchCrossEntropy(const float* predictions, const float* targets, float* loss, int size, float epsilon, cudaStream_t stream = nullptr);
	    void launchCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon, cudaStream_t stream = nullptr);

		void launchCategoricalCrossEntropy(const float* predictions, const float* targets, float* loss, int batch_size, int num_classes, float epsilon, cudaStream_t stream = nullptr);
		void launchCategoricalCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int batch_size, int num_classes, float epsilon, cudaStream_t stream = nullptr);

		void launchSparseCategoricalCrossEntropy(const float* predictions, const float* targets, float* loss, int batch_size, int num_classes, float epsilon, cudaStream_t stream = nullptr);
		void launchSparseCategoricalCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int batch_size, int num_classes, float epsilon, cudaStream_t stream = nullptr);

		void launchHinge(const float* predictions, const float* targets, float* loss, int size, cudaStream_t stream = nullptr);
		void launchHingeGradient(const float* predictions, const float* targets, float* gradients, int size, cudaStream_t stream = nullptr);

		void launchHuber(const float* predictions, const float* targets, float* loss, int size, float delta, cudaStream_t stream = nullptr);
		void launchHuberGradient(const float* predictions, const float* targets, float* gradients, int size, float delta, cudaStream_t stream = nullptr);

		void launchKLDivergence(const float* predictions, const float* targets, float* loss, int size, float epsilon, cudaStream_t stream = nullptr);
		void launchKLDivergenceGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon, cudaStream_t stream = nullptr);

	    float reduceLoss(float* loss, int size, cudaStream_t stream = nullptr);
	}  //  losses
}  // namespace cuda 