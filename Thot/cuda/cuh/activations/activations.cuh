#pragma once

#include <cuda_runtime.h>

namespace cuda {
	namespace activations {

		__global__ void relu_forward(const float* input, float* output, int size);
		__global__ void relu_backward(const float* grad_output, const float* input, float* grad_input, int size);

		__global__ void sigmoid_forward(const float* input, float* output, int size);
		__global__ void sigmoid_backward(const float* grad_output, const float* output, float* grad_input, int size);

		__global__ void tanh_forward(const float* input, float* output, int size);
		__global__ void tanh_backward(const float* grad_output, const float* output, float* grad_input, int size);

		__global__ void leaky_relu_forward(const float* input, float* output, float negative_slope, int size);
		__global__ void leaky_relu_backward(const float* grad_output, const float* input, float* grad_input, float negative_slope, int size);

		__global__ void elu_forward(const float* input, float* output, float alpha, int size);
		__global__ void elu_backward(const float* grad_output, const float* output, const float* input, float* grad_input, float alpha, int size);

		__global__ void gelu_forward(const float* input, float* output, int size);
		__global__ void gelu_backward(const float* grad_output, const float* input, float* grad_input, int size);

		__global__ void softmax_forward(const float* input, float* output, int batch_size, int feature_dim);
		__global__ void softmax_backward(const float* grad_output, const float* output, float* grad_input, int batch_size, int feature_dim);



		// Wrapper functions for launching kernels
		void launchReluForward(const float* input, float* output, int size, cudaStream_t stream = 0);
		void launchReluBackward(const float* grad_output, const float* input, float* grad_input, int size, cudaStream_t stream = 0);

		void launchSigmoidForward(const float* input, float* output, int size, cudaStream_t stream = 0);
		void launchSigmoidBackward(const float* grad_output, const float* output, float* grad_input, int size, cudaStream_t stream = 0);

		void launchTanhForward(const float* input, float* output, int size, cudaStream_t stream = 0);
		void launchTanhBackward(const float* grad_output, const float* output, float* grad_input, int size, cudaStream_t stream = 0);

		void launchLeakyReluForward(const float* input, float* output, float negative_slope, int size, cudaStream_t stream = 0);
		void launchLeakyReluBackward(const float* grad_output, const float* input, float* grad_input, float negative_slope, int size, cudaStream_t stream = 0);

		void launchEluForward(const float* input, float* output, float alpha, int size, cudaStream_t stream = 0);
		void launchEluBackward(const float* grad_output, const float* output, const float* input, float* grad_input, float alpha, int size, cudaStream_t stream = 0);

		void launchGeluForward(const float* input, float* output, int size, cudaStream_t stream = 0);
		void launchGeluBackward(const float* grad_output, const float* input, float* grad_input, int size, cudaStream_t stream = 0);

		void launchSoftmaxForward(const float* input, float* output, int batch_size, int feature_dim, cudaStream_t stream = 0);
		void launchSoftmaxBackward(const float* grad_output, const float* output, float* grad_input, int batch_size, int feature_dim, cudaStream_t stream = 0);

	}  // namespace activations
}  // namespace cuda
