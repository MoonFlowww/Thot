#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include "../cuda/cuh/LowRankCuda/lowrank.cuh"



namespace Thot {
    namespace Utils {

        inline void checkCuda(cudaError_t result, const char* msg) {
            if (result != cudaSuccess) {
                std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        inline void cuda_allocate(void** devPtr, size_t size) {
            checkCuda(cudaMalloc(devPtr, size), "cudaMalloc failed");
        }

        inline void cuda_free(void* devPtr) {
            checkCuda(cudaFree(devPtr), "cudaFree failed");
        }

        inline void cuda_memcpy_to_device(void* dst, const void* src, size_t count) {
            checkCuda(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice), "cudaMemcpy to device failed");
        }

        inline void cuda_memcpy_to_host(void* dst, const void* src, size_t count) {
            checkCuda(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost), "cudaMemcpy to host failed");
        }

        class Tensor {
        private:
            float* data_ = nullptr;
            std::vector<int> shape_;
            std::vector<int> strides_;
            size_t size_ = 0;
            size_t byte_size_ = 0;

            void compute_strides() {
                strides_.resize(shape_.size());
                int acc = 1;
                for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
                    strides_[i] = acc;
                    acc *= shape_[i];
                }
            }

        public:
            Tensor() = default;

            explicit Tensor(const std::vector<int>& shape, bool init_zero = false) {
                reshape(shape, init_zero);
            }

            ~Tensor() {
                if (data_) cuda_free(data_);
            }

            // Move semantics only
            Tensor(const Tensor&) = delete;
            Tensor& operator=(const Tensor&) = delete;

            Tensor(Tensor&& other) noexcept
                : data_(other.data_), shape_(std::move(other.shape_)),
                strides_(std::move(other.strides_)), size_(other.size_), byte_size_(other.byte_size_) {
                other.data_ = nullptr;
                other.size_ = 0;
                other.byte_size_ = 0;
            }

            Tensor& operator=(Tensor&& other) noexcept {
                if (this != &other) {
                    if (data_) cuda_free(data_);
                    data_ = other.data_;
                    shape_ = std::move(other.shape_);
                    strides_ = std::move(other.strides_);
                    size_ = other.size_;
                    byte_size_ = other.byte_size_;
                    other.data_ = nullptr;
                    other.size_ = 0;
                    other.byte_size_ = 0;
                }
                return *this;
            }

            void reshape(const std::vector<int>& shape, bool init_zero = false) {
                if (shape.empty()) throw std::runtime_error("Invalid shape");

                shape_ = shape;
                size_ = 1;
                for (int dim : shape) {
                    if (dim <= 0) throw std::runtime_error("Shape dimension must be positive");
                    size_ *= dim;
                }
                byte_size_ = size_ * sizeof(float);

                compute_strides();

                if (data_) cuda_free(data_);
                cuda_allocate(reinterpret_cast<void**>(&data_), byte_size_);

                if (init_zero) {
                    std::vector<float> zeros(size_, static_cast<float>(0));
                    cuda_memcpy_to_device(data_, zeros.data(), byte_size_);
                }
            }

            void upload(const std::vector<float>& host_data) {
                if (host_data.size() != size_) throw std::runtime_error("Upload size mismatch");
                cuda_memcpy_to_device(data_, host_data.data(), byte_size_);
            }

            void download(std::vector<float>& host_data) const {
                host_data.resize(size_);
                cuda_memcpy_to_host(host_data.data(), data_, byte_size_);
            }

            std::vector<float> download() const {
                std::vector<float> host_data(size_);
                cuda_memcpy_to_host(host_data.data(), data_, byte_size_);
                return host_data;
            }

            bool is_contiguous() const {
                int expected_stride = 1;
                for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
                    if (strides_[i] != expected_stride) return false;
                    expected_stride *= shape_[i];
                }
                return true;
            }

            void validate() const {
                if (!data_ || size_ == 0) throw std::runtime_error("Invalid tensor state");
            }

            float* data() const { return data_; }
            const std::vector<int>& shape() const { return shape_; }
            const std::vector<int>& strides() const { return strides_; }
            size_t size() const { return size_; }
            size_t bytes() const { return byte_size_; }

            void add(const Tensor& other) {
                if (size_ != other.size_) {
                    throw std::invalid_argument("Tensor dimensions don't match for addition");
                }
                launchAdd(
                    data_,
                    other.data_,
                    data_, // in-place operation
                    static_cast<int>(size_)
                );
            }

            // Element-wise multiplication: this *= other
            void multiply(const Tensor& other) {
                if (size_ != other.size_) {
                    throw std::invalid_argument("Tensor dimensions don't match for multiplication");
                }
                
                launchMultiply(
                    data_,
                    other.data_,
                    data_, // in-place operation
                    static_cast<int>(size_)
                );
            }

            // Scalar addition: this += scalar
            void add_scalar(float scalar) {
                launchAddScalar(
                    data_,
                    scalar,
                    data_, // in-place operation
                    static_cast<int>(size_)
                );
            }

            // Scalar multiplication: this *= scalar
            void multiply_scalar(float scalar) {
                launchMultiplyScalar(
                    data_,
                    scalar,
                    data_, // in-place operation
                    static_cast<int>(size_)
                );
            }

            // Return a new tensor as the result of addition: result = this + other
            Tensor add_new(const Tensor& other) const {
                if (size_ != other.size_) {
                    throw std::invalid_argument("Tensor dimensions don't match for addition");
                }

                Tensor result(shape_);

                launchAdd(
                    data_,
                    other.data_,
                    result.data_,
                    static_cast<int>(size_)
                );

                return result;
            }

            // Return a new tensor as the result of multiplication: result = this * other
            Tensor multiply_new(const Tensor& other) const {
                if (size_ != other.size_) {
                    throw std::invalid_argument("Tensor dimensions don't match for multiplication");
                }

                Tensor result(shape_);

                launchMultiply(
                    data_,
                    other.data_,
                    result.data_,
                    static_cast<int>(size_)
                );

                return result;
            }
        };

    } // namespace Utils
} // namespace Thot
