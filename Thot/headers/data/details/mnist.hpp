#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Thot {
    namespace Data {

        inline int32_t swap_endian(int32_t val) {
            return ((val >> 24) & 0xff) | ((val << 8) & 0xff0000) |
                ((val >> 8) & 0xff00) | ((val << 24) & 0xff000000);
        }

        inline std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> load_mnist(
            const std::string& images_path,
            const std::string& labels_path,
            int num_classes = 10,
            float ratio = 1.0f
        ) {
            std::ifstream images_file(images_path, std::ios::binary);
            std::ifstream labels_file(labels_path, std::ios::binary);

            if (!images_file) throw std::runtime_error("Cannot open MNIST images file: " + images_path);
            if (!labels_file) throw std::runtime_error("Cannot open MNIST labels file: " + labels_path);

            int magic_number, num_images, rows, cols;
            images_file.read(reinterpret_cast<char*>(&magic_number), 4);
            images_file.read(reinterpret_cast<char*>(&num_images), 4);
            images_file.read(reinterpret_cast<char*>(&rows), 4);
            images_file.read(reinterpret_cast<char*>(&cols), 4);

            magic_number = swap_endian(magic_number);
            num_images = swap_endian(num_images);
            rows = swap_endian(rows);
            cols = swap_endian(cols);

            int image_size = rows * cols;
            int actual_num_images = static_cast<int>(num_images * ratio);

            std::cout << "\nMNIST Dataset Information:\n";
            std::cout << "-------------------------\n";
            std::cout << "Total images available: " << num_images << "\n";
            std::cout << "Loading ratio: " << std::fixed << std::setprecision(2) << ratio * 100 << "%\n";
            std::cout << "Images to load: " << actual_num_images << "\n";
            std::cout << "Image dimensions: " << rows << "x" << cols << " (" << image_size << " pixels)\n";
            std::cout << "Number of classes: " << num_classes << "\n";
            std::cout << "Memory usage (approx):\n";
            std::cout << "  - Images: " << std::setprecision(2) << (actual_num_images * image_size * sizeof(float)) / (1024.0 * 1024.0) << " MB\n";
            std::cout << "  - Labels: " << std::setprecision(2) << (actual_num_images * num_classes * sizeof(float)) / (1024.0 * 1024.0) << " MB\n";
            std::cout << "-------------------------\n\n";

            std::vector<std::vector<float>> images(actual_num_images, std::vector<float>(image_size));
            std::vector<std::vector<float>> labels(actual_num_images, std::vector<float>(num_classes, 0.0f));

            for (int i = 0; i < actual_num_images; ++i) {
                std::vector<uint8_t> buffer(image_size);
                images_file.read(reinterpret_cast<char*>(buffer.data()), image_size);
                for (int j = 0; j < image_size; ++j) {
                    images[i][j] = buffer[j] / 255.0f;
                }
            }

            int labels_magic, num_labels;
            labels_file.read(reinterpret_cast<char*>(&labels_magic), 4);
            labels_file.read(reinterpret_cast<char*>(&num_labels), 4);
            labels_magic = swap_endian(labels_magic);
            num_labels = swap_endian(num_labels);

            std::vector<int> label_counts(num_classes, 0);
            for (int i = 0; i < actual_num_images; ++i) {
                uint8_t label;
                labels_file.read(reinterpret_cast<char*>(&label), 1);
                labels[i][label] = 1.0f;
                label_counts[label]++;
            }

            std::cout << "Label Distribution:\n";
            std::cout << "------------------\n";
            for (int i = 0; i < num_classes; ++i) {
                double percentage = (100.0 * label_counts[i]) / actual_num_images;
                std::cout << "Class " << i << ": " << label_counts[i] << " images ("
                    << std::fixed << std::setprecision(2) << percentage << "%)\n";
            }
            std::cout << "------------------\n\n";

            return { images, labels };
        }

        inline std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> load_mnist_train(
            const std::string& base_path = "",
            float ratio = 1.0f
        ) {
            std::cout << "Loading MNIST Training Set...\n";
            std::string images_path = base_path + "\\train-images.idx3-ubyte";
            std::string labels_path = base_path + "\\train-labels.idx1-ubyte";
            return load_mnist(images_path, labels_path, 10, ratio);
        }

        inline std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> load_mnist_test(
            const std::string& base_path = "",
            float ratio = 1.0f
        ) {
            std::cout << "Loading MNIST Test Set...\n";
            std::string images_path = base_path + "\\t10k-images.idx3-ubyte";
            std::string labels_path = base_path + "\\t10k-labels.idx1-ubyte";
            return load_mnist(images_path, labels_path, 10, ratio);
        }
    }
}