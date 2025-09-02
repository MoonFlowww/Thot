#ifndef THOT_CIFAR_HPP
#define THOT_CIFAR_HPP
#pragma once
#include <cstdint>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <tuple>

namespace Thot {
    namespace Data {


        inline std::vector<std::string> get_CIFAR10_label_string(const std::string& meta_path) {
            std::ifstream file(meta_path);
            if (!file) throw std::runtime_error("Cannot open CIFAR-10 meta file: " + meta_path);

            std::vector<std::string> label_names;
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty()) {
                    if (!line.empty() && line.back() == '\r') {
                        line.pop_back();
                    }
                    label_names.push_back(line);
                }
            }

            if (label_names.size() != 10) {
                throw std::runtime_error("CIFAR-10 meta file should contain 10 label names, found " + std::to_string(label_names.size()));
            }

            return label_names;
        }

        inline std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> load_cifar10(const std::string& file_path, int num_classes = 10, float ratio = 1.0f) {
            std::ifstream file(file_path, std::ios::binary);
            if (!file) throw std::runtime_error("Cannot open CIFAR-10 file: " + file_path);

            //TODO: rewrite label string
            // computed 2 times for train & test + not saved for other std::cout << class
            std::string p = file_path;
            std::size_t pos = p.find_last_of('/');
            if (pos != std::string::npos) p.erase(pos);
            std::vector<std::string> label_string = get_CIFAR10_label_string(p+"/batches.meta.txt");

            // CIFAR-10 fixed sizes
            const int image_size = 32 * 32 * 3;
            const int record_size = 1 + image_size;

            // Count records in file
            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);

            size_t num_images = file_size / record_size;
            size_t actual_num_images = static_cast<size_t>(num_images * ratio);

            std::cout << "\nCIFAR-10 Dataset Information:\n";
            std::cout << "-------------------------\n";
            std::cout << "Total images available: " << num_images << "\n";
            std::cout << "Loading ratio: " << std::fixed << std::setprecision(2) << ratio * 100 << "%\n";
            std::cout << "Images to load: " << actual_num_images << "\n";
            std::cout << "Image dimensions: 32x32x3 (" << image_size << " values)\n";
            std::cout << "Number of classes: " << num_classes << "\n";
            std::cout << "Memory usage (approx):\n";
            std::cout << "  - Images: " << std::setprecision(2) << (actual_num_images * image_size * sizeof(float)) / (1024.0 * 1024.0) << " MB\n";
            std::cout << "  - Labels: " << std::setprecision(2) << (actual_num_images * num_classes * sizeof(float)) / (1024.0 * 1024.0) << " MB\n";
            std::cout << "-------------------------\n\n";

            std::vector<std::vector<float>> images(actual_num_images, std::vector<float>(image_size));
            std::vector<std::vector<float>> labels(actual_num_images, std::vector<float>(num_classes, 0.0f));

            std::vector<int> label_counts(num_classes, 0);

            for (size_t i = 0; i < actual_num_images; ++i) {
                uint8_t label;
                std::vector<std::uint8_t> buffer(image_size);

                file.read(reinterpret_cast<char*>(&label), 1);
                file.read(reinterpret_cast<char*>(buffer.data()), image_size);

                // Normalize image data
                for (int j = 0; j < image_size; ++j) {
                    images[i][j] = buffer[j] / 255.0f;
                }

                labels[i][label] = 1.0f;
                label_counts[label]++;
            }

            std::cout << "Label Distribution:\n";
            std::cout << "------------------\n";
            for (int i = 0; i < num_classes; ++i) {
                double percentage = (100.0 * label_counts[i]) / actual_num_images;
                std::cout << "Class " << (label_string.empty() ? std::to_string(i) : std::to_string(i)+" "+label_string[i]) << ": " << label_counts[i] << " images ("
                          << std::fixed << std::setprecision(2) << percentage << "%)\n";
            }
            std::cout << "------------------\n\n";

            return { images, labels };
        }

        inline std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> Load_CIFAR10_Train(const std::string& base_path = "", float ratio = 1.0f) {
            std::cout << "Loading CIFAR-10 Training Set...\n";
            std::vector<std::vector<float>> train_images;
            std::vector<std::vector<float>> train_labels;

            for (int batch = 1; batch <= 5; ++batch) {
                std::string path = base_path + "/data_batch_" + std::to_string(batch) + ".bin";
                auto [images, labels] = load_cifar10(path, 10, ratio);
                train_images.insert(train_images.end(), images.begin(), images.end());
                train_labels.insert(train_labels.end(), labels.begin(), labels.end());
            }
            return { train_images, train_labels };
        }

        inline std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
        Load_CIFAR10_Test(const std::string& base_path = "", float ratio = 1.0f) {
            std::cout << "Loading CIFAR-10 Test Set...\n";
            std::string path = base_path + "/test_batch.bin";
            return load_cifar10(path, 10, ratio);
        }

        inline std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>, std::vector<std::vector<float>>, std::vector<std::vector<float>>
        > Load_CIFAR10(const std::string& base_path = "", float train_ratio = 1.0f, float test_ratio = 1.0f) {
            auto [train_images, train_labels] = Load_CIFAR10_Train(base_path, train_ratio);
            auto [test_images, test_labels] = Load_CIFAR10_Test(base_path, test_ratio);
            return { train_images, train_labels, test_images, test_labels };
        }





    }
}




#endif //THOT_CIFAR_HPP