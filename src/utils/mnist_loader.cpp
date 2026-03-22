#include "mnist_loader.h"
#include <iostream>
#include <stdexcept>
#include <cerrno>
#include <cstring>
#include <cstdint>

namespace MNISTUtils {

int readBigEndianUInt32(std::ifstream &f) {
    unsigned char bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    if (!f) {
        throw std::runtime_error("Failed to read 4 bytes from MNIST file header");
    }
    std::uint32_t value = (static_cast<std::uint32_t>(bytes[0]) << 24) |   // 按位左移进行换序
                          (static_cast<std::uint32_t>(bytes[1]) << 16) |
                          (static_cast<std::uint32_t>(bytes[2]) << 8)  |
                          (static_cast<std::uint32_t>(bytes[3]));
    return static_cast<int>(value);
}

void normalizeMNISTImages(MNISTImages& images) {
    const double denom = 127.5; // 归一化范围到 [-1, 1]
    images.data_normalized.resize(images.data.size());
    for (std::size_t i = 0; i < images.data.size(); ++i) {
        images.data_normalized[i] =
            static_cast<double>(images.data[i]) / denom - 1.0;
    }
}

MNISTImages loadMNISTImages(const std::string &path, bool verbose) {
    // 只读方式打开（二进制），避免因为默认 in|out 模式而要求写权限
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "无法打开文件: " << path << std::endl;
        std::cerr << "错误原因: " << strerror(errno) << std::endl;
        throw std::runtime_error("Failed to open file: " + path);
    }

    MNISTImages images;
    images.magic = readBigEndianUInt32(f);
    images.count = readBigEndianUInt32(f);
    images.rows  = readBigEndianUInt32(f);
    images.cols  = readBigEndianUInt32(f);

    // 根据 header 计算像素总数：count * rows * cols
    const std::size_t total_pixels =
        static_cast<std::size_t>(images.count) *
        static_cast<std::size_t>(images.rows) *
        static_cast<std::size_t>(images.cols);

    images.data.resize(total_pixels);
    f.read(reinterpret_cast<char*>(images.data.data()), images.data.size());
    if (!f) {
        throw std::runtime_error("Failed to read MNIST image pixels: " + path);
    }

    normalizeMNISTImages(images);

    if (verbose) {
        std::cout << "magic=" << images.magic
                  << " count=" << images.count
                  << " rows=" << images.rows
                  << " cols=" << images.cols
                  << std::endl;
    }

    return images;
}

MNISTLabels loadMNISTLabels(const std::string &path, bool verbose) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "无法打开文件: " << path << std::endl;
        std::cerr << "错误原因: " << strerror(errno) << std::endl;
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    MNISTLabels labels;
    labels.magic = readBigEndianUInt32(f);
    labels.count = readBigEndianUInt32(f);
    labels.data.resize(labels.count);
    f.read(reinterpret_cast<char*>(labels.data.data()), labels.data.size());
    
    if (verbose) {
        std::cout << "magic=" << labels.magic
                  << " count=" << labels.count
                  << std::endl;
    }
    
    return labels;
}

} // namespace MNISTUtils
