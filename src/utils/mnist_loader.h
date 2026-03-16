#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <string>
#include <vector>
#include <fstream>

/**
 * MNIST图像数据结构
 * [magic number]   4 Bytes 0-3
 * [number of images]  4 Bytes 4-7
 * [rows]      4 Bytes 8-11
 * [cols]      4 Bytes 12-15
 * [pixel data...]
 */
struct MNISTImages {
    int magic;
    int count;
    int rows;
    int cols;
    std::vector<unsigned char> data;
};

/**
 * MNIST标签数据结构
 * [magic number]    4 Bytes 0-3
 * [number of labels]  4 Bytes 4-7 
 * [label data...]
 */
struct MNISTLabels {
    int magic;
    int count;
    std::vector<unsigned char> data;
};

namespace MNISTUtils {

/**
 * 读取大端格式的32位无符号整数
 * IDX文件格式使用大端(big-endian)字节序
 * @param f 输入文件流
 * @return 转换后的int值
 */
int readBigEndianUInt32(std::ifstream &f);

/**
 * 加载MNIST图像数据
 * @param path 图像数据文件路径
 * @param verbose 是否打印加载信息（默认true）
 * @return MNISTImages 图像数据
 */
MNISTImages loadMNISTImages(const std::string &path, bool verbose = true);

/**
 * 加载MNIST标签数据
 * @param path 标签数据文件路径
 * @param verbose 是否打印加载信息（默认true）
 * @return MNISTLabels 标签数据
 */
MNISTLabels loadMNISTLabels(const std::string &path, bool verbose = true);

} // namespace MNISTUtils

#endif // MNIST_LOADER_H
