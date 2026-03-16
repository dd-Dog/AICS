# MNIST 数据加载工具类

## 简介

`mnist_loader` 提供了加载 MNIST 数据集的工具函数，支持 IDX 格式的图像和标签文件。

## 文件说明

- `mnist_loader.h` - 头文件，包含数据结构和函数声明
- `mnist_loader.cpp` - 实现文件，包含具体实现

## 数据结构

### MNISTImages
```cpp
struct MNISTImages {
    int magic;                    // 魔数
    int count;                    // 图像数量
    int rows;                     // 图像行数
    int cols;                     // 图像列数
    std::vector<unsigned char> data;  // 图像数据（按行展开）
};
```

### MNISTLabels
```cpp
struct MNISTLabels {
    int magic;                    // 魔数
    int count;                    // 标签数量
    std::vector<unsigned char> data;  // 标签数据
};
```

## 使用方法

### 1. 包含头文件
```cpp
#include "../utils/mnist_loader.h"
using namespace MNISTUtils;
```

### 2. 加载数据
```cpp
// 加载训练图像
MNISTImages train_images = loadMNISTImages("path/to/train-images-idx3-ubyte");

// 加载训练标签
MNISTLabels train_labels = loadMNISTLabels("path/to/train-labels-idx1-ubyte");

// 加载测试数据（可选，不打印详细信息）
MNISTImages test_images = loadMNISTImages("path/to/t10k-images-idx3-ubyte", false);
MNISTLabels test_labels = loadMNISTLabels("path/to/t10k-labels-idx1-ubyte", false);
```

### 3. 访问数据
```cpp
// 访问第 i 张图像（28x28=784像素）
int image_index = 0;
int dim = train_images.rows * train_images.cols;  // 784
int offset = image_index * dim;

for (int j = 0; j < dim; j++) {
    unsigned char pixel = train_images.data[offset + j];
    // 处理像素值
}

// 访问标签
unsigned char label = train_labels.data[image_index];
```

## 编译说明

使用 `run.bat` 编译时，如果源文件包含 `utils/` 目录的头文件，会自动链接 `utils/mnist_loader.cpp`。

手动编译示例：
```bash
g++ -std=c++17 -I./utils -o program.exe your_file.cpp utils/mnist_loader.cpp
```

## 注意事项

1. **文件路径**：确保使用正确的相对路径或绝对路径
2. **字节序**：IDX 格式使用大端(big-endian)字节序，工具类已自动处理
3. **数据格式**：图像数据为 `unsigned char` 类型，值域 0-255
4. **内存管理**：数据结构使用 `std::vector`，自动管理内存

## 示例

完整示例请参考 `algorithms/04_LogicalRegresssion_2_Categories.cpp`
