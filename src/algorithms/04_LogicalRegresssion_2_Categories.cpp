#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <cstdint>
#include <cmath>
#include "../utils/mnist_loader.h"

using namespace std;
using namespace MNISTUtils;

/**
图片二分类

1. 图片数据(IDX格式)输入，展平为数组
2. 权重及偏移量向量化
3. 运算向量化

*/

vector<double> weights;
double bias = 0.5, learnRate=0.01;

double nomalization(unsigned char x){
    return x/127.5 - 1.0;
}

// 只保留标签为 0 或 1 的样本，并同步裁剪图像与标签
void prune_data(MNISTImages &images, MNISTLabels &labels){
    const int dim = images.rows * images.cols;  // 每张图片的像素数

    std::vector<unsigned char> new_image_data;
    std::vector<unsigned char> new_label_data;

    new_image_data.reserve(labels.count * dim); // 预留大致空间，避免频繁扩容
    new_label_data.reserve(labels.count);

    for (int i = 0; i < labels.count; i++) {
        unsigned char lbl = labels.data[i];
        if (lbl == 0 || lbl == 1) {
            // 保留该样本
            new_label_data.push_back(lbl);

            int offset = i * dim;
            for (int j = 0; j < dim; j++) {
                new_image_data.push_back(images.data[offset + j]);
            }
        }
    }

    // 更新 count 和数据
    int new_count = static_cast<int>(new_label_data.size());
    images.count = new_count;
    labels.count = new_count;
    images.data.swap(new_image_data);
    labels.data.swap(new_label_data);

    cout << "After pruning, remaining samples (label 0 or 1) = " << new_count << endl;
}

/*
向量的点乘运算
*/
double dot_product(const MNISTImages &images, int x, int dim){
    double z = 0.0;
    int offset = x*dim;
    for(int i=0; i<dim; i++){
        z += nomalization(images.data[offset + i]) * weights[i];
    }
    return z + bias;
}

/*
z很大时，sigmoid输出接近0，在计算loss时会被截断1e-12,所有极小概率变成一个数
*/
double sigmoid(double z){
    if(z >= 0){
        return 1.0/(1.0 + exp(-z));
    }else {
        double e = exp(z); 
        return e/(1.0 + e);
    }
}
/*
z很大时，sigmoid输出接近0，在计算loss时会被截断1e-12,所有极小概率变成一个数，loss被量化
*/
double cross_entropy_loss(double y, double p){
    const double eps = 1e-12;
    p = max(eps, min(p, 1-eps));
    return -(y * log(p) + (1-y) * log(1-p));
}


/**
不用P计算损失，因为会失真.
二分类交叉熵
*/
double bce_with_logits(double z, double y)
{
    if (z >= 0)
        return z - z * y + log1p(exp(-z));
    else
        return -z * y + log1p(exp(z));
}

void train(const MNISTImages &train_images, const MNISTLabels &train_labels,int size, int epochs, double learnRate){
    int dim = 784;
    for(int epoch=0; epoch<epochs; epoch++){
        // cout << "epoch=" << epoch << endl;
        double total_loss = 0.0;
        vector<double> gradient_w(dim);
        double gradient_b = 0.0;
        int train_size = min(train_images.count, size);
        for(int i = 0; i < train_size; i++){
            // cout << "i=" << i << endl;
            //取特征值并和权重向量运算，28x28=784个
            double z = dot_product(train_images, i, dim);
            //激活函数
            double p = sigmoid(z);
            //交叉熵损失,使用bce logits
            total_loss += bce_with_logits(z, train_labels.data[i]);
            // cout << "z=" << z << ", p=" << p << ",total_loss=" << total_loss << endl;
            
            //梯度累积
            int offset = i * dim; 
            for(int g=0; g<dim; g++){
                gradient_w[g] += (p - train_labels.data[i]) * nomalization(train_images.data[offset+g]);
                // if(g<10) cout << gradient_w[g] << ", ";
            }
            // cout << endl;

            gradient_b += p - train_labels.data[i];
        }
        //平均损失
        total_loss /= train_size;
        //更新梯度
        for(int i=0; i<dim; i++){
            weights[i] -= learnRate * gradient_w[i]/train_size;
            // if(i<10) cout << weights[i] << ", ";
        }
        // cout << endl;

        bias -= learnRate * gradient_b/train_size;
        cout << "epoch=" << epoch << ", total_loss=" << total_loss << endl;
    }
}


void test(const MNISTImages &test_images, const MNISTLabels &test_labels){
    int dim = 784;
    double loss = 0.0, total_loss = 0.0;
    int correct = 0;
    for(int i=0; i<test_images.count; i++){
        double z = dot_product(test_images, i, dim);
        double p = sigmoid(z);

        int y_hat = (p > 0.5);
        loss = bce_with_logits(z, test_labels.data[i]);
        total_loss += loss;
        int y = static_cast<int>(test_labels.data[i]);
        if(y == y_hat) correct ++;
        cout << "i=" << i << ", y=" << static_cast<int>(test_labels.data[i]) << ", y_hat=" << y_hat << ", loss=" << loss << endl;
    }
    total_loss /= test_images.count;
    cout << "total_loss=" << total_loss << ",accuracy=" << correct*1.0/test_images.count <<  endl;
}

int main(){
    namespace fs = std::filesystem;

    // 相对路径（从 src/ 目录到数据文件）
    std::string train_MNISTImage_path = "..\\data\\raw\\MNIST\\raw\\train-images-idx3-ubyte";
    std::string train_MNISTLabel_path = "..\\data\\raw\\MNIST\\raw\\train-labels-idx1-ubyte";
    std::string test_MNISTImage_path = "..\\data\\raw\\MNIST\\raw\\t10k-images-idx3-ubyte";
    std::string test_MNISTLabel_path = "..\\data\\raw\\MNIST\\raw\\t10k-labels-idx1-ubyte";
    MNISTImages train_MNISTImages = loadMNISTImages(train_MNISTImage_path);
    MNISTLabels train_MNISTLabels = loadMNISTLabels(train_MNISTLabel_path);
    MNISTImages test_MNISTImages = loadMNISTImages(test_MNISTImage_path);
    MNISTLabels test_MNISTLabels = loadMNISTLabels(test_MNISTLabel_path);
    
    cout << "Image buffer size: " << train_MNISTImages.data.size() << endl;
    cout << "Label buffer size: " << train_MNISTLabels.data.size() << endl;
    cout << "Image buffer size: " << test_MNISTImages.data.size() << endl;
    cout << "Label buffer size: " << test_MNISTLabels.data.size() << endl;

    prune_data(train_MNISTImages, train_MNISTLabels);
    prune_data(test_MNISTImages, test_MNISTLabels);
    cout << "=====数据清洗处理====" << endl;
    cout << "Image buffer size: " << train_MNISTImages.data.size() << endl;
    cout << "Label buffer size: " << train_MNISTLabels.data.size() << endl;
    cout << "Image buffer size: " << test_MNISTImages.data.size() << endl;
    cout << "Label buffer size: " << test_MNISTLabels.data.size() << endl;

    //初始化权重向量
    weights.resize(static_cast<std::size_t>(train_MNISTImages.cols) *
                   static_cast<std::size_t>(train_MNISTImages.rows));
    for(std::size_t i = 0; i < weights.size(); i++){
        weights[i] = (rand()/double(RAND_MAX) - 0.5) * 0.01;
    }

    prune_data(train_MNISTImages, train_MNISTLabels);
    train(train_MNISTImages, train_MNISTLabels,1000, 500, learnRate);
    test(test_MNISTImages, test_MNISTLabels);
    return 0;
}