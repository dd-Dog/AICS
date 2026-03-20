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

vector<vector<double>> weights;
vector<double> biases;
double learnRate=0.1;

double nomalization(unsigned char x){
    return x/127.5 - 1.0;
}

/*
向量的点乘运算
*/
vector<double> dot_product(const MNISTImages &images, int x, int dim){
    vector<double> z(10, 0.0);
    int offset = x*dim;
    for(int k=0; k<=9; k++){
        for(int i=0; i<dim; i++){
            z[k] += nomalization(images.data[offset + i]) * weights[k][i];
        }
        z[k] += biases[k];
    }

    return z;
}

/**
softmas适合于单标签多分类，即分类结果是互斥的关系
sigmoid多输出适合多标签分类，分类结果互相无约束
p(i)=e^z(i)/sum(e^zj)

训练时softmax与CE合并，避免数值不稳定

grad是梯度，t是真实值
*/
double softmax_entropy(const vector<double>& z, vector<double>& grad, int t){
    double sum = 0;
    int n = z.size();
    double max_z = z[0];

    //找最大值，然后整体平移，避免数字过大溢出，但是不影响概率分布
    for(int i=0; i<n; i++){
        max_z = std::max(z[i], max_z);
    }
    for(int i=0; i<n; i++){
        grad[i] = exp(z[i]-max_z);   //平移
        sum += grad[i];
    }
    
    //归一化
    for(int i=0; i<n; i++){
        grad[i] = grad[i]/sum;
    }

    //Softmax+CE的梯度 p-y，y的概率为1.0
    grad[t] -= 1.0;

    double log_sum_exp = max_z + log(sum); //把平移加回来
    double loss = log_sum_exp - z[t];

    return loss;
}

void softmax(const vector<double>& z, vector<double>& grad){
    double sum = 0;
    int n = z.size();
    double max_z = z[0];

    //找最大值，然后整体平移，避免数字过大溢出，但是不影响概率分布
    for(int i=0; i<n; i++){
        max_z = std::max(z[i], max_z);
    }
    for(int i=0; i<n; i++){
        grad[i] = exp(z[i]-max_z);   //平移
        sum += grad[i];
    }
    
    //归一化
    for(int i=0; i<n; i++){
        grad[i] = grad[i]/sum;
    }
}

void train(const MNISTImages &train_images, const MNISTLabels &train_labels,int size, int epochs, double learnRate, int category){
    int dim = 784;
    double total_loss = 0.0;
    int epoch=0;
    for(epoch=0; epoch<epochs; epoch++){
        // cout << "epoch=" << epoch << endl;
        
        //w梯度
        vector<vector<double>> gradient_w(category, vector<double>(dim, 0.0));
        //z梯度
        vector<double> gradient_z(category, 0.0);
        vector<double> gradient_b(category,0.0);
        int train_size = min(train_images.count, size);
        for(int i = 0; i < train_size; i++){
            // cout << "i=" << i << endl;
            //取特征值并和权重向量运算，28x28=784个
            vector<double> z = dot_product(train_images, i, dim);
            //激活函数+多分类交叉熵
            double loss = softmax_entropy(z, gradient_z, train_labels.data[i]);
            total_loss += loss;
           
            
            //梯度累积
            int offset = i * dim; 
            for(int k=0; k<category; k++){
                for(int j=0; j<dim; j++){
                    double x = nomalization(train_images.data[offset + j]);
                    gradient_w[k][j] += gradient_z[k]*x;
                    // if(g<10) cout << gradient_w[g] << ", ";
                }
            }
            
            // cout << endl;
            for(int j=0; j<category; j++)
                gradient_b[j] += gradient_z[j];
        }
        //平均损失
        total_loss /= train_size;
        //更新梯度
        for(int k=0; k<category; k++){
            for(int i=0; i<dim; i++){
                weights[k][i] -= learnRate * gradient_w[k][i]/train_size;
            }
            // 
        }
        for(int k=0; k<category; k++)
            biases[k] -= learnRate * gradient_b[k]/train_size;
        
    }
    cout << "epoch=" << epoch << ", total_loss=" << total_loss << endl;
}


void test(const MNISTImages &test_images, const MNISTLabels &test_labels,  int category){
    int dim = 784;
    double loss = 0.0, total_loss = 0.0;
    int correct = 0;

    vector<double> z(10);
    vector<double> p(category, 0.0);

    for(int i=0; i<test_images.count; i++){
        z = dot_product(test_images, i, dim);
        softmax(z, p);
        total_loss += loss;

        int max_index = 0;
        for(int i=0; i<category; i++){
            if(p[max_index] < p[i]){
                max_index = i;
            }
        }
        if(max_index == test_labels.data[i]){
            correct ++;
        }
        // printf("z_pred=%d  z_real=%d\n", max_index, test_labels.data[i]);
    }
    printf("correct accuracy=%f\n", 100.0 * correct / test_images.count);
}

int main(){
    namespace fs = std::filesystem;

    int category = 10;
    int dim = 784;

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

    //初始化权重向量
    weights.resize(category, vector<double>(static_cast<int>(train_MNISTImages.cols) * static_cast<int>(train_MNISTImages.rows)));
    biases.resize(category);
    for(int k=0; k<category; k++){
        for(int i = 0; i < dim; i++){
            weights[k][i] = (rand()/double(RAND_MAX) - 0.5) * 0.01;
        }
    }

    for(std::size_t i=0; i<biases.size(); i++){
        biases[i] = 0.1;
    }

    train(train_MNISTImages, train_MNISTLabels, 20000, 5000, learnRate, 10);
    test(test_MNISTImages, test_MNISTLabels, category);
    return 0;
}