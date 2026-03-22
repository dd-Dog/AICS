
/**
实现2层MLP

输入特征784个
隐藏层神经元个数 128-256个
输出层分类 10种

权重矩阵：隐藏层 784*128个，输出层128*10个
偏置值：  隐藏层 128个，输出层10个

权重梯度矩阵：隐藏层 784*128个，输出层 128*10个
偏置梯度：隐藏层128个，输出层10个

结构：
全连接->ReLU->全连接->softmax

正向传播：h1=x*w1+b1,ReLu(h1),z=h1*w2+b2=(x*w1+b1)*w2+b2,softmax(z),cross_entropy
反向传播：
输出层梯度：根据损失函数L对输出z求导，得delta_out=p-y，grad=delta_out*隐藏层输入
隐藏层梯度：根据损失函数对delta_hide=δh​=δout​⋅W2​⋅ReLU′(hpre​)=

分类及损失函数：softmax + cross_entropy; 该部分梯度计算后为delta_output=p-y
输出层的梯度：delta_w2=delta_output*(d(z)/d(w2))=delta_output*h1，这是w2权重的梯度，
             delta_b2=delta_output*(d(z)/d(b2))=delta_output*1,这是b2偏置值的梯度
到达隐藏层  d(L)/d(h1)=delta_output*w2^T,对w2的梯度的转置，这是由正向传播时的矩阵相乘决定的
隐藏层ReLU函数的导数,(ReLU)'=a1>0?1:0; a1是隐藏层输出
隐藏层梯度  d(L)/d(a1)=d(L)/d(h1)*(ReLU(a1))'  a1是隐藏层输入
隐藏层参数w1梯度 d(L)/d(w1)=d(L)/d(a1)*d(a1)/d(w1)=d(L)/d(h1)*(ReLU)'*x^T
隐藏层参数b1覆盖率 d(L)/d(b1)=d(L)/d(h1)*(ReLU)'
*/

#include <iostream>
#include <random>
#include <cmath>
#include <vector>

#include "../utils/mnist_loader.h"
#include "../utils/common.h"
using namespace MNISTUtils;
using namespace std;

#define DIM 28*28
#define HIDEN_UNIT_COUNT 128
#define CATEGORIES 10

//代码布局 行 = 输出维度，列 = 输入维度  W[out-行][in-列]
//vector 第一维列，第二维是行  vector(行-第二维，列-第一维)
vector<vector<double>> w1(HIDEN_UNIT_COUNT,vector<double>(DIM));
vector<vector<double>> w2(CATEGORIES,vector<double>(HIDEN_UNIT_COUNT));
vector<double> b1(128), b2(10);
double learnRate;



void martrix_multiply_layer1(std::vector<double>& data, int index, vector<double>& hide_input){
    int offset = index*DIM;
    for(std::size_t i=0; i<hide_input.size(); i++){
        hide_input[i] = 0.0;
        for(std::size_t j=0; j<DIM; j++){
            hide_input[i] += w1[i][j]*data[offset+j];
        }
        hide_input[i] += b1[i];
    }
}

void martrix_multiply_layer2(vector<double>& hide_output, vector<double>& z){
    for(std::size_t i=0; i<z.size(); i++){
        z[i] = 0.0;
        for(std::size_t j=0; j<hide_output.size(); j++){
            z[i] += hide_output[j]*w2[i][j];
        }
        z[i] += b2[i];
    }
}


void ReLU(vector<double>& hide_input, vector<double>& hide_output){
    
    for(int i=0; i<HIDEN_UNIT_COUNT; i++){
        hide_output[i] = hide_input[i]>0?hide_input[i]:0;
    }
}

void martrix_multiply_hide_grad(vector<double>& delta_output, vector<double> gradient_hide){
    for(std::size_t i=0; i<gradient_hide.size();i++){
        for(std::size_t j=0; j<delta_output.size(); j++){
            // cout << "i=" << i << ",j=" << j << endl;
            gradient_hide[i] += delta_output[j] + w2[j][i];
        }
    }
}

void ReLU_derivative(vector<double> gradient_hide, vector<double> hide_input){   
    for(std::size_t i=0; i<gradient_hide.size(); i++){
        gradient_hide[i] *= hide_input[i]>0?1:0;
    }
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

/**
多分类交叉熵公式 L=−i∑​yi​logpi​
例如y=3，则[0,0,0,1,0,0,0,0,0,0],公式展开得=−logp3​

为什么在加eps，防止
p[y] = 0
log(0) = -∞
loss = +∞ → NaN
 */
double cross_entropy(int y, vector<double> p){
    // double loss = 0.0;
    // for(std::size_t i=0; i<p.size(); i++){
    //     loss += -(y*log(p[i]));
    // }
    // return loss;
    const double eps = 1e-12;
    return -log(p[y] + eps);
}


void train(MNISTImages& train_images, MNISTLabels& train_labels, int train_size, int epoches){
    cout << "========== start training ========="<< endl;
    int epoch = 0;
    vector<double> hide_output(HIDEN_UNIT_COUNT,0.0);
    vector<double> hide_input(HIDEN_UNIT_COUNT,0.0);
    vector<double> z(CATEGORIES,0.0);
    vector<double> p(CATEGORIES,0.0);
    vector<double> delta_output(CATEGORIES,0.0);
    vector<vector<double>> gradient_w1(HIDEN_UNIT_COUNT,vector<double>(DIM,0.0));
    vector<vector<double>> gradient_w2(CATEGORIES,vector<double>(HIDEN_UNIT_COUNT,0.0));
    vector<double> gradient_b1(HIDEN_UNIT_COUNT, 0.0);
    vector<double> gradient_b2(CATEGORIES, 0.0);
    vector<double> gradient_hide(HIDEN_UNIT_COUNT,0.0);

    // printMatrix(w2);
    for(epoch=0; epoch<epoches; epoch++){
        // cout << "epoch " << epoch << endl;
        //======初始化临时变量=======
        double total_loss = 0.0;
        double loss = 0.0;
        //======初始化临时变量=======

        for(int i=0; i<train_size; i++){
            //每次训练梯度清零,使用SGD
            for (auto& row : gradient_w1) fill(row.begin(), row.end(), 0.0);
            for (auto& row : gradient_w2) fill(row.begin(), row.end(), 0.0);
            fill(gradient_b1.begin(), gradient_b1.end(), 0.0);
            fill(gradient_b2.begin(), gradient_b2.end(), 0.0);

            // cout << "i=" << i << endl;
            //1.计算隐藏层输入
            martrix_multiply_layer1(train_images.data_normalized,i,hide_input);
            // cout << "1.计算隐藏层输入" <<  endl;
            // printVector(hide_input);
            //2.ReLU激活
            ReLU(hide_input, hide_output);
            // printVector(hide_output);
            //3.计算隐藏输出
            martrix_multiply_layer2(hide_output, z);
            // printVector(z);
            //4.Softmax激活
            softmax(z, p);
            // cout << "y=" << static_cast <int>(train_labels.data[i]) << ",p[y]=" << p[train_labels.data[i]] << endl; 
            // printVector(p);
            //5.计算损失函数
            loss = cross_entropy(train_labels.data[i], p);
            // cout << "5.计算损失函数 loss=" << loss <<  endl;
            total_loss += loss;
            //6.计算w2及b2的梯度
                //计算delta_output
            for(std::size_t j=0;j<p.size();j++){
                delta_output[j] = p[j];
            }
            delta_output[train_labels.data[i]] -= 1.0;

            // cout << "6.1 计算delta_output" <<  endl;
                //计算w2梯度
            for(std::size_t k=0; k<w2.size(); k++){
                for(std::size_t j=0; j<w2[0].size(); j++){
                    gradient_w2[k][j] += delta_output[k] * hide_output[j];
                }
            }
            // cout << "6.2 计算w2梯度" <<  endl;
                //计算b2梯度
            for(std::size_t j=0; j<gradient_b2.size(); j++){
                gradient_b2[j] += delta_output[j]*1;
            }
            // cout << "6.3 计算b2梯度" <<  endl;
            
            //7.计算w1及b1的梯度
                //到达隐藏层的梯度
            martrix_multiply_hide_grad(delta_output, gradient_hide);
            // cout << "7.1 到达隐藏层的梯度" <<  endl;
                //乘以ReLU的导数
            ReLU_derivative(gradient_hide, hide_input);
            // cout << "7.2 乘以ReLU的导数" <<  endl;
                //w1的梯度           
            for(std::size_t k=0; k<gradient_w1.size(); k++){
                int offset = DIM*i;
                for(std::size_t j=0; j<gradient_w1[0].size(); j++){
                    //隐藏层梯度*x^T
                    gradient_w1[k][j] += gradient_hide[k]*train_images.data_normalized[offset+j];
                }
            }
            // cout << "7.3 w1的梯度" <<  endl;
                //b1的梯度
            for(std::size_t k=0; k<gradient_b1.size(); k++){
                gradient_b1[k] += gradient_hide[k];
            }
            // cout << "7.4 b1的梯度" <<  endl;
            //8.更新梯度
            for(std::size_t k=0; k<gradient_w1.size(); k++){
                for(std::size_t j = 0;j<gradient_w1[0].size(); j++){
                    w1[k][j] -= learnRate * gradient_w1[k][j];
                }
            }
            for(std::size_t k=0; k<gradient_w2.size(); k++){
                for(std::size_t j = 0;j<gradient_w2[0].size(); j++){
                    w2[k][j] -= learnRate * gradient_w2[k][j];
                }
            }
            for(std::size_t k=0; k<gradient_b1.size(); k++){
                b1[k] -= learnRate*gradient_b1[k];
            }
            for(std::size_t k=0; k<gradient_b2.size(); k++){
                b2[k] -= learnRate*gradient_b2[k];
            }
            // cout << "8.更新梯度" <<  endl;

        }
        total_loss /= train_size;
        cout << "epoch=" << epoch << ", total_loss=" << total_loss << endl;
    }
}


int main(){
    // 加载训练图像
    MNISTImages train_images = loadMNISTImages("..\\data\\raw\\MNIST\\raw\\train-images-idx3-ubyte");

    // 加载训练标签
    MNISTLabels train_labels = loadMNISTLabels("..\\data\\raw\\MNIST\\raw\\train-labels-idx1-ubyte");

    // 加载测试数据（可选，不打印详细信息）
    MNISTImages test_images = loadMNISTImages("..\\data\\raw\\MNIST\\raw\\t10k-images-idx3-ubyte", false);
    MNISTLabels test_labels = loadMNISTLabels("..\\data\\raw\\MNIST\\raw\\t10k-labels-idx1-ubyte", false);

    std::mt19937 rng(42);   //高质量伪随机数算法  seed 42

    //对于 128 输入维度，这是 非常大的初始化。
    int fan_in = 784;
    double limit = sqrt(6.0 / fan_in);
    std::uniform_real_distribution<double> dist(-limit, limit);

    for (auto& row : w1)
        for (auto& v : row)
            v = dist(rng);

    int fan_in_2 = 784;
    double limit2 = sqrt(6.0 / fan_in_2);
    std::uniform_real_distribution<double> dist2(-limit2, limit2);
    for (auto& row : w2)
        for (auto& v : row)
            v = dist2(rng);

    for (auto& v : b1)
        v = dist(rng);
    for (auto& v : b2)
        v = dist(rng);

    learnRate = 0.02;



    train(train_images, train_labels, 1000, 100);

    return 0;
}