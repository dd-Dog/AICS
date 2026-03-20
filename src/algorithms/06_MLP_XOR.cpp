#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;


/**
XOR 即异或运算
线性无法完成，本例意在证明使用两层的神经网络即可实现XOR运算

网络结构，两层全连接
       H1
X1           
       H2         
                Y
       H3
X2
       H4

输入层到隐藏层---全连接---ReLU激活--->全连接--->输出层--->Sigmoid激活--->计算损失，梯度--->反向传播
MLP
ReLU
MLP
Sigmoid
BCE       
*/

vector<vector<int>> train_data = {
    {0,0},
    {0,1},
    {1,0},
    {1,1}
};
vector<int> train_tag={0,1,1,0};

vector<vector<double>> test_data = {
    {0,0},
    {0,1},
    {1,0},
    {1,1},

    {0.01,0.02},
    {0.02,0.98},
    {0.98,0.02},
    {0.99,0.99},

    {0.25,0.75},
    {0.75,0.25},
    {0.3,0.3},
    {0.7,0.7}
};
vector<int> test_label = {
    0,1,1,0,
    0,1,1,0,
    1,1,0,0
};

#define HIDEN_LAYER 4
//权重参数，偏置值,隐藏层定义4个参数

//输出层只有一个值，因此输出层偏置值有一个
double b2,learnRate;

//隐藏层的偏置，有四个输出就需要四个偏置值
vector<double> b1(4,0.0);
//输入层到隐藏层的权重矩阵，4X2
vector<vector<double>> w1(4,vector<double>(2,0.0)),w2(4,vector<double>(1,0.0));

vector<double> multiply_martrix(vector<double>& input, const vector<vector<double>>& w, vector<double> b){
    vector<double> output(4);
    for(std::size_t i=0; i<w.size(); i++){
        output[i] = input[0]*w[i][0] + input[1]*w[i][1] + b[i];
    }
    return output;
}

double ReLU(double z){
    return z<=0? 0 : z;
}

double tanh_act(double z) {
    return tanh(z);
}

/**
二分类交叉熵损失函数，用于计算损失
    输入标签值，和预测概率，配合sigmoid激活函数使用
*/
double binary_cross_entropy_loss(double y, double p){
    const double eps = 1e-12;
    p = max(eps, min(p, 1-eps));
    return -(y * log(p) + (1-y) * log(1-p));
}

double sigmoid(double z){
    // return 1/(exp(-y) + 1);
    if (z >= 0)
        return 1.0 / (1.0 + exp(-z));
    else {
        double e = exp(z);
        return e / (1.0 + e);
    }
}

void train(const vector<vector<int>>& train_data, const vector<int>& train_tag, int epoches){

    //轮次训练
    int epoch;
    int train_size = train_data.size();
    for(epoch=0; epoch < epoches; epoch++){
        
        double total_loss = 0.0, loss = 0.0;

        vector<vector<double>> gradient_w1(4,vector<double>(2,0.0)), gradient_w2(4,vector<double>(1,0.0));
        vector<double> gradient_b1(4, 0.0);
        double gradient_b2=0.0;
        for(int i=0; i<train_size; i++){
            //计算预测值 Y=W2*(W1*X+B1)+B2，
            //输入层到隐藏层全连接
            vector<double> input(2);
            input[0] = static_cast<double>(train_data[i][0]);
            input[1] = static_cast<double>(train_data[i][1]);
            vector<double> hiden_input = multiply_martrix(input, w1, b1);

            //ReLU激活,
            for(int i=0; i<4; i++){
                hiden_input[i] = tanh_act(hiden_input[i]);
                // cout << hiden_input[i] << " ";
            }
            // cout << endl;
            //隐藏层到输出层
            double y_hat = 0.0;
            for(int k = 0; k < 4; k++){
                y_hat += hiden_input[k] * w2[k][0];
            }
            y_hat += b2; 

            //Sigmoid激活函数 
            double p = sigmoid(y_hat);

            //损失函数，二分类交叉熵
            loss = binary_cross_entropy_loss(train_tag[i], p);
            total_loss += loss;

            //计算输出层误差，根据损失函数L对输出z求导，得delta=p-y
            double delta_out = p-train_tag[i];

            //计算输出层w2和b2的梯度
            for(std::size_t k=0; k<w2.size(); k++){
                for(std::size_t l=0; l<w2[0].size(); l++){
                    gradient_w2[k][l] += delta_out*hiden_input[k];
                }
                
            }
            gradient_b2 += delta_out*1;


            //计算隐藏层w1和b1的梯度
            for(std::size_t k=0; k<w1.size(); k++){
                //计算隐藏层误差，根据损失函数链式求导得：delta_hide=δh​=δout​⋅W2​⋅ReLU′(hpre​)
                //即delta_hide=输出层误差(delta_out)*输出层权重w2*ReLU'(输入层输入)
                //注意最后乘的是ReLU的导数
                // double relu_grad = hiden_input[k]>0? 1.0: 0.0;
                double tanh_grad = 1-hiden_input[k]*hiden_input[k];
                double delta_hide = delta_out*w2[k][0]*tanh_grad;
                for(std::size_t l=0; l<w1[0].size(); l++){
                    gradient_w1[k][l] += delta_hide*input[l];
                }  
                gradient_b1[k] += delta_hide*1;             
            }           
        }

        //更新权重参数
        for(std::size_t k=0; k<w2.size(); k++){
            for(std::size_t l=0; l<w2[0].size(); l++){
                w2[k][l] -= learnRate*gradient_w2[k][l] / train_size;
            }           
        }
        
        b2 -= learnRate*gradient_b2/train_size;

        //计算w1和b1的梯度
        for(std::size_t k=0; k<w1.size(); k++){
            for(std::size_t l=0; l<w1[0].size(); l++){
                w1[k][l] -= learnRate*gradient_w1[k][l] / train_size;
            }               
        }
        for(int k=0; k<4; k++){
            b1[k] -= learnRate*gradient_b1[k]/train_size;
        }
        

        total_loss /= train_size;
        cout << "epoch=" << epoch << ", total_loss=" << total_loss << endl;
    }
}

void test(const vector<vector<double>>& test_data, const vector<int>& test_tag){

    vector<double> input(2);
    int correct = 0;
    vector<double> hiden_input(4);
    cout << "testsize=" << test_data.size() << endl;
    for(std::size_t i=0; i<test_data.size(); i++){
        // 修复：这里应该使用 test_data，不然 i>=4 会访问越界导致程序中止
        input[0] = test_data[i][0];
        input[1] = test_data[i][1];
        hiden_input = multiply_martrix(input, w1, b1);
        for(int j=0;j<hiden_input.size(); j++){
            hiden_input[j] = tanh(hiden_input[j]);
        }
        double y_hat=0.0;
        for(int j=0; j<hiden_input.size();j++){
            y_hat += hiden_input[j]*w2[j][0];
        }
        y_hat += b2;
        double p = sigmoid(y_hat);
        int result = 0;
        if(p>0.5){
            result = 1;
        }else {
            result = 0;
        }
        if(result == test_tag[i]) correct ++;
        cout << input[0] << "," << input[1] << " → p=" << p << endl;
    }
    cout << "test accuracy=" << correct*100.0/test_data.size() <<"%"<< endl;

}


int main(){

    //初始化参数,权重参数必须随机初始化，如果全部为0 
    //权重全部初始化为 0 → 对称性 + 梯度为 0 → 隐藏层无法学习
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (auto& row : w1)
        for (auto& v : row)
            v = dist(rng);

    for (auto& row : w2)
        for (auto& v : row)
            v = dist(rng);
    for(std::size_t i=0; i<b1.size(); i++){
        b1[i] = 0.1;
    }
    b2 = 0.1;
    learnRate = 0.1;
    
    train(train_data, train_tag, 1000);
    test(test_data, test_label);
}