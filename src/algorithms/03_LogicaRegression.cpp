#include <iostream>
#include <cmath>

using namespace std;

/**
逻辑回归，二分类问题
相对于线性回归，增加了sigmoid激活和概率解释
*/

//初始化权重和偏置值
double w1=1, w2 = 1, b=1,learnRate=0.04;

double train_data[][3]={
    {2, 3, 1},
    {3, 3, 1},
    {4, 5, 1},
    {5, 6, 1},
    {1, 1, 0},
    {2, 1, 0},
    {3, 2, 0},
    {1, 2, 0}
};

double test_data[][3]={
    {3, 4, 1},
    {2 ,2, 0},
    {0, 0, 0},
    {6, 6, 1},
    {1, 0, 0},
    {4, 3, 1}
};


//激活函数,增加负数保护，当y为负值且很大时，exp(-z) 可能 overflow,是一个巨大的正数，导致overflow
double sigmoid(double z){
    // return 1/(exp(-y) + 1);
    if (z >= 0)
        return 1.0 / (1.0 + exp(-z));
    else {
        double e = exp(z);
        return e / (1.0 + e);
    }
}
//交叉熵损失函数:y_i·log(p_i) + (1-y_i)·log(1-p_i)
//增加epsilon裁剪，如果p=0或1，会溢出
double cross_entropy_loss(double y, double p){
    const double eps = 1e-12;
    p = max(eps, min(1-eps, p)); //相当于前去了接近0和非常接近1的部分
    return -(y*log(p) + (1-y)*log(1-p));
}

void train(double train_data[][3], int train_size, double learnRate, int epochs){

    for(int epoch=0; epoch<epochs; epoch++){
        double total_loss = 0;
        double gradient_w1 = 0;
        double gradient_w2 = 0;
        double gradient_b = 0;
        for(int i=0; i<train_size; i++){
            //取输入特征
            double x1 = train_data[i][0];
            double x2 = train_data[i][1];
            //预测输出
            double z = w1*x1 + w2*x2 + b;
            double y = train_data[i][2];
            //sigmoid激活,得出概率
            double p = sigmoid(z);
            
            
            //计算损失，交叉熵
            total_loss += cross_entropy_loss(y, p);

            //计算梯度，并累加 ∂L/∂w = (1/n) Σ (p - y)·x
            gradient_w1 += (p-y)*x1;
            gradient_w2 += (p-y)*x2;
            //∂L/∂b = (1/n) Σ (p - y)
            gradient_b += p-y;
            // cout << "gradient_w1=" << gradient_w1 << ",gradient_w2" << gradient_w1 << ",gradient_b=" << gradient_w1 << endl;
        }
        //平均损失
        total_loss /= train_size;
        // cout << learnRate * gradient_w1/train_size << endl;
        //更新权重
        w1 -= learnRate * gradient_w1/train_size;
        w2 -= learnRate * gradient_w2/train_size;
        b -=  learnRate * gradient_b/train_size;

        //打印信息
        cout << "total_loss:" << total_loss << ", w1=" << w1 << ",w2=" << w2 << ", b=" << b << endl;
    }
}


void test(double test_data[][3], int test_size){

    for(int i=0; i<test_size; i++){
        double x1 = test_data[i][0];
        double x2 = test_data[i][1];
        double z = w1*x1 + w2*x2 + b;

        double p = sigmoid(z);
        int y_hat = (p >= 0.5);
        double loss = cross_entropy_loss(test_data[i][2], p);
        cout << "x1=" << x1 << ", x2=" << x2 << ", z=" << z << ", y_hat=" << y_hat << ", loss=" << loss << endl;
    }


}


int main(){

    train(train_data, 8, learnRate, 10000);

    test(test_data, 6);

    return 0;
}