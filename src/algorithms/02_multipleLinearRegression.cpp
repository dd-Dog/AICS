#include <iostream>
#include <vector>
#include <cmath>
#include <random>


#define FEATURE_SIZE 3
#define TRAIN_SIZE 14
#define TEST_SIZE 5


double learnRate = 0.01;
double w[FEATURE_SIZE], b;  //y = w1*x1 + w2*x2 + w3*x3 + b

/**
 * 训练数据
 房价 ≈ 1.8×面积 + 8×房间数 − 1.2×房龄 + 5
 x1: 面积, x2: 房间数, x3: 房龄, y: 房价

 三个特征值数值差异过大，需要做标准化，否则会产生“数值发散”问题
 epoch: 0, total_loss: 68921.3, w1: 64.1529, w2: 2.12729, w3: 2.03057, b: 0.488571
 如三个权重的差异过大，每步过都跨过最低谷，再被梯度拉回，越来越远

 而一元线性回归没有这个问题，因为特征值只有一个，没有尺度差异
 */
double train_mean[FEATURE_SIZE], train_std[FEATURE_SIZE];  //训练数据的均值和标准差,以预测时要使用
double train_data[][4] = {
    {50, 1, 20, 92},
    {60, 2, 18, 110},
    {75, 2, 10, 145},
    {85, 3, 12, 165},
    {90, 3, 8, 178},
    {100, 3, 5, 205},
    {110, 4, 6, 230},
    {120, 4, 4, 255},
    {130, 4, 3, 275},
    {140, 5, 2, 305},
    {150, 5, 1, 330},
    {160, 5, 1, 350},
    {170, 6, 0, 380},
    {180, 6, 0, 400}
};

/**
 * 测试数据
 */
double test_data[][4] = {
    {65, 2, 15, 125},
    {95, 3, 7, 190},
    {125, 4, 3, 265},
    {155, 5, 1, 340},
    {175, 6, 0, 390}
};

void init(){
    for(int i=0;i<FEATURE_SIZE;i++){
        w[i] = 1;
    }
    b = 2;
}

/**
 * Z-score标准化，将特征值缩放到均值为0，标准差为1的范围内  
 公式：(x - mean) / std
 其中，x为特征值，mean为均值，std为标准差
 标准差：std = sqrt(sum((x - mean)^2) / n)
 均值：mean = sum(x) / n
 其中，n为特征值的个数
 */
void z_score_normalize_train(double train_data[][4], int train_size){

    for(int i=0; i<FEATURE_SIZE; i++){
        double sum = 0;
        for(int j=0; j<train_size; j++){
            sum += train_data[j][i];
        }
        train_mean[i] = sum / train_size;
        double sum_std = 0;
        for(int j=0; j<train_size; j++){
            sum_std += pow(train_data[j][i] - train_mean[i], 2);
        }
        train_std[i] = sqrt(sum_std / train_size);
        for(int j=0; j<train_size; j++){
            train_data[j][i] = (train_data[j][i] - train_mean[i]) / train_std[i];
        }
    }
}
/*
 * 标准化测试数据，使用训练数据的均值和标准差
 公式：(x - mean) / std
 其中，均值和标准差为训练数据的均值和标准差
 */
void z_score_normalize_test(double test_data[][4], int test_size){
    for(int i=0; i<FEATURE_SIZE; i++){  
        for(int j=0; j<test_size; j++){
            test_data[j][i] = (test_data[j][i] - train_mean[i]) / train_std[i];
        }
    }
}

void train(double train_data[][4], int train_size, int epochs, float learnRate){
    z_score_normalize_train(train_data, train_size);
    for(int i=0;i<epochs;i++){
        double total_loss = 0;
        double gradient_w1 = 0, gradient_w2 = 0, gradient_w3 = 0, gradient_b = 0;
        for(int j=0;j<train_size;j++){
            double x1 = train_data[j][0];
            double x2 = train_data[j][1];
            double x3 = train_data[j][2];
            double y = train_data[j][3];
            double y_pred = w[0]*x1 + w[1]*x2 + w[2]*x3 + b;
            //计算损失
            total_loss += pow(y_pred - y, 2);
            //计算累计梯度
            gradient_w1 += (y_pred - y) * x1;
            gradient_w2 += (y_pred - y) * x2;
            gradient_w3 += (y_pred - y) * x3;
            gradient_b += y_pred - y;

            
        }
        //计算平均损失
        total_loss /= train_size;
        //反向传播，更新权重
        w[0] -= 2 * learnRate * gradient_w1 / train_size;
        w[1] -= 2 * learnRate * gradient_w2 / train_size;
        w[2] -= 2 * learnRate * gradient_w3 / train_size;
        b -= 2 * learnRate * gradient_b / train_size;

        std::cout << "epoch: " << i << ", total_loss: " << total_loss << ", w1: " << w[0] << ", w2: " << w[1] << ", w3: " << w[2] << ", b: " << b << std::endl;

    }
}

void test(double test_data[][4], int test_size){
    z_score_normalize_test(test_data, test_size);
    double total_loss = 0;
    for(int i=0; i<test_size; i++){
        double x1 = test_data[i][0];
        double x2 = test_data[i][1];
        double x3 = test_data[i][2];
        double y = test_data[i][3];
        double y_pred = w[0]*x1 + w[1]*x2 + w[2]*x3 + b;
        total_loss += pow(y_pred - y, 2);

        std::cout << "test_data[" << i << "]: x1: " << x1 << ", x2: " << x2 << ", x3: " << x3 << ", y: " << y << ", y_pred: " << y_pred << ", loss: " << pow(y_pred - y, 2) << std::endl;

    }
}

int main(){
    train(train_data, TRAIN_SIZE, 10000, learnRate);
    test(test_data, TEST_SIZE);
    return 0;
}