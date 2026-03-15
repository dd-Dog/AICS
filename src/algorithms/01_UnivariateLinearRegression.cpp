#include <iostream>
#include <cmath>

using namespace std;


/**
 * 一元线性回归
 * @param w 权重
 * @param b 偏置
 * @param learnRate 学习率
 * @param train_data 训练数据
 * @param test_data 测试数据
 */
double w=1, b=1,learnRate=0.001;


double train_data[][2] = {
    {0, 3.1},
    {1, 5.6},
    {2, 8.0},
    {3, 10.7},
    {4, 12.9},
    {5, 15.6},
    {6, 17.8},
    {7, 20.4},
    {8, 22.9},
    {9, 25.3},
    {10, 27.9},
    {12, 32.7},
    {14, 37.8},
    {16, 43.2},
    {18, 48.1}
};

double test_data[][2] = {
    {11, 30.5},
    {13, 35.4},
    {15, 40.5},
    {17, 45.6},
    {20, 52.8}
};

/**
正确答案：
w ≈ 2.5
b ≈ 2
*/

/**
 * 损失函数
 * @param y 真实值
 * @param y_pred 预测值
 * @return 损失值
 */
 double loss_function(double y, double y_pred){
    return pow(y - y_pred,2);
}

/**
 * 梯度下降
 * @param x 输入值
 * @param y 真实值
 * @param y_pred 预测值
 * @return 梯度
 */
double gradient_descent(double gradient_comulative, double learnRate, int train_size){
    return 2*(gradient_comulative/train_size) * learnRate;
}

 
/**
 * 训练
 * @param train_data 训练数据
 * @param train_size 训练数据大小
 * @param learnRate 学习率
 * @param rounds 轮数
 */
void train(double train_data[][2], int train_size, double learnRate, int rounds){
    //训练轮数
    for(int round=0;round<rounds;round++){
        double total_loss = 0;
        double gradient_w = 0;
        double gradient_b = 0;
        //遍历训练数据集
        for(int i=0;i<train_size;i++){
            double x = train_data[i][0];
            double y = train_data[i][1];
            //预测值
            double y_pred = w*x+b;
            //损失值
            total_loss += loss_function(y, y_pred);
            //梯度
            gradient_w += (y_pred - y)*x;
            gradient_b += y_pred - y;
        }
        total_loss /= train_size;
        w -= gradient_descent(gradient_w, learnRate, train_size);
        b -= gradient_descent(gradient_b, learnRate, train_size);
        printf("w: %f, b: %f, total_loss: %f\n", w, b, total_loss);
    }
}


void test(double test_data[][2], int test_size){
    for(int i=0;i<test_size;i++){
        double x = test_data[i][0];
        double y = test_data[i][1];
        double y_pred = w*x + b;
        cout << "x: " << x << " y: " << y << " y_pred: " << y_pred << endl;
        cout << "loss: " << loss_function(y, y_pred) << endl;
    }
}

int main(){

    train(train_data, sizeof(train_data)/sizeof(train_data[0]), learnRate, 1000);
    test(test_data, sizeof(test_data)/sizeof(test_data[0]));
    return 0;
}