#include <vector>
#include <iostream>

using namespace std;

//打印一维数组
void printVector(vector<double>& v){
    for(auto& x : v)
        cout << x << " ";
    cout << endl;
}

void printVector(vector<int>& v){
    for(auto& x : v)
        cout << x << " ";
    cout << endl;
}

void printVector(vector<unsigned char>& v){
    for(auto& x : v)
        cout << x << " ";
    cout << endl;
}


//打印二维数组（矩阵）
void printMatrix(vector<vector<double>> martrix){
    for(auto& v1 : martrix){
        for(auto& v2 : v1)
            cout << v2 << " ";
        cout << endl;
    }
        
}
