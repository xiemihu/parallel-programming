#include <iostream>
#include <windows.h>
#include <stdlib.h>
using namespace std;
//二重循环算法

int main() {
    const int N = 33554432;
    double* a = new double[N];
    // double sum = 0.0;
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0;
    }
    cout<<"平凡算法："<<endl;
    // test_plain(a, sum);
    // sum = 0.0;
    for (int m = N; m > 1; m /= 2) // log(n)个步骤
        for (int i = 0; i < m / 2; i++)
            a[i] = a[i * 2] + a[i * 2 + 1];
    cout << a[0];
    delete[] a;
    return 0;
}

