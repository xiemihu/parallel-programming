#include <iostream>
#include <windows.h>
#include <stdlib.h>
using namespace std;

// const int N = 33554432;

// void cache(int n, double* a, double& sum) {
//     sum = 0.0;
//     double sum1 = 0.0;
//     double sum2 = 0.0;
//     for(int i=0;i<n;i+=2){
//         sum1 += a[i];
//         sum2 += a[i+1];
//     }
//     sum = sum1 + sum2;
//     // cout << sum << " ";
// }

// void test_cache(double* a, double sum){
//     int step = 2, n = 1;
//     long long head, tail, current, freq;
//     QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
//     long counter = 0;
//     // double* temp = new double[N];
//     for(; n<=N; n *= step){
//         QueryPerformanceCounter((LARGE_INTEGER*)&head);
//         counter = 0;
//         do {
//             counter++;
//             cache(n, a, sum);
//             QueryPerformanceCounter((LARGE_INTEGER*)&current);
//         } while ((current - head) < (freq / 5)) ;
//         QueryPerformanceCounter((LARGE_INTEGER*)&tail);
//         double time = (tail - head) * 1000.0 / freq / counter;
//         cout << n << "\t" << counter << "\t" << time << " ms" << endl;
//     }

// }

int main() {
    const int N = 33554432;
    double* a = new double[N];
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0;
    }
    cout<<"优化算法："<<endl;
    sum = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    for(int i=0;i<N;i+=2){
        sum1 += a[i];
        sum2 += a[i+1];
    }
    sum = sum1 + sum2;
    cout << sum;
    delete[] a;
    return 0;
}