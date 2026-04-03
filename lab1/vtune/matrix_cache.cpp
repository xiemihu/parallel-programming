#include <iostream>
#include <windows.h>
#include <stdlib.h>
using namespace std;

// const int N = 3000;

// void cache(int n, double* b, double* a, double* sum) {
//     for (int i = 0; i < n; i++)
//         sum[i] = 0.0;
    
//     for (int j = 0; j < n; j++)
//         for (int i = 0; i < n; i++)
//             sum[i] += b[j * n + i] * a[j];
// }

// void test_cache(double* b, double* a, double* sum){
//     int step = 10, n = 10;
//     long long head, tail, current, freq;
//     QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
//     long counter = 0;
//     for(; n<=N; n += step){
//         QueryPerformanceCounter((LARGE_INTEGER*)&head);
//         counter = 0;
//         do {
//             counter++;
//             cache(n, b, a, sum);
//             QueryPerformanceCounter((LARGE_INTEGER*)&current);
//         } while ((current - head) < (freq / 5)) ;
//         QueryPerformanceCounter((LARGE_INTEGER*)&tail);
//         double time = (tail - head) * 1000.0 / freq / counter;
//         cout << n << "\t" << counter << "\t" << time << " ms" << endl;
//         if(n == 100) step = 100;
//     }

// }

int main() {
    const int N = 3000;
    double* b = new double[N * N];
    double* a = new double[N];
    double* sum = new double[N];
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0;
        for (int j = 0; j < N; j++)
            b[i * N + j] = i + j;
    }
    cout<<"优化算法："<<endl;
    for (int i = 0; i < N; i++)
        sum[i] = 0.0;
    
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            sum[i] += b[j * N + i] * a[j];
    return 0;
}