#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <iomanip>
using namespace std;

const int N = 33554432;

void plain(int n, double* a, double& sum) {
    sum = 0.0;
    for(int i = 0; i < n; i++)
        sum += a[i];
    // cout << sum << " ";
}

// void cache(int n, double* a, double* temp, double sum) {
//     if (n == 1) {
//         sum += a[0];
//         return;
//     }
//     int m = n;
//     for (int i = 0; i < m / 2; i++) {
//         temp[i] = a[i * 2] + a[i * 2 + 1];
//     }
//     for (m = m / 2; m > 1; m /= 2) { 
//         for (int i = 0; i < m / 2; i++) {
//             temp[i] = temp[i * 2] + temp[i * 2 + 1]; 
//         }
//     }
//     sum = temp[0];

// }

void cache(int n, double* a, double& sum) {
    sum = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    for(int i=0;i < n;i+=2){
        sum1 += a[i];
        sum2 += a[i+1];
    }
    sum = sum1 + sum2;
    // cout << sum << " ";
}

void test_plain(double* a, double sum){
    int step = 2, n = 2;
    long long head, tail, current, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    long counter = 0;
    for(; n<=N; n *= step){
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        counter = 0;       
        do {
            counter++;
            plain(n, a, sum);
            QueryPerformanceCounter((LARGE_INTEGER*)&current);
        } while ((current - head) < (freq / 5)) ;
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double total_time = (tail - head) * 1000.0 / freq;
        double time = total_time / counter;
        // cout << n << "\t" << counter << "\t" << total_time << " ms" << "\t" << time << " ms" << endl;
        cout << n << "\t" << counter << "\t" << time << " ms"<<" "<< fixed << setprecision(10) << sum << endl;
    }
}

void test_cache(double* a, double sum){
    int step = 2, n = 2;
    long long head, tail, current, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    long counter = 0;
    // double* temp = new double[N];
    for(; n<=N; n *= step){
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        counter = 0;
        do {
            counter++;
            cache(n, a, sum);
            QueryPerformanceCounter((LARGE_INTEGER*)&current);
        } while ((current - head) < (freq / 5)) ;
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double total_time = (tail - head) * 1000.0 / freq;
        double time = total_time / counter;
        // cout << n << "\t" << counter << "\t" << total_time << " ms" << "\t" << time << " ms" << endl;
        cout << n << "\t" << counter << "\t" << time << " ms" <<" "<< fixed << setprecision(10) << sum << endl;
    }

}

int main() {
    
    double* a = new double[N];
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        a[i] = (i+1) * 1.0;
    }
    cout<<"平凡算法："<<endl;
    test_plain(a, sum);
    cout<<"优化算法："<<endl;
    test_cache(a, sum);
    delete[] a;
    return 0;
}