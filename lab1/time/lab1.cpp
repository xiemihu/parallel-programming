#include <iostream>
#include <windows.h>
#include <stdlib.h>
using namespace std;

const int N = 3000;

void plain(int n, double* b, double* a, double* sum) {
    for (int i = 0; i < n; i++) {
        sum[i] = 0.0;
        for (int j = 0; j < n; j++)
            sum[i] += b[j * n + i] * a[j];
    }
}

void cache(int n, double* b, double* a, double* sum) {
    for (int i = 0; i < n; i++)
        sum[i] = 0.0;
    
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            sum[i] += b[j * n + i] * a[j];
}

void test_plain(double* b, double* a, double* sum){
    int step = 10, n = 10;
    long long head, tail, current, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    long counter = 0;
    for(; n<=N; n += step){
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        counter = 0;       
        do {
            counter++;
            plain(n, b, a, sum);
            QueryPerformanceCounter((LARGE_INTEGER*)&current);
        } while ((current - head) < (freq / 5)) ;
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double total_time = (tail - head) * 1000.0 / freq;
        double time = total_time / counter;
        cout << n << "\t" << counter << "\t" << total_time << " ms" << "\t" << time << " ms" << endl;
 
        if(n == 100) step = 100;
        if(n == 1000) step = 500;
    }
}

void test_cache(double* b, double* a, double* sum){
    int step = 10, n = 10;
    long long head, tail, current, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    long counter = 0;
    for(; n<=N; n += step){
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        counter = 0;
        do {
            counter++;
            cache(n, b, a, sum);
            QueryPerformanceCounter((LARGE_INTEGER*)&current);
        } while ((current - head) < (freq / 5)) ;
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double total_time = (tail - head) * 1000.0 / freq;
        double time = total_time / counter;
        cout << n << "\t" << counter << "\t" << total_time << " ms" << "\t" << time << " ms" << endl;
        if(n == 100) step = 100;
        if(n == 1000) step = 500;
    }

}

int main() {
    
    double* b = new double[N * N];
    double* a = new double[N];
    double* sum = new double[N];
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0;
        for (int j = 0; j < N; j++)
            b[i * N + j] = i + j;
    }
    cout<<"平凡算法："<<endl;
    test_plain(b, a, sum);
    cout<<"优化算法："<<endl;
    test_cache(b, a, sum);
    return 0;
}