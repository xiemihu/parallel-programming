#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <mpi.h>
#include <omp.h>
#include <iomanip>
#include <unordered_set>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2
// mpic++ correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o main -O2
// mpic++ correctness_guess.cpp train.cpp guessing.cpp md5.cpp -O2 -fopenmp -o main
// qsub qsub_mpi.sh
int mpi_rank = 0;
int mpi_size = 1;

long long hashLocalGuesses(const vector<string> &guesses,
                           const unordered_set<string> &test_set,
                           double &local_hash_time)
{
    auto start_hash = system_clock::now();

    long long local_cracked = 0;
    int pw_count = guesses.size();
    int pad_count = pw_count % 4;
    int hash_count = pw_count - pad_count;

#pragma omp parallel reduction(+:local_cracked)
{
    bit32 state_simd[4][4];
    #pragma omp for schedule(static)
        for (int i = 0; i < hash_count; i += 4)
        {
            MD5Hash_NEON(&guesses[i], state_simd);

            if (test_set.find(guesses[i]) != test_set.end())
            {
                local_cracked += 1;
            }
            if (test_set.find(guesses[i + 1]) != test_set.end())
            {
                local_cracked += 1;
            }
            if (test_set.find(guesses[i + 2]) != test_set.end())
            {
                local_cracked += 1;
            }
            if (test_set.find(guesses[i + 3]) != test_set.end())
            {
                local_cracked += 1;
            }
        }
}

    bit32 state[4];
    for (int i = hash_count; i < pw_count; i += 1)
    {
        MD5Hash(guesses[i], state);
        if (test_set.find(guesses[i]) != test_set.end())
        {
            local_cracked += 1;
        }
    }

    auto end_hash = system_clock::now();
    auto duration = duration_cast<microseconds>(end_hash - start_hash);
    local_hash_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;

    return local_cracked;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    mpi_rank = rank;
    mpi_size = size;

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    double time_generate = 0;

    double time_train_parse = 0;
    double time_train_order = 0;

    // 加载一些测试数据
    unordered_set<std::string> test_set;
    test_set.max_load_factor(0.7f);
    test_set.reserve(1000000);
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count=0;
    string pw;
    while(test_data>>pw)
    {   
        test_count+=1;
        test_set.insert(pw);
        if (test_count>=1000000)
        {
            break;
        }
    }

    PriorityQueue q;

    if (rank != 0)
    {
        cout.setstate(ios_base::failbit);
    }

    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");

    auto end_train_parse = system_clock::now();

    q.m.order();
    auto end_train = system_clock::now();

    if (rank != 0)
    {
        cout.clear();
    }

    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    auto duration_train_parse = duration_cast<microseconds>(end_train_parse - start_train);

    auto duration_train_order = duration_cast<microseconds>(end_train - end_train_parse);

    time_train_parse = double(duration_train_parse.count()) * microseconds::period::num / microseconds::period::den;

    time_train_order = double(duration_train_order.count()) * microseconds::period::num / microseconds::period::den;
    int cracked = 0;

    q.init();

    int curr_num = 0;
    int history = 0;
    int generate_n = 10000000;
    int global_total = 0;
    auto start = system_clock::now();

    while (!q.priority.empty())
    {
        auto start_generate = system_clock::now();
        int total_tasks = q.PopNext();
        auto end_generate = system_clock::now();
        auto duration_generate = duration_cast<microseconds>(end_generate - start_generate);
        time_generate += double(duration_generate.count()) * microseconds::period::num / microseconds::period::den;
        
        global_total += total_tasks;

        if (global_total - curr_num >= 100000)
        {
            if (rank == 0)
            {
                cout << "Guesses generated: " << history + global_total << endl;
            }

            curr_num = global_total;

            if (history + global_total > generate_n)
            {
                // double max_generate_time = 0.0;
                // MPI_Reduce(&time_generate, &max_generate_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                double max_pop_assign = 0.0;
                double max_generate_only = 0.0;
                double max_newpts = 0.0;
                double max_insert = 0.0;
                double max_popnext_total = 0.0;

                MPI_Reduce(&q.time_pop_assign, &max_pop_assign, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                MPI_Reduce(&q.time_generate_only, &max_generate_only, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                MPI_Reduce(&q.time_newpts, &max_newpts, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                MPI_Reduce(&q.time_insert, &max_insert, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                MPI_Reduce(&q.time_popnext_total, &max_popnext_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                
                if (rank == 0)
                {
                    auto end = system_clock::now();
                    auto duration = duration_cast<microseconds>(end - start);
                    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

                    // // cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                    // cout << "Guess time:" << max_generate_time << "seconds" << endl;
                    // cout << "Hash time:" << time_hash << "seconds" << endl;
                    // cout << "Train time:" << time_train << "seconds" << endl;
                    // cout << "Cracked:" << cracked << endl;
                    cout << "Guess time:" << max_popnext_total << "seconds" << endl;
                    cout << "PopAssign time:" << max_pop_assign << "seconds" << endl;
                    cout << "GenerateOnly time:" << max_generate_only << "seconds" << endl;
                    cout << "NewPT time:" << max_newpts << "seconds" << endl;
                    cout << "Insert time:" << max_insert << "seconds" << endl;
                    cout << "Hash time:" << time_hash << "seconds" << endl;
                    cout << "Train time:" << time_train << "seconds" << endl;
                    cout << "TrainParse time:" << time_train_parse << "seconds" << endl;
                    cout << "TrainOrder time:" << time_train_order << "seconds" << endl;
                    cout << "TrainLocal time:" << q.m.time_train_local << "seconds" << endl;
                    cout << "TrainMerge time:" << q.m.time_train_merge << "seconds" << endl;
                    cout << "TrainOther time:" << time_train_parse - q.m.time_train_local - q.m.time_train_merge  << "seconds" << endl;
                    cout << "Cracked:" << cracked << endl;
                }

                break;
            }
        }

        if (curr_num > 1000000)
        {
            double local_hash_time = 0.0;
            long long local_cracked = hashLocalGuesses(q.guesses, test_set, local_hash_time);

            long long batch_cracked = 0;
            double max_hash_time = 0.0;

            MPI_Reduce(&local_cracked, &batch_cracked, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_hash_time, &max_hash_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                cracked += batch_cracked;
                time_hash += max_hash_time;
            }

            history += curr_num;
            curr_num = 0;
            global_total = 0;
            q.guesses.clear();
        }
    }
    MPI_Finalize();
    return 0;
}