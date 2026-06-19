#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <algorithm>
using namespace std;
static long long gpu_call_count = 0;
static long long gpu_task_count = 0;
static long long cpu_call_count = 0;
static long long cpu_task_count = 0;
static double gpu_kernel_time_ms = 0.0;
static bool cuda_stats_registered = false;

static double pack_time_ms = 0.0;
static double malloc_time_ms = 0.0;
static double h2d_time_ms = 0.0;
static double d2h_time_ms = 0.0;
static double rebuild_time_ms = 0.0;
static double free_time_ms = 0.0;

static double batch_generate_time_ms = 0.0;
static double batch_newpt_time_ms = 0.0;
static double batch_insert_time_ms = 0.0;

static char *d_guess_buf = nullptr;
static char *d_values_buf = nullptr;
static char *d_out_buf = nullptr;

static size_t d_guess_cap = 0;
static size_t d_values_cap = 0;
static size_t d_out_cap = 0;
static long long cuda_alloc_count = 0;
static long long cuda_expand_count = 0;

static void printCudaStats()
{
    cout << "\n========== CUDA Generate Stats ==========" << endl;
    cout << "GPU calls: " << gpu_call_count << endl;
    cout << "GPU tasks: " << gpu_task_count << endl;
    cout << "CPU fallback calls: " << cpu_call_count << endl;
    cout << "CPU fallback tasks: " << cpu_task_count << endl;

    cout << "GPU kernel time: " << gpu_kernel_time_ms << " ms" << endl;

    cout << "\n---------- Time Breakdown ----------" << endl;
    cout << "Pack time: " << pack_time_ms << " ms" << endl;
    cout << "cudaMalloc time: " << malloc_time_ms << " ms" << endl;
    cout << "H2D memcpy time: " << h2d_time_ms << " ms" << endl;
    cout << "D2H memcpy time: " << d2h_time_ms << " ms" << endl;
    cout << "Rebuild string time: " << rebuild_time_ms << " ms" << endl;
    cout << "cudaFree time: " << free_time_ms << " ms" << endl;

    double total_measured =
        pack_time_ms +
        malloc_time_ms +
        h2d_time_ms +
        gpu_kernel_time_ms +
        d2h_time_ms +
        rebuild_time_ms +
        free_time_ms;

    cout << "Measured total: " << total_measured << " ms" << endl;

    cout << "\n---------- Batch Time ----------" << endl;
    cout << "Batch Generate time: " << batch_generate_time_ms << " ms" << endl;
    cout << "Batch NewPTs time: " << batch_newpt_time_ms << " ms" << endl;
    cout << "Batch Insert/Sort time: " << batch_insert_time_ms << " ms" << endl;
    cout << "=========================================" << endl;
}
static inline double elapsedMs(
    const chrono::high_resolution_clock::time_point &start,
    const chrono::high_resolution_clock::time_point &end
) {
    return chrono::duration<double, std::milli>(end - start).count();
}
static void freeCudaBuffers()
{
    if (d_guess_buf != nullptr) {
        cudaFree(d_guess_buf);
        d_guess_buf = nullptr;
        d_guess_cap = 0;
    }

    if (d_values_buf != nullptr) {
        cudaFree(d_values_buf);
        d_values_buf = nullptr;
        d_values_cap = 0;
    }

    if (d_out_buf != nullptr) {
        cudaFree(d_out_buf);
        d_out_buf = nullptr;
        d_out_cap = 0;
    }
}
static void registerCudaStats()
{
    if (!cuda_stats_registered)
    {
        cuda_stats_registered = true;

        int dev = 0;
        cudaGetDevice(&dev);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        cout << "CUDA device: " << prop.name << endl;
        cout << "SM count: " << prop.multiProcessorCount << endl;
        cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;

        atexit(freeCudaBuffers);
        atexit(printCudaStats);
    }
}

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        cerr << "CUDA error: " << cudaGetErrorString(err)          \
             << " at " << __FILE__ << ":" << __LINE__ << endl;   \
        exit(1);                                                   \
    }                                                             \
} while (0)

static size_t nextCudaCap(size_t old_cap, size_t need_bytes)
{
    size_t new_cap = old_cap;

    if (new_cap == 0) {
        new_cap = need_bytes;
    }

    while (new_cap < need_bytes) {
        new_cap *= 2;
    }

    return new_cap;
}
static void ensureCudaBuffers(size_t guess_bytes, size_t values_bytes, size_t out_bytes)
{
    if (d_guess_cap < guess_bytes) {
        cuda_expand_count += 1;
        size_t new_cap = nextCudaCap(d_guess_cap, guess_bytes);

        if (d_guess_buf != nullptr) {
            CUDA_CHECK(cudaFree(d_guess_buf));
        }

        CUDA_CHECK(cudaMalloc((void **)&d_guess_buf, new_cap));
        cuda_alloc_count += 1;
        d_guess_cap = new_cap;
    }

    if (d_values_cap < values_bytes) {
        cuda_expand_count += 1;
        size_t new_cap = nextCudaCap(d_values_cap, values_bytes);

        if (d_values_buf != nullptr) {
            CUDA_CHECK(cudaFree(d_values_buf));
        }

        CUDA_CHECK(cudaMalloc((void **)&d_values_buf, new_cap));
        cuda_alloc_count += 1;
        d_values_cap = new_cap;
    }

    if (d_out_cap < out_bytes) {
        cuda_expand_count += 1;
        size_t new_cap = nextCudaCap(d_out_cap, out_bytes);

        if (d_out_buf != nullptr) {
            CUDA_CHECK(cudaFree(d_out_buf));
        }

        CUDA_CHECK(cudaMalloc((void **)&d_out_buf, new_cap));
        cuda_alloc_count += 1;
        d_out_cap = new_cap;
    }
}

__global__ void generateKernel(
    const char *d_guess,
    int guess_len,
    const char *d_values,
    int value_len,
    int total_tasks,
    char *d_out,
    int out_stride
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total_tasks)  return;

    char *dst = d_out + (size_t)tid * out_stride;
    const char *value = d_values + (size_t)tid * value_len;

    for (int i = 0; i < guess_len; ++i) dst[i] = d_guess[i];

    for (int i = 0; i < value_len; ++i) dst[guess_len + i] = value[i];

    dst[guess_len + value_len] = '\0';
}

static void cpuGenerateFallback(PriorityQueue *q, const string &guess, segment *a, int total_tasks) {
    size_t gbegin = q->guesses.size();
    q->guesses.resize(gbegin + (size_t)total_tasks);

    for (int i = 0; i < total_tasks; ++i)  q->guesses[gbegin + (size_t)i] = guess + a->ordered_values[i];

    q->total_guesses = (int)q->guesses.size();
}

void threadMain(PriorityQueue *q, const string &guess, segment *a, int total_tasks) {
    registerCudaStats();
    if (total_tasks <= 0)  return;

    /*
     * 小 PT 不一定适合上 GPU。
     * 因为 cudaMalloc、cudaMemcpy、kernel launch 都有开销。
     * 这个阈值后面可以作为进阶要求 3 进行测试。
     */
    const int GPU_THRESHOLD = 131072;

    if (total_tasks < GPU_THRESHOLD) {
        cpu_call_count += 1;
        cpu_task_count += total_tasks;
        cpuGenerateFallback(q, guess, a, total_tasks);
        return;
    }
    // if (total_tasks < GPU_THRESHOLD) {
    //     cpuGenerateFallback(q, guess, a, total_tasks);
    //     return;
    // }

    size_t gbegin = q->guesses.size();
    q->guesses.resize(gbegin + (size_t)total_tasks);

    int guess_len = (int)guess.size();
    int value_len = a->length;
    int out_stride = guess_len + value_len + 1;

    gpu_call_count += 1;
    gpu_task_count += total_tasks;
    /*
     * 分块处理，避免某些 PT 的候选 value 过多，导致一次性显存占用太大。
     * 这里一次最多处理 1M 个 value。
     */
    const int CHUNK = 1 << 20;

    for (int base = 0; base < total_tasks; base += CHUNK) {
        int now = min(CHUNK, total_tasks - base);

        // 1. CPU 打包 h_guess / h_values / h_out
        auto t_pack_start = chrono::high_resolution_clock::now();

        vector<char> h_guess(max(1, guess_len));
        if (guess_len > 0) {
            memcpy(h_guess.data(), guess.data(), guess_len);
        }

        vector<char> h_values((size_t)now * value_len, 0);

        for (int i = 0; i < now; ++i) {
            const string &s = a->ordered_values[base + i];
            int copy_len = min(value_len, (int)s.size());
            memcpy(&h_values[(size_t)i * value_len], s.data(), copy_len);
        }

        vector<char> h_out((size_t)now * out_stride, 0);

        auto t_pack_end = chrono::high_resolution_clock::now();
        pack_time_ms += elapsedMs(t_pack_start, t_pack_end);

        size_t guess_bytes = h_guess.size() * sizeof(char);
        size_t values_bytes = h_values.size() * sizeof(char);
        size_t out_bytes = h_out.size() * sizeof(char);

        // 2. GPU buffer 申请/扩容计时
        auto t_malloc_start = chrono::high_resolution_clock::now();

        ensureCudaBuffers(guess_bytes, values_bytes, out_bytes);

        auto t_malloc_end = chrono::high_resolution_clock::now();
        malloc_time_ms += elapsedMs(t_malloc_start, t_malloc_end);

        // 3. Host -> Device 拷贝计时
        auto t_h2d_start = chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMemcpy(
            d_guess_buf,
            h_guess.data(),
            guess_bytes,
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_values_buf,
            h_values.data(),
            values_bytes,
            cudaMemcpyHostToDevice
        ));

        auto t_h2d_end = chrono::high_resolution_clock::now();
        h2d_time_ms += elapsedMs(t_h2d_start, t_h2d_end);

        int block_size = 256;
        int grid_size = (now + block_size - 1) / block_size;

        // 4. kernel 计时，继续用 CUDA event
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        CUDA_CHECK(cudaEventRecord(ev_start));

        generateKernel<<<grid_size, block_size>>>(
            d_guess_buf,
            guess_len,
            d_values_buf,
            value_len,
            now,
            d_out_buf,
            out_stride
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));
        gpu_kernel_time_ms += kernel_ms;

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));

        CUDA_CHECK(cudaDeviceSynchronize());

        // 5. Device -> Host 拷贝计时
        auto t_d2h_start = chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMemcpy(
            h_out.data(),
            d_out_buf,
            out_bytes,
            cudaMemcpyDeviceToHost
        ));

        auto t_d2h_end = chrono::high_resolution_clock::now();
        d2h_time_ms += elapsedMs(t_d2h_start, t_d2h_end);

        // 6. h_out 重新构造成 string 的计时
        auto t_rebuild_start = chrono::high_resolution_clock::now();

        for (int i = 0; i < now; ++i)  q->guesses[gbegin + (size_t)base + (size_t)i] = string(&h_out[(size_t)i * out_stride], out_stride - 1);

        auto t_rebuild_end = chrono::high_resolution_clock::now();
        rebuild_time_ms += elapsedMs(t_rebuild_start, t_rebuild_end);
    }

    q->total_guesses = (int)q->guesses.size();
}

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back((int)m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back((int)m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back((int)m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

static bool PTProbGreater(const PT &a, const PT &b)
{
    return a.prob > b.prob;
}

static void insertPTByProb(PriorityQueue *q, PT pt)
{
    q->CalProb(pt);

    if (q->priority.empty())
    {
        q->priority.emplace_back(pt);
        return;
    }

    auto iter = q->priority.begin();

    while (iter != q->priority.end() && iter->prob >= pt.prob)
    {
        ++iter;
    }

    q->priority.emplace(iter, pt);
}

/*
 * 进阶要求 1：
 * 一次性从优先队列中取出多个 PT，
 * 等这些 PT 都完成 Generate 后，
 * 再把它们产生的新 PT 挨个插回优先队列。
 */
void PopNextBatch(PriorityQueue *q, int pt_batch_num)
{
    if (q->priority.empty())
    {
        return;
    }

    int real_batch_num = std::min(pt_batch_num, (int)q->priority.size());

    vector<PT> pt_batch;
    pt_batch.reserve(real_batch_num);

    // 1. 一次性取出多个 PT
    for (int i = 0; i < real_batch_num; ++i)
    {
        pt_batch.emplace_back(q->priority[i]);
    }

    // 2. 先从优先队列中删除这些 PT
    q->priority.erase(
        q->priority.begin(),
        q->priority.begin() + real_batch_num
    );

    // 3. 批量 Generate
    // 这里每个 PT 内部仍然会调用你已经实现的 threadMain，
    // 即 PT 内部仍然使用 GPU/CPU fallback 生成候选口令。
    auto t_batch_generate_start = chrono::high_resolution_clock::now();

    for (PT &pt : pt_batch)
    {
        q->Generate(pt);
    }

    auto t_batch_generate_end = chrono::high_resolution_clock::now();
    batch_generate_time_ms += elapsedMs(t_batch_generate_start, t_batch_generate_end);

    // 4. 等所有 PT 都 Generate 完，再统一生成新 PT
    auto t_batch_newpt_start = chrono::high_resolution_clock::now();

    vector<PT> new_pt_batch;

    for (PT &pt : pt_batch)
    {
        vector<PT> new_pts = pt.NewPTs();

        for (PT &new_pt : new_pts)
        {
            new_pt_batch.emplace_back(new_pt);
        }
    }

    auto t_batch_newpt_end = chrono::high_resolution_clock::now();
    batch_newpt_time_ms += elapsedMs(t_batch_newpt_start, t_batch_newpt_end);

    auto t_batch_insert_start = chrono::high_resolution_clock::now();

    if (new_pt_batch.size() <= 16)
    {
        for (PT &new_pt : new_pt_batch)
        {
            insertPTByProb(q, new_pt);
        }
    }
    else
    {
        for (PT &new_pt : new_pt_batch)
        {
            q->CalProb(new_pt);
        }

        q->priority.reserve(q->priority.size() + new_pt_batch.size());

        q->priority.insert(
            q->priority.end(),
            new_pt_batch.begin(),
            new_pt_batch.end()
        );

        std::sort(q->priority.begin(), q->priority.end(),
                [](const PT &a, const PT &b)
                {
                    return a.prob > b.prob;
                });
    }

    auto t_batch_insert_end = chrono::high_resolution_clock::now();
    batch_insert_time_ms += elapsedMs(t_batch_insert_start, t_batch_insert_end);
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        // for (int i = 0; i < pt.max_indices[0]; i += 1)
        // {
        //     string guess = a->ordered_values[i];
        //     // cout << guess << endl;
        //     guesses.emplace_back(guess);
        //     total_guesses += 1;
        // }
        threadMain(this, "", a, pt.max_indices[0]);
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        // for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        // {
        //     string temp = guess + a->ordered_values[i];
        //     // cout << temp << endl;
        //     guesses.emplace_back(temp);
        //     total_guesses += 1;
        // }
        threadMain(this, guess, a, pt.max_indices[pt.content.size() - 1]);
    }
}