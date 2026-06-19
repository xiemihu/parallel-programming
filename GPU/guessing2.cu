#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <algorithm>
using namespace std;

static char *d_guess_buf = nullptr;
static char *d_values_buf = nullptr;
static char *d_out_buf = nullptr;

static size_t d_guess_cap = 0;
static size_t d_values_cap = 0;
static size_t d_out_cap = 0;

// 多 PT 批量 GPU 生成需要的偏移量和长度数组
static int *d_prefix_offsets_buf = nullptr;
static int *d_prefix_lens_buf = nullptr;
static int *d_value_offsets_buf = nullptr;
static int *d_value_lens_buf = nullptr;
static int *d_pt_starts_buf = nullptr;
static int *d_pt_counts_buf = nullptr;
static int *d_output_offsets_buf = nullptr;

static size_t d_prefix_offsets_cap = 0;
static size_t d_prefix_lens_cap = 0;
static size_t d_value_offsets_cap = 0;
static size_t d_value_lens_cap = 0;
static size_t d_pt_starts_cap = 0;
static size_t d_pt_counts_cap = 0;
static size_t d_output_offsets_cap = 0;

// 进阶要求2：CPU/GPU 重叠统计
static double cpu_queue_work_time_ms = 0.0;
static double gpu_wait_after_cpu_work_ms = 0.0;
static long long overlap_batch_count = 0;

static void printOverlapStats()
{
    cout << endl;
    cout << "========== CPU/GPU Overlap Stats ==========" << endl;
    cout << "Overlap batch count: " << overlap_batch_count << endl;
    cout << "CPU queue work during GPU time: " << cpu_queue_work_time_ms << " ms" << endl;
    cout << "GPU wait after CPU work: " << gpu_wait_after_cpu_work_ms << " ms" << endl;
    cout << "===========================================" << endl;
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

    if(d_prefix_offsets_buf != nullptr) {
        cudaFree(d_prefix_offsets_buf);
        d_prefix_offsets_buf = nullptr;
        d_prefix_offsets_cap = 0;
    }

    if (d_prefix_lens_buf != nullptr) {
        cudaFree(d_prefix_lens_buf);
        d_prefix_lens_buf = nullptr;
        d_prefix_lens_cap = 0;
    }

    if (d_value_offsets_buf != nullptr) {
        cudaFree(d_value_offsets_buf);
        d_value_offsets_buf = nullptr;
        d_value_offsets_cap = 0;
    }

    if (d_value_lens_buf != nullptr) {
        cudaFree(d_value_lens_buf);
        d_value_lens_buf = nullptr;
        d_value_lens_cap = 0;
    }

    if (d_pt_starts_buf != nullptr) {
        cudaFree(d_pt_starts_buf);
        d_pt_starts_buf = nullptr;
        d_pt_starts_cap = 0;
    }

    if (d_pt_counts_buf != nullptr) {
        cudaFree(d_pt_counts_buf);
        d_pt_counts_buf = nullptr;
        d_pt_counts_cap = 0;
    }

    if (d_output_offsets_buf != nullptr) {
        cudaFree(d_output_offsets_buf);
        d_output_offsets_buf = nullptr;
        d_output_offsets_cap = 0;
    }
}

static bool cuda_registered = false;
static void registerCudaResources(){
    if (!cuda_registered){
        cuda_registered = true;
        atexit(freeCudaBuffers);
        atexit(printOverlapStats);
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

static double elapsedMs(
    std::chrono::high_resolution_clock::time_point start,
    std::chrono::high_resolution_clock::time_point end
) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
static size_t nextCudaCap(size_t old_cap, size_t need_bytes)
{
    size_t new_cap = old_cap;
    if (new_cap == 0) new_cap = need_bytes;
    while (new_cap < need_bytes) new_cap *= 2;
    return new_cap;
}
static void ensureCudaBuffers(size_t guess_bytes, size_t values_bytes, size_t out_bytes)
{
    if (d_guess_cap < guess_bytes) {
        size_t new_cap = nextCudaCap(d_guess_cap, guess_bytes);

        if (d_guess_buf != nullptr) CUDA_CHECK(cudaFree(d_guess_buf));

        CUDA_CHECK(cudaMalloc((void **)&d_guess_buf, new_cap));
        d_guess_cap = new_cap;
    }

    if (d_values_cap < values_bytes) {
        size_t new_cap = nextCudaCap(d_values_cap, values_bytes);

        if (d_values_buf != nullptr) CUDA_CHECK(cudaFree(d_values_buf));

        CUDA_CHECK(cudaMalloc((void **)&d_values_buf, new_cap));
        d_values_cap = new_cap;
    }

    if (d_out_cap < out_bytes) {
        size_t new_cap = nextCudaCap(d_out_cap, out_bytes);

        if (d_out_buf != nullptr) CUDA_CHECK(cudaFree(d_out_buf));

        CUDA_CHECK(cudaMalloc((void **)&d_out_buf, new_cap));
        d_out_cap = new_cap;
    }
}

static void ensureCudaIntBuffer(int **buf, size_t &cap, size_t need_bytes)
{
    if (cap < need_bytes) {
        size_t new_cap = nextCudaCap(cap, need_bytes);

        if (*buf != nullptr) CUDA_CHECK(cudaFree(*buf));

        CUDA_CHECK(cudaMalloc((void **)buf, new_cap));
        cap = new_cap;
    }
}

__global__ void generateKernel(
    const char *d_guess,
    int guess_len,
    const char *d_values,
    int value_len,
    int total_tasks,
    char *d_out,
    int out_stride) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total_tasks)  return;

    char *dst = d_out + (size_t)tid * out_stride;
    const char *value = d_values + (size_t)tid * value_len;

    for (int i = 0; i < guess_len; ++i) dst[i] = d_guess[i];

    for (int i = 0; i < value_len; ++i) dst[guess_len + i] = value[i];

    dst[guess_len + value_len] = '\0';
}

__global__ void generateBatchKernel(
    const char *d_prefixes,
    const int *d_prefix_offsets,
    const int *d_prefix_lens,
    const char *d_values,
    const int *d_value_offsets,
    const int *d_value_lens,
    const int *d_pt_starts,
    const int *d_pt_counts,
    int num_pts,
    char *d_out,
    const int *d_output_offsets,
    int total_tasks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total_tasks) return;

    int pt_idx = 0;
    int local_idx = tid;

    while (pt_idx < num_pts && local_idx >= d_pt_counts[pt_idx]) {
        local_idx -= d_pt_counts[pt_idx];
        pt_idx += 1;
    }

    if (pt_idx >= num_pts) return;

    int value_idx = d_pt_starts[pt_idx] + local_idx;

    const char *prefix = d_prefixes + d_prefix_offsets[pt_idx];
    int prefix_len = d_prefix_lens[pt_idx];

    const char *value = d_values + d_value_offsets[value_idx];
    int value_len = d_value_lens[value_idx];

    char *dst = d_out + d_output_offsets[tid];

    for (int i = 0; i < prefix_len; ++i) {
        dst[i] = prefix[i];
    }

    for (int i = 0; i < value_len; ++i) {
        dst[prefix_len + i] = value[i];
    }

    dst[prefix_len + value_len] = '\0';
}

static void cpuGenerateFallback(PriorityQueue *q, const string &guess, segment *a, int total_tasks) {
    size_t gbegin = q->guesses.size();
    q->guesses.resize(gbegin + (size_t)total_tasks);

    for (int i = 0; i < total_tasks; ++i)  q->guesses[gbegin + (size_t)i] = guess + a->ordered_values[i];

    q->total_guesses = (int)q->guesses.size();
}

void threadMain(PriorityQueue *q, const string &guess, segment *a, int total_tasks) {
    registerCudaResources();

    if (total_tasks <= 0) return;

    const int GPU_THRESHOLD = 131072;

    if (total_tasks < GPU_THRESHOLD) {
        cpuGenerateFallback(q, guess, a, total_tasks);
        return;
    }

    size_t gbegin = q->guesses.size();
    q->guesses.resize(gbegin + (size_t)total_tasks);

    int guess_len = (int)guess.size();
    int value_len = a->length;
    int out_stride = guess_len + value_len + 1;

    const int CHUNK = 1 << 20;

    for (int base = 0; base < total_tasks; base += CHUNK) {
        int now = min(CHUNK, total_tasks - base);

        vector<char> h_guess(max(1, guess_len));
        if (guess_len > 0)  memcpy(h_guess.data(), guess.data(), guess_len);

        vector<char> h_values((size_t)now * value_len, 0);

        for (int i = 0; i < now; ++i) {
            const string &s = a->ordered_values[base + i];
            int copy_len = min(value_len, (int)s.size());
            memcpy(&h_values[(size_t)i * value_len], s.data(), copy_len);
        }

        vector<char> h_out((size_t)now * out_stride, 0);

        size_t guess_bytes = h_guess.size() * sizeof(char);
        size_t values_bytes = h_values.size() * sizeof(char);
        size_t out_bytes = h_out.size() * sizeof(char);

        ensureCudaBuffers(guess_bytes, values_bytes, out_bytes);

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

        int block_size = 256;
        int grid_size = (now + block_size - 1) / block_size;

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
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            h_out.data(),
            d_out_buf,
            out_bytes,
            cudaMemcpyDeviceToHost
        ));

        for (int i = 0; i < now; ++i) {
            q->guesses[gbegin + (size_t)base + (size_t)i] =
                string(&h_out[(size_t)i * out_stride], out_stride - 1);
        }
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

void PopNextBatch(PriorityQueue *q, int pt_batch_num)
{
    if (q->priority.empty()) {
        return;
    }

    registerCudaResources();

    int real_batch_num = min(pt_batch_num, (int)q->priority.size());

    vector<PT> pt_batch;
    pt_batch.reserve(real_batch_num);

    for (int i = 0; i < real_batch_num; ++i) {
        pt_batch.emplace_back(q->priority[i]);
    }

    // 先从优先队列中删除这批 PT
    q->priority.erase(
        q->priority.begin(),
        q->priority.begin() + real_batch_num
    );

    const int BATCH_GPU_THRESHOLD = 131072;

    int batch_tasks = 0;

    for (PT &pt : pt_batch) {
        int last_idx = (int)pt.content.size() - 1;
        batch_tasks += pt.max_indices[last_idx];
    }

    /*
    * 如果整批任务本身就很小，就不要做多 PT GPU 打包。
    * 否则会出现：先构造大量 prefix/value/offset 数组，
    * 最后又发现没必要上 GPU，白白增加 CPU 端开销。
    */
    if (batch_tasks < BATCH_GPU_THRESHOLD) {
        for (PT &pt : pt_batch) {
            q->Generate(pt);
        }

        vector<PT> new_pts;

        for (PT &pt : pt_batch) {
            vector<PT> tmp = pt.NewPTs();

            for (PT &new_pt : tmp) {
                q->CalProb(new_pt);
                new_pts.emplace_back(new_pt);
            }
        }

        q->priority.insert(
            q->priority.end(),
            new_pts.begin(),
            new_pts.end()
        );

        sort(
            q->priority.begin(),
            q->priority.end(),
            [](const PT &a, const PT &b) {
                return a.prob > b.prob;
            }
        );

        return;
    }

    vector<char> h_prefixes;
    vector<char> h_values;
    vector<char> h_out;

    vector<int> h_prefix_offsets;
    vector<int> h_prefix_lens;
    vector<int> h_value_offsets;
    vector<int> h_value_lens;
    vector<int> h_pt_starts;
    vector<int> h_pt_counts;
    vector<int> h_output_offsets;

    // CPU 端解析输出时使用，不需要传给 GPU
    vector<int> h_output_lens;

    h_prefix_offsets.reserve(real_batch_num);
    h_prefix_lens.reserve(real_batch_num);
    h_pt_starts.reserve(real_batch_num);
    h_pt_counts.reserve(real_batch_num);

    h_value_offsets.reserve((size_t)batch_tasks);
    h_value_lens.reserve((size_t)batch_tasks);
    h_output_offsets.reserve((size_t)batch_tasks);
    h_output_lens.reserve((size_t)batch_tasks);

    int total_tasks = 0;
    int total_output_size = 0;

    // 1. 把多个 PT 的 prefix 和 value 全部打包
    const int SINGLE_PT_GPU_THRESHOLD = 131072;

    // 1. 把多个 PT 的 prefix 和 value 全部打包
    for (PT &pt : pt_batch) {
        int last_idx = (int)pt.content.size() - 1;
        int pt_count = pt.max_indices[last_idx];

        /*
        * 单个 PT 太小，不放进多 PT GPU batch。
        * 小 PT 直接用原来的 Generate 处理。
        * 注意：这个判断要放在构造 prefix 之前，
        * 否则小 PT 会在这里构造一次 prefix，
        * q->Generate(pt) 里面又构造一次 prefix。
        */
        if (pt_count < SINGLE_PT_GPU_THRESHOLD) {
            q->Generate(pt);
            continue;
        }

        string guess;

        // 构造 prefix，也就是最后一个 segment 之前的所有 segment value
        for (int seg_idx = 0; seg_idx < last_idx; ++seg_idx) {
            int idx = pt.curr_indices[seg_idx];

            if (pt.content[seg_idx].type == 1) {
                guess += q->m.letters[q->m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2) {
                guess += q->m.digits[q->m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3) {
                guess += q->m.symbols[q->m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
        }

        segment *a = nullptr;

        if (pt.content[last_idx].type == 1) {
            a = &q->m.letters[q->m.FindLetter(pt.content[last_idx])];
        }
        if (pt.content[last_idx].type == 2) {
            a = &q->m.digits[q->m.FindDigit(pt.content[last_idx])];
        }
        if (pt.content[last_idx].type == 3) {
            a = &q->m.symbols[q->m.FindSymbol(pt.content[last_idx])];
        }

        if (a == nullptr) {
            cerr << "Invalid segment type in PopNextBatch" << endl;
            exit(1);
        }

        h_prefix_offsets.emplace_back((int)h_prefixes.size());
        h_prefix_lens.emplace_back((int)guess.size());

        h_prefixes.insert(
            h_prefixes.end(),
            guess.begin(),
            guess.end()
        );

        h_pt_starts.emplace_back((int)h_value_offsets.size());
        h_pt_counts.emplace_back(pt_count);

        for (int i = 0; i < pt_count; ++i) {
            const string &value = a->ordered_values[i];

            h_value_offsets.emplace_back((int)h_values.size());
            h_value_lens.emplace_back((int)value.size());

            h_values.insert(
                h_values.end(),
                value.begin(),
                value.end()
            );

            h_output_offsets.emplace_back(total_output_size);
            h_output_lens.emplace_back((int)guess.size() + (int)value.size());

            total_output_size += (int)guess.size() + (int)value.size() + 1;
        }

        total_tasks += pt_count;
    }

    if (total_tasks <= 0) {
        vector<PT> new_pts;

        for (PT &pt : pt_batch) {
            vector<PT> tmp = pt.NewPTs();

            for (PT &new_pt : tmp) {
                q->CalProb(new_pt);
                new_pts.emplace_back(new_pt);
            }
        }

        q->priority.insert(
            q->priority.end(),
            new_pts.begin(),
            new_pts.end()
        );

        sort(
            q->priority.begin(),
            q->priority.end(),
            [](const PT &a, const PT &b) {
                return a.prob > b.prob;
            }
        );

        return;
    }

    /*
     * h_prefixes 或 h_values 可能为空。
     * 例如单 segment PT 的 prefix 为空。
     * 为了避免 cudaMemcpy 访问空 vector，这里补一个 dummy 字节。
     */
    if (h_prefixes.empty()) {
        h_prefixes.emplace_back('\0');
    }

    if (h_values.empty()) {
        h_values.emplace_back('\0');
    }

    h_out.resize((size_t)total_output_size, 0);

    size_t prefix_bytes = h_prefixes.size() * sizeof(char);
    size_t values_bytes = h_values.size() * sizeof(char);
    size_t out_bytes = h_out.size() * sizeof(char);

    size_t prefix_offsets_bytes = h_prefix_offsets.size() * sizeof(int);
    size_t prefix_lens_bytes = h_prefix_lens.size() * sizeof(int);
    size_t value_offsets_bytes = h_value_offsets.size() * sizeof(int);
    size_t value_lens_bytes = h_value_lens.size() * sizeof(int);
    size_t pt_starts_bytes = h_pt_starts.size() * sizeof(int);
    size_t pt_counts_bytes = h_pt_counts.size() * sizeof(int);
    size_t output_offsets_bytes = h_output_offsets.size() * sizeof(int);

    // 2. 申请或复用 GPU buffer
    ensureCudaBuffers(prefix_bytes, values_bytes, out_bytes);

    ensureCudaIntBuffer(
        &d_prefix_offsets_buf,
        d_prefix_offsets_cap,
        prefix_offsets_bytes
    );

    ensureCudaIntBuffer(
        &d_prefix_lens_buf,
        d_prefix_lens_cap,
        prefix_lens_bytes
    );

    ensureCudaIntBuffer(
        &d_value_offsets_buf,
        d_value_offsets_cap,
        value_offsets_bytes
    );

    ensureCudaIntBuffer(
        &d_value_lens_buf,
        d_value_lens_cap,
        value_lens_bytes
    );

    ensureCudaIntBuffer(
        &d_pt_starts_buf,
        d_pt_starts_cap,
        pt_starts_bytes
    );

    ensureCudaIntBuffer(
        &d_pt_counts_buf,
        d_pt_counts_cap,
        pt_counts_bytes
    );

    ensureCudaIntBuffer(
        &d_output_offsets_buf,
        d_output_offsets_cap,
        output_offsets_bytes
    );

    // 3. Host -> Device
    CUDA_CHECK(cudaMemcpy(
        d_guess_buf,
        h_prefixes.data(),
        prefix_bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_values_buf,
        h_values.data(),
        values_bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_prefix_offsets_buf,
        h_prefix_offsets.data(),
        prefix_offsets_bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_prefix_lens_buf,
        h_prefix_lens.data(),
        prefix_lens_bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_value_offsets_buf,
        h_value_offsets.data(),
        value_offsets_bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_value_lens_buf,
        h_value_lens.data(),
        value_lens_bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_pt_starts_buf,
        h_pt_starts.data(),
        pt_starts_bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_pt_counts_buf,
        h_pt_counts.data(),
        pt_counts_bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_output_offsets_buf,
        h_output_offsets.data(),
        output_offsets_bytes,
        cudaMemcpyHostToDevice
    ));

    // 4. 一个 kernel 同时生成多个 PT 的候选口令
    int block_size = 256;
    int grid_size = (total_tasks + block_size - 1) / block_size;

    int gpu_pt_num = (int)h_pt_counts.size();
    generateBatchKernel<<<grid_size, block_size>>>(
        d_guess_buf,
        d_prefix_offsets_buf,
        d_prefix_lens_buf,
        d_values_buf,
        d_value_offsets_buf,
        d_value_lens_buf,
        d_pt_starts_buf,
        d_pt_counts_buf,
        gpu_pt_num,
        d_out_buf,
        d_output_offsets_buf,
        total_tasks
    );

    CUDA_CHECK(cudaGetLastError());

    /*
    * 进阶要求2：
    * GPU kernel 启动后，CPU 不马上等待 GPU，
    * 而是利用 GPU 计算时间生成 new_pts、计算概率并维护优先队列。
    */
    auto cpu_work_start = std::chrono::high_resolution_clock::now();

    vector<PT> new_pts;

    for (PT &pt : pt_batch) {
        vector<PT> tmp = pt.NewPTs();

        for (PT &new_pt : tmp) {
            q->CalProb(new_pt);
            new_pts.emplace_back(new_pt);
        }
    }

    q->priority.insert(
        q->priority.end(),
        new_pts.begin(),
        new_pts.end()
    );

    sort(
        q->priority.begin(),
        q->priority.end(),
        [](const PT &a, const PT &b) {
            return a.prob > b.prob;
        }
    );

    auto cpu_work_end = std::chrono::high_resolution_clock::now();
    cpu_queue_work_time_ms += elapsedMs(cpu_work_start, cpu_work_end);
    overlap_batch_count += 1;

    /*
    * CPU 做完队列维护后，再等待 GPU。
    * 如果 GPU 已经完成，这里的等待时间会很小。
    */
    auto wait_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaDeviceSynchronize());

    auto wait_end = std::chrono::high_resolution_clock::now();
    gpu_wait_after_cpu_work_ms += elapsedMs(wait_start, wait_end);

    // 5. Device -> Host
    CUDA_CHECK(cudaMemcpy(
        h_out.data(),
        d_out_buf,
        out_bytes,
        cudaMemcpyDeviceToHost
    ));

    // 6. 解析 GPU 结果，写回 q->guesses
    size_t gbegin = q->guesses.size();
    q->guesses.resize(gbegin + (size_t)total_tasks);

    for (int i = 0; i < total_tasks; ++i) {
        q->guesses[gbegin + (size_t)i] =
            string(
                &h_out[(size_t)h_output_offsets[i]],
                h_output_lens[i]
            );
    }

    q->total_guesses = (int)q->guesses.size();
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