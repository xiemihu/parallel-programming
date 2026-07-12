#include "PCFG.h"
#include <fstream>
#include <cctype>
#include <algorithm>

// 这个文件里面的各函数你都不需要完全理解，甚至根本不需要看
// 从学术价值上讲，加速模型的训练过程是一个没什么价值的问题，因为我们一般假定统计学模型的训练成本较低
// 但是，假如你是一个投稿时顶着ddl做实验的倒霉研究生/实习生，提高训练速度就可以大幅节省你的时间了
// 所以如果你愿意，也可以尝试用多线程加速训练过程

class FastInput
{
public:
    explicit FastInput(const string &path)
    {
        file = fopen(path.c_str(), "rb");
    }

    ~FastInput()
    {
        if (file != nullptr)
        {
            fclose(file);
        }
    }

    bool isOpen() const
    {
        return file != nullptr;
    }

    bool read(string &value)
    {
        value.clear();

        int ch = GetChar();

        while (ch != EOF && IsSpace(ch))
        {
            ch = GetChar();
        }

        if (ch == EOF)
        {
            return false;
        }

        while (ch != EOF && !IsSpace(ch))
        {
            value.push_back(static_cast<char>(ch));
            ch = GetChar();
        }

        return true;
    }

private:
    FILE *file = nullptr;
    unsigned char buffer[1 << 20];
    size_t buffer_pos = 0;
    size_t buffer_num = 0;

    static bool IsSpace(int ch)
    {
        return ch == ' ' || ch == '\n' || ch == '\r' ||
               ch == '\t' || ch == '\v' || ch == '\f';
    }

    int GetChar()
    {
        if (buffer_pos == buffer_num)
        {
            buffer_num = fread(buffer, 1, sizeof(buffer), file);
            buffer_pos = 0;

            if (buffer_num == 0)
            {
                return EOF;
            }
        }

        return buffer[buffer_pos++];
    }
};

static string GetPTKey(const PT &pt)
{
    string pt_key;
    // pt_key.reserve(pt.content.size() * 4);
    pt_key.reserve(pt.content.size() * (sizeof(int) + 1));

    for (const segment &seg : pt.content)
    {
        // pt_key.push_back(char('0' + seg.type));
        // pt_key += to_string(seg.length);
        // pt_key.push_back(';');
        pt_key.push_back(static_cast<char>(seg.type));
        pt_key.append(reinterpret_cast<const char *>(&seg.length), sizeof(seg.length));
    }

    return pt_key;
}

static inline int GetCharType(char ch)
{
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) return 1;
    if (ch >= '0' && ch <= '9')  return 2;
    return 3;
}

static void MergeSegments(vector<segment> &segments,
                          vector<int> &segment_freq,
                          unordered_map<int, int> &segment_ids,
                          int &segment_id,
                          vector<segment> &local_segments,
                          const vector<int> &local_freq)
{
    vector<int> merge_ids(local_segments.size(), -1);

    for (int i = 0; i < (int)local_segments.size(); i += 1)
    {
        segment &local_seg = local_segments[i];
        auto iter = segment_ids.find(local_seg.length);

        if (iter == segment_ids.end())
        {
            segment_id += 1;
            int id = segment_id;

            segments.emplace_back(std::move(local_seg));
            segment_ids.emplace(segments.back().length, id);
            segment_freq.emplace_back(local_freq[i]);
        }
        else
        {
            int id = iter->second;
            segment_freq[id] += local_freq[i];
            merge_ids[i] = id;
        }
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < (int)local_segments.size(); i += 1)
    {
        int id = merge_ids[i];

        if (id == -1)
        {
            continue;
        }

        segment &local_seg = local_segments[i];

        for (int j = 0; j < (int)local_seg.values_list.size(); j += 1)
        {
            segments[id].insert(local_seg.values_list[j], local_seg.freqs[j]);
        }
    }
}

void model::merge(model &other_model)
{
    total_preterm += other_model.total_preterm;

    for (int i = 0; i < (int)other_model.preterminals.size(); i += 1)
    {
        PT &pt = other_model.preterminals[i];
        string pt_key = GetPTKey(pt);
        auto iter = preterm_ids.find(pt_key);

        if (iter == preterm_ids.end())
        {
            int id = GetNextPretermID();

            preterminals.emplace_back(std::move(pt));
            preterm_ids.emplace(std::move(pt_key), id);
            preterm_freq.emplace_back(other_model.preterm_freq[i]);
        }
        else
        {
            preterm_freq[iter->second] += other_model.preterm_freq[i];
        }
    }

    MergeSegments(letters,
                  letters_freq,
                  letters_ids,
                  letters_id,
                  other_model.letters,
                  other_model.letters_freq);

    MergeSegments(digits,
                  digits_freq,
                  digits_ids,
                  digits_id,
                  other_model.digits,
                  other_model.digits_freq);

    MergeSegments(symbols,
                  symbols_freq,
                  symbols_ids,
                  symbols_id,
                  other_model.symbols,
                  other_model.symbols_freq);
}

void model::trainBatch(const vector<string> &passwords, int password_num)
{
    int thread_count = omp_get_max_threads();
    int active_threads = 1;

    double t0 = omp_get_wtime();
    vector<model> local_models(thread_count);

#pragma omp parallel num_threads(thread_count)
    {
#pragma omp single
        active_threads = omp_get_num_threads();

        int tid = omp_get_thread_num();
        int begin = (long long)password_num * tid / active_threads;
        int end = (long long)password_num * (tid + 1) / active_threads;

        for (int i = begin; i < end; i += 1)
        {
            local_models[tid].parse(passwords[i]);
        }
    }

    double t1 = omp_get_wtime();

    for (int tid = 0; tid < active_threads; tid += 1)
    {
        merge(local_models[tid]);
    }

    double t2 = omp_get_wtime();

    time_train_local += t1 - t0;
    time_train_merge += t2 - t1;
}
/**
 * 怎么加速PCFG训练过程？据助教所知，没有公开文献提出过有效的加速方法（因为这么做基本无学术价值）
 * 
 * 但是统计学模型好就好在其数据是可加的。例如，假如我把数据集拆分成4个部分，并行训练4个不同的模型。
 * 然后我可以直接将四个模型的统计数据进行简单加和，就得到了和串行训练相同的模型了。
 * 
 * 说起来容易，做起来不一定容易，你可能会碰到一系列具体的工程问题。如果你决定加速训练过程，祝你好运！
 * 
 */

// 训练的wrapper，实际上就是读取训练集
// void model::train(const string &path)
// {
//     string pw;
//     ifstream train_set(path);
//     int lines = 0;
//     cout<<"Training..."<<endl;
//     cout<<"Training phase 1: reading and parsing passwords..."<<endl;
//     while (train_set >> pw)
//     {
//         lines += 1;
//         if (lines % 10000 == 0)
//         {
//             cout <<"Lines processed: "<< lines << endl;
//             // 在这里更改读取的训练集口令上限
//             if (lines > 3000000)
//             {
//                 break;
//             }
//         }
//         // 读取单个口令之后，就可以将其扔进parse函数进行PT/segment的分割、识别、统计了
//         parse(pw);
//     }
// }
void model::train(const string &path)
{
    static const int train_batch_num = 800000;

    // string pw;
    // ifstream train_set(path);
    FastInput train_set(path);

    if (!train_set.isOpen())
    {
        cerr << "Failed to open training set" << endl;
        return;
    }
    int lines = 0;

    // vector<string> passwords;
    // passwords.reserve(train_batch_num);

    cout << "Training..." << endl;
    cout << "Training phase 1: reading and parsing passwords..." << endl;

    vector<string> passwords(train_batch_num);
    int password_num = 0;

    // while (train_set >> passwords[password_num])
    while (train_set.read(passwords[password_num]))
    {
        lines += 1;

        if (lines % 10000 == 0)
        {
            cout << "Lines processed: " << lines << endl;

            if (lines > 3000000)
            {
                break;
            }
        }

        password_num += 1;

        if (password_num == train_batch_num)
        {
            trainBatch(passwords, password_num);
            password_num = 0;
        }
    }

    if (password_num > 0)
    {
        trainBatch(passwords, password_num);
    }
}

/// @brief 在模型中找到一个PT的统计数据
/// @param pt 需要查找的PT
/// @return 目标PT在模型中的对应下标
int model::FindPT(const PT &pt)
{
    string pt_key = GetPTKey(pt);

    auto iter = preterm_ids.find(pt_key);
    if (iter != preterm_ids.end())
    {
        return iter->second;
    }

    for (int id = 0; id < (int)preterminals.size(); id += 1)
    {
        if (preterminals[id].content.size() != pt.content.size())
        {
            continue;
        }

        bool equal_flag = true;

        for (int idx = 0;
             idx < (int)preterminals[id].content.size();
             idx += 1)
        {
            if (preterminals[id].content[idx].type
                    != pt.content[idx].type ||
                preterminals[id].content[idx].length
                    != pt.content[idx].length)
            {
                equal_flag = false;
                break;
            }
        }

        if (equal_flag)
        {
            preterm_ids.emplace(pt_key, id);
            return id;
        }
    }

    return -1;
}

/// @brief 在模型中找到一个letter segment的统计数据
/// @param seg 要找的letter segment
/// @return 目标letter segment的对应下标
int model::FindLetter(const segment &seg)
{
    auto iter = letters_ids.find(seg.length);
    if (iter != letters_ids.end())
    {
        return iter->second;
    }

    for (int id = 0; id < (int)letters.size(); id += 1)
    {
        if (letters[id].length == seg.length)
        {
            letters_ids.emplace(seg.length, id);
            return id;
        }
    }

    return -1;
}

/// @brief 在模型中找到一个digit segment的统计数据
/// @param seg 要找的digit segment
/// @return 目标digit segment的对应下标
int model::FindDigit(const segment &seg)
{
    auto iter = digits_ids.find(seg.length);
    if (iter != digits_ids.end())
    {
        return iter->second;
    }

    for (int id = 0; id < (int)digits.size(); id += 1)
    {
        if (digits[id].length == seg.length)
        {
            digits_ids.emplace(seg.length, id);
            return id;
        }
    }

    return -1;
}

int model::FindSymbol(const segment &seg)
{
    auto iter = symbols_ids.find(seg.length);
    if (iter != symbols_ids.end())
    {
        return iter->second;
    }

    for (int id = 0; id < (int)symbols.size(); id += 1)
    {
        if (symbols[id].length == seg.length)
        {
            symbols_ids.emplace(seg.length, id);
            return id;
        }
    }

    return -1;
}

void PT::insert(const segment &seg)
{
    content.emplace_back(seg.type, seg.length);
}

void segment::insert(const string &value)
{
    insert(value, 1);
}

void segment::insert(const string &value, int count)
{
    auto result = values.emplace(value, (int)values.size());
    int id = result.first->second;

    if (result.second)
    {
        values_list.emplace_back(value);
        freqs.emplace_back(count);
    }
    else
    {
        freqs[id] += count;
    }
}


void segment::order()
{
    vector<pair<string, int>> value_freqs;

    value_freqs.reserve(values.size());

    for (const auto &value : values)
    {
        value_freqs.emplace_back(value.first, freqs[value.second]);
    }

    sort(
        value_freqs.begin(),
        value_freqs.end(),
        [](const pair<string, int> &a,
           const pair<string, int> &b)
        {
            return a.second > b.second;
        });

    ordered_values.reserve(value_freqs.size());
    ordered_freqs.reserve(value_freqs.size() * 2);

    for (const auto &value : value_freqs)
    {
        ordered_values.emplace_back(value.first);
        ordered_freqs.emplace_back(value.second);
        total_freq += value.second;
    }

    for (const auto &value : value_freqs)
    {
        ordered_freqs.emplace_back(value.second);
        total_freq += value.second;
    }
}

void model::InsertSegment(int type, const string &value, PT &pt)
{
    segment seg(type, (int)value.size());
    int id = -1;

    if (type == 1)
    {
        id = FindLetter(seg);

        if (id == -1)
        {
            id = GetNextLettersID();
            letters.emplace_back(type, seg.length);
            letters_ids.emplace(seg.length, id);
            letters_freq.emplace_back(1);
        }
        else
        {
            letters_freq[id] += 1;
        }

        letters[id].insert(value);
    }
    else if (type == 2)
    {
        id = FindDigit(seg);

        if (id == -1)
        {
            id = GetNextDigitsID();
            digits.emplace_back(type, seg.length);
            digits_ids.emplace(seg.length, id);
            digits_freq.emplace_back(1);
        }
        else
        {
            digits_freq[id] += 1;
        }

        digits[id].insert(value);
    }
    else
    {
        id = FindSymbol(seg);

        if (id == -1)
        {
            id = GetNextSymbolsID();
            symbols.emplace_back(type, seg.length);
            symbols_ids.emplace(seg.length, id);
            symbols_freq.emplace_back(1);
        }
        else
        {
            symbols_freq[id] += 1;
        }

        symbols[id].insert(value);
    }

    pt.insert(seg);
}

void model::parse(const string &pw)
{
    PT pt;
    pt.content.reserve(4);

    int start = 0;
    int curr_type = 0;

    for (int i = 0; i < (int)pw.size(); i += 1)
    {
        int next_type = GetCharType(pw[i]);

        if (curr_type == 0)
        {
            curr_type = next_type;
            start = i;
        }
        else if (next_type != curr_type)
        {
            InsertSegment(curr_type, pw.substr(start, i - start), pt);
            curr_type = next_type;
            start = i;
        }
    }

    if (curr_type != 0)
    {
        InsertSegment(curr_type, pw.substr(start), pt);
    }

    total_preterm += 1;

    string pt_key = GetPTKey(pt);
    auto iter = preterm_ids.find(pt_key);

    if (iter == preterm_ids.end())
    {
        pt.curr_indices.assign(pt.content.size(), 0);

        int id = GetNextPretermID();
        preterminals.emplace_back(std::move(pt));
        preterm_ids.emplace(std::move(pt_key), id);
        preterm_freq.emplace_back(1);
    }
    else
    {
        preterm_freq[iter->second] += 1;
    }
}

// void model::parse(const string &pw)
// {
//     PT pt;
//     string curr_part = "";
//     int curr_type = 0; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
//     // 请学会使用这种方式写for循环：for (auto it : iterable)
//     // 相信我，以后你会用上的。You're welcome :)
//     for (char ch : pw)
//     {
//         if (isalpha(ch))
//         {
//             if (curr_type != 1)
//             {
//                 if (curr_type == 2)
//                 {
//                     segment seg(curr_type, curr_part.length());
//                     int id = FindDigit(seg);
//                     if (id == -1)
//                     {
//                         id = GetNextDigitsID();
//                         digits.emplace_back(seg);
//                         digits[id].insert(curr_part);
//                         digits_freq.emplace_back(1);
//                     }
//                     else
//                     {
//                         digits_freq[id] += 1;
//                         digits[id].insert(curr_part);
//                     }
//                     curr_part.clear();
//                     pt.insert(seg);
//                 }
//                 else if (curr_type == 3)
//                 {
//                     segment seg(curr_type, curr_part.length());
//                     int id = FindSymbol(seg);
//                     if (id == -1)
//                     {
//                         id = GetNextSymbolsID();
//                         symbols.emplace_back(seg);
//                         symbols_freq.emplace_back(1);
//                         symbols[id].insert(curr_part);
//                     }
//                     else
//                     {
//                         symbols_freq[id] += 1;
//                         symbols[id].insert(curr_part);
//                     }
//                     curr_part.clear();
//                     pt.insert(seg);
//                 }
//             }
//             curr_type = 1;
//             curr_part += ch;
//         }
//         else if (isdigit(ch))
//         {
//             if (curr_type != 2)
//             {
//                 if (curr_type == 1)
//                 {
//                     segment seg(curr_type, curr_part.length());
//                     int id = FindLetter(seg);
//                     if (id == -1)
//                     {
//                         id = GetNextLettersID();
//                         letters.emplace_back(seg);
//                         letters_freq.emplace_back(1);
//                         letters[id].insert(curr_part);
//                     }
//                     else
//                     {
//                         letters_freq[id] += 1;
//                         letters[id].insert(curr_part);
//                     }
//                     curr_part.clear();
//                     pt.insert(seg);
//                 }
//                 else if (curr_type == 3)
//                 {
//                     segment seg(curr_type, curr_part.length());
//                     int id = FindSymbol(seg);
//                     if (id == -1)
//                     {
//                         id = GetNextSymbolsID();
//                         symbols.emplace_back(seg);
//                         symbols_freq.emplace_back(1);
//                         symbols[id].insert(curr_part);
//                     }
//                     else
//                     {
//                         symbols_freq[id] += 1;
//                         symbols[id].insert(curr_part);
//                     }
//                     curr_part.clear();
//                     pt.insert(seg);
//                 }
//             }
//             curr_type = 2;
//             curr_part += ch;
//         }
//         else
//         {
//             if (curr_type != 3)
//             {
//                 if (curr_type == 1)
//                 {
//                     segment seg(curr_type, curr_part.length());
//                     int id = FindLetter(seg);
//                     if (id == -1)
//                     {
//                         id = GetNextLettersID();
//                         letters.emplace_back(seg);
//                         letters_freq.emplace_back(1);
//                         letters[id].insert(curr_part);
//                     }
//                     else
//                     {
//                         letters_freq[id] += 1;
//                         letters[id].insert(curr_part);
//                     }
//                     curr_part.clear();
//                     pt.insert(seg);
//                 }
//                 else if (curr_type == 2)
//                 {
//                     segment seg(curr_type, curr_part.length());
//                     int id = FindDigit(seg);
//                     if (id == -1)
//                     {
//                         id = GetNextDigitsID();
//                         digits.emplace_back(seg);
//                         digits_freq.emplace_back(1);
//                         digits[id].insert(curr_part);
//                     }
//                     else
//                     {
//                         digits_freq[id] += 1;
//                         digits[id].insert(curr_part);
//                     }
//                     curr_part.clear();
//                     pt.insert(seg);
//                 }
//             }
//             curr_type = 3;
//             curr_part += ch;
//         }
//     }
//     if (!curr_part.empty())
//     {
//         if (curr_type == 1)
//         {
//             segment seg(curr_type, curr_part.length());
//             int id = FindLetter(seg);
//             if (id == -1)
//             {
//                 id = GetNextLettersID();
//                 letters.emplace_back(seg);
//                 letters_freq.emplace_back(1);
//                 letters[id].insert(curr_part);
//             }
//             else
//             {
//                 letters_freq[id] += 1;
//                 letters[id].insert(curr_part);
//             }
//             curr_part.clear();
//             pt.insert(seg);
//         }
//         else if (curr_type == 2)
//         {
//             segment seg(curr_type, curr_part.length());
//             int id = FindDigit(seg);
//             if (id == -1)
//             {
//                 id = GetNextDigitsID();
//                 digits.emplace_back(seg);
//                 digits_freq.emplace_back(1);
//                 digits[id].insert(curr_part);
//             }
//             else
//             {
//                 digits_freq[id] += 1;
//                 digits[id].insert(curr_part);
//             }
//             curr_part.clear();
//             pt.insert(seg);
//         }
//         else
//         {
//             segment seg(curr_type, curr_part.length());
//             int id = FindSymbol(seg);
//             if (id == -1)
//             {
//                 id = GetNextSymbolsID();
//                 symbols.emplace_back(seg);
//                 symbols_freq.emplace_back(1);
//                 symbols[id].insert(curr_part);
//             }
//             else
//             {
//                 symbols_freq[id] += 1;
//                 symbols[id].insert(curr_part);
//             }
//             curr_part.clear();
//             pt.insert(seg);
//         }
//     }
//     // pt.PrintPT();
//     // cout<<endl;
//     // cout << FindPT(pt) << endl;
//     total_preterm += 1;

//     string pt_key = GetPTKey(pt);
//     auto iter = preterm_ids.find(pt_key);

//     if (iter == preterm_ids.end())
//     {
//         for (int i = 0; i < (int)pt.content.size(); i += 1)
//         {
//             pt.curr_indices.emplace_back(0);
//         }

//         int id = GetNextPretermID();

//         preterminals.emplace_back(pt);
//         preterm_ids.emplace(pt_key, id);
//         preterm_freq.emplace_back(1);
//     }
//     else
//     {
//         preterm_freq[iter->second] += 1;
//     }
// }

void segment::PrintSeg()
{
    if (type == 1)
    {
        cout << "L" << length;
    }
    if (type == 2)
    {
        cout << "D" << length;
    }
    if (type == 3)
    {
        cout << "S" << length;
    }
}

void segment::PrintValues()
{
    // order();
    for (string iter : ordered_values)
    {
        cout << iter << " freq:" << freqs[values[iter]] << endl;
    }
}

void PT::PrintPT()
{
    for (auto iter : content)
    {
        iter.PrintSeg();
    }
}

void model::print()
{
    cout << "preterminals:" << endl;
    for (int i = 0; i < preterminals.size(); i += 1)
    {
        preterminals[i].PrintPT();
        // cout << preterminals[i].curr_indices.size() << endl;
        cout << " freq:" << preterm_freq[i];
        cout << endl;
    }
    // order();
    for (auto iter : ordered_pts)
    {
        iter.PrintPT();
        cout << " freq:" << preterm_freq[FindPT(iter)];
        cout << endl;
    }
    cout << "segments:" << endl;
    for (int i = 0; i < letters.size(); i += 1)
    {
        letters[i].PrintSeg();
        // letters[i].PrintValues();
        cout << " freq:" << letters_freq[i];
        cout << endl;
    }
    for (int i = 0; i < digits.size(); i += 1)
    {
        digits[i].PrintSeg();
        // digits[i].PrintValues();
        cout << " freq:" << digits_freq[i];
        cout << endl;
    }
    for (int i = 0; i < symbols.size(); i += 1)
    {
        symbols[i].PrintSeg();
        // symbols[i].PrintValues();
        cout << " freq:" << symbols_freq[i];
        cout << endl;
    }
}

bool compareByPretermProb(const PT& a, const PT& b) {
    return a.preterm_prob > b.preterm_prob;  // 降序排序
}

void model::order()
{
    cout << "Training phase 2: Ordering segment values and PTs..." << endl;
    ordered_pts.reserve(preterminals.size());
    for (int id = 0; id < (int)preterminals.size(); id += 1)
    {
        PT pt = preterminals[id];
        pt.preterm_prob = float(preterm_freq[id]) / total_preterm;
        ordered_pts.emplace_back(pt);
    }
    bool swapped;
    cout << "total pts" << ordered_pts.size() << endl;
    std::sort(ordered_pts.begin(), ordered_pts.end(), compareByPretermProb);
    cout << "Ordering letters" << endl;
    // cout << "total letters" << endl;
    // #pragma omp parallel for schedule(dynamic, 1)
    // for (int i = 0; i < letters.size(); i += 1)
    // {
    //     // cout << i << endl;
    //     letters[i].order();
    // }
    // cout << "Ordering digits" << endl;
    // // cout << "total letters" << endl;
    // #pragma omp parallel for schedule(dynamic, 1)
    // for (int i = 0; i < digits.size(); i += 1)
    // {
    //     digits[i].order();
    // }
    // cout << "ordering symbols" << endl;
    // // cout << "total letters" << endl;
    // #pragma omp parallel for schedule(dynamic, 1)
    // for (int i = 0; i < symbols.size(); i += 1)
    // {
    //     symbols[i].order();
    // }
    cout << "Ordering segments" << endl;
    vector<segment *> segments;
    segments.reserve(letters.size() + digits.size() + symbols.size());
    for (segment &seg : letters) segments.emplace_back(&seg);
    for (segment &seg : digits) segments.emplace_back(&seg);
    for (segment &seg : symbols) segments.emplace_back(&seg);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < (int)segments.size(); i += 1) segments[i]->order();
}