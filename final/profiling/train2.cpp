#include "PCFG.h"
#include <fstream>
#include <cctype>
#include <algorithm>

// 这个文件里面的各函数你都不需要完全理解，甚至根本不需要看
// 从学术价值上讲，加速模型的训练过程是一个没什么价值的问题，因为我们一般假定统计学模型的训练成本较低
// 但是，假如你是一个投稿时顶着ddl做实验的倒霉研究生/实习生，提高训练速度就可以大幅节省你的时间了
// 所以如果你愿意，也可以尝试用多线程加速训练过程

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
void model::train(const string &path)
{
    string pw;
    ifstream train_set(path);
    int lines = 0;
    cout<<"Training..."<<endl;
    cout<<"Training phase 1: reading and parsing passwords..."<<endl;
    while (train_set >> pw)
    {
        lines += 1;
        if (lines % 10000 == 0)
        {
            cout <<"Lines processed: "<< lines << endl;
            // 在这里更改读取的训练集口令上限
            if (lines > 3000000)
            {
                break;
            }
        }
        // 读取单个口令之后，就可以将其扔进parse函数进行PT/segment的分割、识别、统计了
        parse(pw);
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
    content.emplace_back(seg);
}

void segment::insert(const string &value)
{
    auto result = values.emplace(value, (int)values.size());
    int id = result.first->second;

    if (result.second)
    {
        freqs.emplace_back(1);
    }
    else
    {
        freqs[id] += 1;
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

void model::parse(const string &pw)
{
    PT pt;
    string curr_part = "";
    int curr_type = 0; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    // 请学会使用这种方式写for循环：for (auto it : iterable)
    // 相信我，以后你会用上的。You're welcome :)
    for (char ch : pw)
    {
        if (isalpha(ch))
        {
            if (curr_type != 1)
            {
                if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    int id = FindDigit(seg);
                    if (id == -1)
                    {
                        id = GetNextDigitsID();
                        digits.emplace_back(seg);
                        digits[id].insert(curr_part);
                        digits_freq.emplace_back(1);
                    }
                    else
                    {
                        digits_freq[id] += 1;
                        digits[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    int id = FindSymbol(seg);
                    if (id == -1)
                    {
                        id = GetNextSymbolsID();
                        symbols.emplace_back(seg);
                        symbols_freq.emplace_back(1);
                        symbols[id].insert(curr_part);
                    }
                    else
                    {
                        symbols_freq[id] += 1;
                        symbols[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 1;
            curr_part += ch;
        }
        else if (isdigit(ch))
        {
            if (curr_type != 2)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    int id = FindLetter(seg);
                    if (id == -1)
                    {
                        id = GetNextLettersID();
                        letters.emplace_back(seg);
                        letters_freq.emplace_back(1);
                        letters[id].insert(curr_part);
                    }
                    else
                    {
                        letters_freq[id] += 1;
                        letters[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    int id = FindSymbol(seg);
                    if (id == -1)
                    {
                        id = GetNextSymbolsID();
                        symbols.emplace_back(seg);
                        symbols_freq.emplace_back(1);
                        symbols[id].insert(curr_part);
                    }
                    else
                    {
                        symbols_freq[id] += 1;
                        symbols[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 2;
            curr_part += ch;
        }
        else
        {
            if (curr_type != 3)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    int id = FindLetter(seg);
                    if (id == -1)
                    {
                        id = GetNextLettersID();
                        letters.emplace_back(seg);
                        letters_freq.emplace_back(1);
                        letters[id].insert(curr_part);
                    }
                    else
                    {
                        letters_freq[id] += 1;
                        letters[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    int id = FindDigit(seg);
                    if (id == -1)
                    {
                        id = GetNextDigitsID();
                        digits.emplace_back(seg);
                        digits_freq.emplace_back(1);
                        digits[id].insert(curr_part);
                    }
                    else
                    {
                        digits_freq[id] += 1;
                        digits[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 3;
            curr_part += ch;
        }
    }
    if (!curr_part.empty())
    {
        if (curr_type == 1)
        {
            segment seg(curr_type, curr_part.length());
            int id = FindLetter(seg);
            if (id == -1)
            {
                id = GetNextLettersID();
                letters.emplace_back(seg);
                letters_freq.emplace_back(1);
                letters[id].insert(curr_part);
            }
            else
            {
                letters_freq[id] += 1;
                letters[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
        else if (curr_type == 2)
        {
            segment seg(curr_type, curr_part.length());
            int id = FindDigit(seg);
            if (id == -1)
            {
                id = GetNextDigitsID();
                digits.emplace_back(seg);
                digits_freq.emplace_back(1);
                digits[id].insert(curr_part);
            }
            else
            {
                digits_freq[id] += 1;
                digits[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
        else
        {
            segment seg(curr_type, curr_part.length());
            int id = FindSymbol(seg);
            if (id == -1)
            {
                id = GetNextSymbolsID();
                symbols.emplace_back(seg);
                symbols_freq.emplace_back(1);
                symbols[id].insert(curr_part);
            }
            else
            {
                symbols_freq[id] += 1;
                symbols[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
    }
    // pt.PrintPT();
    // cout<<endl;
    // cout << FindPT(pt) << endl;
    total_preterm += 1;

    string pt_key = GetPTKey(pt);
    auto iter = preterm_ids.find(pt_key);

    if (iter == preterm_ids.end())
    {
        for (int i = 0; i < (int)pt.content.size(); i += 1)
        {
            pt.curr_indices.emplace_back(0);
        }

        int id = GetNextPretermID();

        preterminals.emplace_back(pt);
        preterm_ids.emplace(pt_key, id);
        preterm_freq.emplace_back(1);
    }
    else
    {
        preterm_freq[iter->second] += 1;
    }
}

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