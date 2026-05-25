// my_multithread.h
#ifndef MY_MULTITHREAD_H
#define MY_MULTITHREAD_H

#include "my_simd.h"
#include <pthread.h>
#include <omp.h>
#include <queue>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <set>
#include <functional>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>
// 在文件开头（全局位置）定义对齐的堆包装器
template<typename T>
struct alignas(64) AlignedHeap {
    std::priority_queue<T> heap;
};
// 辅助函数：计算一个查询的召回率（res 是搜索返回的优先队列，gt 是该查询的真实 top‑k 索引）
inline float compute_recall(std::priority_queue<std::pair<float, uint32_t>> res,
                            const int* gt, size_t k) {
    std::set<uint32_t> gtset(gt, gt + k);
    size_t hit = 0;
    while (!res.empty()) {
        uint32_t idx = res.top().second;
        if (gtset.find(idx) != gtset.end()) ++hit;
        res.pop();
    }
    return static_cast<float>(hit) / k;
}

// ===== Flat‑SIMD 多线程并行 =====
// 思路：将 base 向量集连续分块，每个线程处理一块，维护局部优先队列（大小 = local_p），最后主线程归并。

struct FlatThreadParam {
    const float* base;
    const float* query;
    size_t start;
    size_t end;
    size_t vecdim;
    size_t local_p;   // 每个线程保留的局部候选数（≥k），用于 trade‑off
    std::priority_queue<std::pair<float, uint32_t>> local_heap;
};

void* flat_worker_pthread(void* arg) {
    auto* p = (FlatThreadParam*)arg;
    for (size_t i = p->start; i < p->end; ++i) {
        float dis = inner_product_neon_optB(p->base + i * p->vecdim, p->query, p->vecdim);
        if (p->local_heap.size() < p->local_p) {
            p->local_heap.push({dis, i});
        } else if (dis < p->local_heap.top().first) {
            p->local_heap.push({dis, i});
            p->local_heap.pop();
        }
    }
    return nullptr;
}

// Pthread 版本（带 local_p 参数）
inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pthread(
    float* base, const float* query, size_t base_num, size_t vecdim,
    size_t k, size_t local_p, int num_threads) {
    std::vector<pthread_t> threads(num_threads);
    std::vector<FlatThreadParam> params(num_threads);
    size_t chunk = (base_num + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = std::min((t + 1) * chunk, base_num);
        params[t] = {base, query, start, end, vecdim, local_p, {}};
        pthread_create(&threads[t], nullptr, flat_worker_pthread, &params[t]);
    }
    for (int t = 0; t < num_threads; ++t) pthread_join(threads[t], nullptr);

    // 归并所有局部堆，取全局 top‑k
    std::priority_queue<std::pair<float, uint32_t>> global_heap;
    for (int t = 0; t < num_threads; ++t) {
        auto& local = params[t].local_heap;
        while (!local.empty()) {
            auto item = local.top(); local.pop();
            if (global_heap.size() < k) global_heap.push(item);
            else if (item.first < global_heap.top().first) {
                global_heap.push(item); global_heap.pop();
            }
        }
    }
    return global_heap;
}

// 兼容旧接口：local_p 默认等于 k
inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pthread(
    float* base, const float* query, size_t base_num, size_t vecdim,
    size_t k, int num_threads) {
    return flat_search_pthread(base, query, base_num, vecdim, k, k, num_threads);
}

// OpenMP 版本（带 local_p）
inline std::priority_queue<std::pair<float, uint32_t>> flat_search_omp(
    float* base, const float* query, size_t base_num, size_t vecdim,
    size_t k, size_t local_p) {
    int num_threads = omp_get_max_threads();
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_heaps(num_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < base_num; ++i) {
            float dis = inner_product_neon_optB(base + i * vecdim, query, vecdim);
            auto& heap = local_heaps[tid];
            if (heap.size() < local_p) heap.push({dis, i});
            else if (dis < heap.top().first) {
                heap.push({dis, i}); heap.pop();
            }
        }
    }
    // 归并
    std::priority_queue<std::pair<float, uint32_t>> global_heap;
    for (auto& heap : local_heaps) {
        while (!heap.empty()) {
            auto item = heap.top(); heap.pop();
            if (global_heap.size() < k) global_heap.push(item);
            else if (item.first < global_heap.top().first) {
                global_heap.push(item); global_heap.pop();
            }
        }
    }
    return global_heap;
}

// 兼容旧接口
inline std::priority_queue<std::pair<float, uint32_t>> flat_search_omp(
    float* base, const float* query, size_t base_num, size_t vecdim,
    size_t k) {
    return flat_search_omp(base, query, base_num, vecdim, k, k);
}

// ===== PQ‑SIMD 多线程优化 =====
// 查询阶段分为 LUT 构建（子空间并行）和查表累加（base 分块并行）
// 这里实现 Pthread 和 OpenMP 两个版本。

struct LutBuildParam {
    const float* sub_q;
    const float* codebook_sub;
    float* lut_sub;
    int sub_dim;
};

void* build_lut_worker(void* arg) {
    auto* p = (LutBuildParam*)arg;
    for (int c = 0; c < 256; ++c) {
        const float* center = p->codebook_sub + c * p->sub_dim;
        p->lut_sub[c] = inner_product_neon_optB(center, p->sub_q, p->sub_dim);
    }
    return nullptr;
}

struct ScanParam {
    const std::vector<std::vector<uint8_t>>& codes;
    const std::vector<std::vector<float>>& lut;
    size_t start;
    size_t end;
    int m;
    std::vector<std::pair<float, uint32_t>> local_approx;
};

void* scan_worker(void* arg) {
    auto* p = (ScanParam*)arg;
    const size_t BLOCK = 8;
    p->local_approx.reserve(p->end - p->start);
    for (size_t i = p->start; i < p->end; i += BLOCK) {
        size_t valid = std::min(BLOCK, p->end - i);
        float dist[BLOCK] = {0.0f};
        for (int s = 0; s < p->m; ++s) {
            const uint8_t* code_ptr = p->codes[s].data() + i;
            const float* lut_s = p->lut[s].data();
            __builtin_prefetch(code_ptr + BLOCK, 0, 1);
            if (valid == BLOCK) {
                dist[0] += lut_s[code_ptr[0]];
                dist[1] += lut_s[code_ptr[1]];
                dist[2] += lut_s[code_ptr[2]];
                dist[3] += lut_s[code_ptr[3]];
                dist[4] += lut_s[code_ptr[4]];
                dist[5] += lut_s[code_ptr[5]];
                dist[6] += lut_s[code_ptr[6]];
                dist[7] += lut_s[code_ptr[7]];
            } else {
                for (size_t j = 0; j < valid; ++j) {
                    dist[j] += lut_s[code_ptr[j]];
                }
            }
        }
        for (size_t j = 0; j < valid; ++j) {
            p->local_approx.emplace_back(dist[j], static_cast<uint32_t>(i + j));
        }
    }
    return nullptr;
}

// Pthread 版本：LUT 构建子空间并行，查表累加 base 分块并行
inline std::priority_queue<std::pair<float, uint32_t>> pq_search_pthread(
    float* base,
    const float* query,
    const std::vector<std::vector<float>>& codebook,
    const std::vector<std::vector<uint8_t>>& codes,
    size_t base_num,
    size_t vecdim,
    size_t k,
    size_t p,
    int num_threads)
{
    int m = codebook.size();
    int sub_dim = vecdim / m;

    // LUT 构建（子空间并行）
    std::vector<std::vector<float>> lut(m, std::vector<float>(256));
    std::vector<pthread_t> lut_threads(m);
    std::vector<LutBuildParam> lut_params(m);
    for (int s = 0; s < m; ++s) {
        const float* sub_q = query + s * sub_dim;
        const float* cb_sub = codebook[s].data();
        float* lut_sub = lut[s].data();
        lut_params[s] = {sub_q, cb_sub, lut_sub, sub_dim};
        pthread_create(&lut_threads[s], nullptr, build_lut_worker, &lut_params[s]);
    }
    for (int s = 0; s < m; ++s) pthread_join(lut_threads[s], nullptr);

    // 查表累加（base 分块并行）
    std::vector<pthread_t> scan_threads(num_threads);
    std::vector<ScanParam> scan_params;
    scan_params.reserve(num_threads);
    size_t chunk = (base_num + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = std::min((t + 1) * chunk, base_num);
        scan_params.emplace_back(ScanParam{codes, lut, start, end, m, {}});
        pthread_create(&scan_threads[t], nullptr, scan_worker, &scan_params.back());
    }
    for (int t = 0; t < num_threads; ++t) pthread_join(scan_threads[t], nullptr);

    // 合并局部近似距离，取前 p 个候选
    std::vector<std::pair<float, uint32_t>> global_approx;
    global_approx.reserve(base_num);
    for (const auto& param : scan_params) {
        global_approx.insert(global_approx.end(), param.local_approx.begin(), param.local_approx.end());
    }
    if (global_approx.size() > p) {
        std::partial_sort(global_approx.begin(), global_approx.begin() + p, global_approx.end());
        global_approx.resize(p);
    } else {
        std::sort(global_approx.begin(), global_approx.end());
    }

    // 精排：精确浮点距离
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (const auto& item : global_approx) {
        float dis = inner_product_neon_optB(base + item.second * vecdim, query, vecdim);
        if (q.size() < k) q.push({dis, item.second});
        else if (dis < q.top().first) { q.push({dis, item.second}); q.pop(); }
    }
    return q;
}

// OpenMP 版本：LUT 构建子空间并行，查表累加 base 分块并行
inline std::priority_queue<std::pair<float, uint32_t>> pq_search_omp(
    float* base,
    const float* query,
    const std::vector<std::vector<float>>& codebook,
    const std::vector<std::vector<uint8_t>>& codes,
    size_t base_num,
    size_t vecdim,
    size_t k,
    size_t p)
{
    int m = codebook.size();
    int sub_dim = vecdim / m;

    // LUT 构建
    std::vector<std::vector<float>> lut(m, std::vector<float>(256));
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < m; ++s) {
        const float* sub_q = query + s * sub_dim;
        const float* cb_sub = codebook[s].data();
        float* lut_s = lut[s].data();
        for (int c = 0; c < 256; ++c) {
            const float* center = cb_sub + c * sub_dim;
            lut_s[c] = inner_product_neon_optB(center, sub_q, sub_dim);
        }
    }

    // 查表累加
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<std::pair<float, uint32_t>>> local_approx(num_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t chunk = (base_num + num_threads - 1) / num_threads;
        size_t start = tid * chunk;
        size_t end = std::min((tid + 1) * chunk, base_num);
        local_approx[tid].reserve(end - start);
        const size_t BLOCK = 8;
        for (size_t i = start; i < end; i += BLOCK) {
            size_t valid = std::min(BLOCK, end - i);
            float dist[BLOCK] = {0.0f};
            for (int s = 0; s < m; ++s) {
                const uint8_t* code_ptr = codes[s].data() + i;
                const float* lut_s = lut[s].data();
                if (valid == BLOCK) {
                    dist[0] += lut_s[code_ptr[0]];
                    dist[1] += lut_s[code_ptr[1]];
                    dist[2] += lut_s[code_ptr[2]];
                    dist[3] += lut_s[code_ptr[3]];
                    dist[4] += lut_s[code_ptr[4]];
                    dist[5] += lut_s[code_ptr[5]];
                    dist[6] += lut_s[code_ptr[6]];
                    dist[7] += lut_s[code_ptr[7]];
                } else {
                    for (size_t j = 0; j < valid; ++j) {
                        dist[j] += lut_s[code_ptr[j]];
                    }
                }
            }
            for (size_t j = 0; j < valid; ++j) {
                local_approx[tid].emplace_back(dist[j], static_cast<uint32_t>(i + j));
            }
        }
    }

    // 合并局部近似距离
    std::vector<std::pair<float, uint32_t>> global_approx;
    global_approx.reserve(base_num);
    for (const auto& vec : local_approx) {
        global_approx.insert(global_approx.end(), vec.begin(), vec.end());
    }
    if (global_approx.size() > p) {
        std::partial_sort(global_approx.begin(), global_approx.begin() + p, global_approx.end());
        global_approx.resize(p);
    } else {
        std::sort(global_approx.begin(), global_approx.end());
    }

    // 精排
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (const auto& item : global_approx) {
        float dis = inner_product_neon_optB(base + item.second * vecdim, query, vecdim);
        if (q.size() < k) q.push({dis, item.second});
        else if (dis < q.top().first) { q.push({dis, item.second}); q.pop(); }
    }
    return q;
}

// ===== IVF‑SIMD 单线程 baseline 及多线程优化 =====
// IVF 查询：先对 256 个簇中心做粗排（SIMD），再对选中的簇做精确扫描。

inline std::priority_queue<std::pair<float, uint32_t>> ivf_search_simd(
    float* base, const float* query,
    const std::vector<float>& centroids,
    const std::vector<std::vector<uint32_t>>& lists,
    size_t vecdim, size_t k, int nprobe)
{
    int n_centroids = centroids.size() / vecdim;
    std::vector<std::pair<float, int>> cent_dist(n_centroids);
    for (int c = 0; c < n_centroids; ++c) {
        const float* center = &centroids[c * vecdim];
        cent_dist[c] = {inner_product_neon_optB(center, query, vecdim), c};
    }
    std::partial_sort(cent_dist.begin(), cent_dist.begin() + nprobe, cent_dist.end());

    std::priority_queue<std::pair<float, uint32_t>> heap;
    for (int i = 0; i < nprobe; ++i) {
        int cid = cent_dist[i].second;
        for (uint32_t idx : lists[cid]) {
            float dis = inner_product_neon_optB(base + idx * vecdim, query, vecdim);
            if (heap.size() < k) heap.push({dis, idx});
            else if (dis < heap.top().first) { heap.push({dis, idx}); heap.pop(); }
        }
    }
    return heap;
}

// Pthread 版本：静态划分选中的簇，每个线程处理多个簇。
struct IVFThreadParam {
    float* base;
    const float* query;
    size_t vecdim;
    size_t k;
    const std::vector<std::vector<uint32_t>>* lists;
    std::vector<int> cluster_ids;
    std::priority_queue<std::pair<float, uint32_t>> local_heap;
};

void* ivf_worker_pthread(void* arg) {
    auto* p = (IVFThreadParam*)arg;
    for (int cid : p->cluster_ids) {
        for (uint32_t idx : (*(p->lists))[cid]) {
            float dis = inner_product_neon_optB(p->base + idx * p->vecdim, p->query, p->vecdim);
            auto& heap = p->local_heap;
            if (heap.size() < p->k) heap.push({dis, idx});
            else if (dis < heap.top().first) { heap.push({dis, idx}); heap.pop(); }
        }
    }
    return nullptr;
}

inline std::priority_queue<std::pair<float, uint32_t>> ivf_search_pthread(
    float* base, const float* query,
    const std::vector<float>& centroids,
    const std::vector<std::vector<uint32_t>>& lists,
    size_t vecdim, size_t k, int nprobe, int num_threads)
{
    // 粗排串行
    int n_centroids = centroids.size() / vecdim;
    std::vector<std::pair<float, int>> cent_dist(n_centroids);
    for (int c = 0; c < n_centroids; ++c) {
        const float* center = &centroids[c * vecdim];
        cent_dist[c] = {inner_product_neon_optB(center, query, vecdim), c};
    }
    std::partial_sort(cent_dist.begin(), cent_dist.begin() + nprobe, cent_dist.end());

    std::vector<int> selected;
    for (int i = 0; i < nprobe; ++i) selected.push_back(cent_dist[i].second);

    // 静态划分选中的簇
    std::vector<pthread_t> threads(num_threads);
    std::vector<IVFThreadParam> params(num_threads);
    int total = selected.size();
    for (int t = 0; t < num_threads; ++t) {
        int start = t * total / num_threads;
        int end = (t + 1) * total / num_threads;
        params[t].base = base;
        params[t].query = query;
        params[t].vecdim = vecdim;
        params[t].k = k;
        params[t].lists = &lists;
        params[t].cluster_ids.assign(selected.begin() + start, selected.begin() + end);
        pthread_create(&threads[t], nullptr, ivf_worker_pthread, &params[t]);
    }
    for (int t = 0; t < num_threads; ++t) pthread_join(threads[t], nullptr);

    // 归并
    std::priority_queue<std::pair<float, uint32_t>> global_heap;
    for (int t = 0; t < num_threads; ++t) {
        auto& heap = params[t].local_heap;
        while (!heap.empty()) {
            auto item = heap.top(); heap.pop();
            if (global_heap.size() < k) global_heap.push(item);
            else if (item.first < global_heap.top().first) { global_heap.push(item); global_heap.pop(); }
        }
    }
    return global_heap;
}

//OpenMP 版本：使用 dynamic schedule 分配每个簇，自动负载均衡
inline std::priority_queue<std::pair<float, uint32_t>> ivf_search_omp(
    float* base, const float* query,
    const std::vector<float>& centroids,
    const std::vector<std::vector<uint32_t>>& lists,
    size_t vecdim, size_t k, int nprobe)
{
    int n_centroids = centroids.size() / vecdim;
    std::vector<std::pair<float, int>> cent_dist(n_centroids);
    for (int c = 0; c < n_centroids; ++c) {
        const float* center = &centroids[c * vecdim];
        cent_dist[c] = {inner_product_neon_optB(center, query, vecdim), c};
    }
    std::partial_sort(cent_dist.begin(), cent_dist.begin() + nprobe, cent_dist.end());

    std::vector<int> selected;
    for (int i = 0; i < nprobe; ++i) selected.push_back(cent_dist[i].second);

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_heaps;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp single
        local_heaps.resize(omp_get_num_threads());
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < nprobe; ++i) {
            int cid = selected[i];
            for (uint32_t idx : lists[cid]) {
                float dis = inner_product_neon_optB(base + idx * vecdim, query, vecdim);
                auto& heap = local_heaps[tid];
                if (heap.size() < k) heap.push({dis, idx});
                else if (dis < heap.top().first) { heap.push({dis, idx}); heap.pop(); }
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> global_heap;
    for (auto& heap : local_heaps) {
        while (!heap.empty()) {
            auto item = heap.top(); heap.pop();
            if (global_heap.size() < k) global_heap.push(item);
            else if (item.first < global_heap.top().first) { global_heap.push(item); global_heap.pop(); }
        }
    }
    return global_heap;
}

// 对齐版本的 IVF-SIMD OpenMP 并行
inline std::priority_queue<std::pair<float, uint32_t>> ivf_search_omp_aligned(
    float* base, const float* query,
    const std::vector<float>& centroids,
    const std::vector<std::vector<uint32_t>>& lists,
    size_t vecdim, size_t k, int nprobe)
{
    int n_centroids = centroids.size() / vecdim;
    std::vector<std::pair<float, int>> cent_dist(n_centroids);
    for (int c = 0; c < n_centroids; ++c) {
        const float* center = &centroids[c * vecdim];
        cent_dist[c] = {inner_product_neon_optB(center, query, vecdim), c};
    }
    std::partial_sort(cent_dist.begin(), cent_dist.begin() + nprobe, cent_dist.end());

    std::vector<int> selected;
    for (int i = 0; i < nprobe; ++i) selected.push_back(cent_dist[i].second);

    // 使用对齐的堆数组
    std::vector<AlignedHeap<std::pair<float, uint32_t>>> local_heaps;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp single
        local_heaps.resize(omp_get_num_threads());
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < nprobe; ++i) {
            int cid = selected[i];
            for (uint32_t idx : lists[cid]) {
                float dis = inner_product_neon_optB(base + idx * vecdim, query, vecdim);
                auto& heap = local_heaps[tid].heap;
                if (heap.size() < k) heap.push({dis, idx});
                else if (dis < heap.top().first) { heap.push({dis, idx}); heap.pop(); }
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> global_heap;
    for (auto& heap_wrapper : local_heaps) {
        auto& heap = heap_wrapper.heap;
        while (!heap.empty()) {
            auto item = heap.top(); heap.pop();
            if (global_heap.size() < k) global_heap.push(item);
            else if (item.first < global_heap.top().first) {
                global_heap.push(item); global_heap.pop();
            }
        }
    }
    return global_heap;
}
// ===== IVF‑PQ 单线程 baseline 及多线程优化 =====
// 单线程版本：完整的 IVF+PQ 查询流程。
std::priority_queue<std::pair<float, uint32_t>> ivf_pq_search_singlethread(
    float* base,
    const float* query,
    const std::vector<std::vector<float>>& pq_codebook,
    const std::vector<std::vector<uint8_t>>& pq_codes,
    const std::vector<float>& ivf_centroids,
    const std::vector<std::vector<uint32_t>>& ivf_lists,
    size_t vecdim,
    size_t k,
    size_t p_pq,
    int nprobe)
{
    int n_centroids = ivf_centroids.size() / vecdim;
    int m = pq_codebook.size();
    int sub_dim = vecdim / m;

    // 粗排：计算所有 IV F中心距离
    std::vector<std::pair<float, int>> cent_dist(n_centroids);
    for (int c = 0; c < n_centroids; ++c) {
        const float* center = &ivf_centroids[c * vecdim];
        cent_dist[c] = {inner_product_neon_optB(center, query, vecdim), c};
    }
    std::partial_sort(cent_dist.begin(), cent_dist.begin() + nprobe, cent_dist.end());

    // 收集候选索引（选中簇中所有 base 向量的索引）
    std::vector<uint32_t> cand_indices;
    for (int i = 0; i < nprobe; ++i) {
        int cid = cent_dist[i].second;
        cand_indices.insert(cand_indices.end(), ivf_lists[cid].begin(), ivf_lists[cid].end());
    }
    if (cand_indices.empty()) return {};

    // 构建 PQ LUT
    std::vector<std::vector<float>> lut(m, std::vector<float>(256));
    for (int s = 0; s < m; ++s) {
        const float* sub_q = query + s * sub_dim;
        const float* cb = pq_codebook[s].data();
        for (int c = 0; c < 256; ++c) {
            lut[s][c] = inner_product_neon_optB(cb + c * sub_dim, sub_q, sub_dim);
        }
    }

    // 查表累加得到近似距离，取前 p_pq 个候选
    std::vector<std::pair<float, uint32_t>> approx;
    approx.reserve(cand_indices.size());
    for (uint32_t base_idx : cand_indices) {
        float dist = 0.0f;
        for (int s = 0; s < m; ++s) {
            dist += lut[s][pq_codes[s][base_idx]];
        }
        approx.emplace_back(dist, base_idx);
    }
    if (approx.size() > p_pq) {
        std::partial_sort(approx.begin(), approx.begin() + p_pq, approx.end());
        approx.resize(p_pq);
    } else {
        std::sort(approx.begin(), approx.end());
    }

    // 精排：精确内积
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (auto& item : approx) {
        float dis = inner_product_neon_optB(base + item.second * vecdim, query, vecdim);
        if (q.size() < k) q.push({dis, item.second});
        else if (dis < q.top().first) { q.push({dis, item.second}); q.pop(); }
    }
    return q;
}

// Pthread 多线程 IVF‑PQ：主线程构建 LUT，静态划分候选索引给各线程。
struct IVF_PQ_ThreadParam {
    float* base;
    const float* query;
    const std::vector<std::vector<float>>* codebook;
    const std::vector<std::vector<uint8_t>>* codes;
    const std::vector<std::vector<uint32_t>>* ivf_lists;
    const std::vector<std::vector<float>>* lut;   // 共享的 LUT
    std::vector<uint32_t> cand_indices;
    size_t vecdim;
    size_t k;
    size_t p_pq;
    int m;
    int sub_dim;
    std::priority_queue<std::pair<float, uint32_t>> local_heap;
};

void* ivf_pq_worker_pthread(void* arg) {
    auto* p = (IVF_PQ_ThreadParam*)arg;
    const auto& lut = *(p->lut);
    std::vector<std::pair<float, uint32_t>> approx;
    approx.reserve(p->cand_indices.size());

    // 查表累加
    for (uint32_t base_idx : p->cand_indices) {
        float dist = 0.0f;
        for (int s = 0; s < p->m; ++s) {
            dist += lut[s][(*p->codes)[s][base_idx]];
        }
        approx.emplace_back(dist, base_idx);
    }

    // 取局部 top‑p_pq 并精排
    if (approx.size() > p->p_pq) {
        std::partial_sort(approx.begin(), approx.begin() + p->p_pq, approx.end());
        approx.resize(p->p_pq);
    } else {
        std::sort(approx.begin(), approx.end());
    }
    for (auto& item : approx) {
        float dis = inner_product_neon_optB(p->base + item.second * p->vecdim, p->query, p->vecdim);
        auto& heap = p->local_heap;
        if (heap.size() < p->k) heap.push({dis, item.second});
        else if (dis < heap.top().first) { heap.push({dis, item.second}); heap.pop(); }
    }
    return nullptr;
}

inline std::priority_queue<std::pair<float, uint32_t>> ivf_pq_search_pthread(
    float* base,
    const float* query,
    const std::vector<std::vector<float>>& pq_codebook,
    const std::vector<std::vector<uint8_t>>& pq_codes,
    const std::vector<float>& ivf_centroids,
    const std::vector<std::vector<uint32_t>>& ivf_lists,
    size_t base_num,
    size_t vecdim,
    size_t k,
    size_t p_pq,
    int nprobe,
    int num_threads)
{
    int n_centroids = ivf_centroids.size() / vecdim;
    int m = pq_codebook.size();
    int sub_dim = vecdim / m;
    // 线程数为1时直接用单线程 baseline，避免额外开销
    if (num_threads == 1) {
        return ivf_pq_search_singlethread(base, query, pq_codebook, pq_codes,
                                          ivf_centroids, ivf_lists,
                                          vecdim, k, p_pq, nprobe);
    }

    // 粗排（串行）
    std::vector<std::pair<float, int>> cent_dist(n_centroids);
    for (int c = 0; c < n_centroids; ++c) {
        const float* center = &ivf_centroids[c * vecdim];
        cent_dist[c] = {inner_product_neon_optB(center, query, vecdim), c};
    }
    std::partial_sort(cent_dist.begin(), cent_dist.begin() + nprobe, cent_dist.end());

    // 收集候选索引
    std::vector<uint32_t> cand_indices;
    for (int i = 0; i < nprobe; ++i) {
        int cid = cent_dist[i].second;
        cand_indices.insert(cand_indices.end(), ivf_lists[cid].begin(), ivf_lists[cid].end());
    }
    if (cand_indices.empty()) return {};

    // 主线程构建 LUT（一次）
    std::vector<std::vector<float>> lut(m, std::vector<float>(256));
    for (int s = 0; s < m; ++s) {
        const float* sub_q = query + s * sub_dim;
        const float* cb = pq_codebook[s].data();
        for (int c = 0; c < 256; ++c) {
            lut[s][c] = inner_product_neon_optB(cb + c * sub_dim, sub_q, sub_dim);
        }
    }

    // 静态划分候选索引
    std::vector<pthread_t> threads(num_threads);
    std::vector<IVF_PQ_ThreadParam> params(num_threads);
    size_t chunk = (cand_indices.size() + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = std::min((t+1)*chunk, cand_indices.size());
        params[t].base = base;
        params[t].query = query;
        params[t].codebook = &pq_codebook;
        params[t].codes = &pq_codes;
        params[t].ivf_lists = &ivf_lists;
        params[t].cand_indices.assign(cand_indices.begin() + start, cand_indices.begin() + end);
        params[t].vecdim = vecdim;
        params[t].k = k;
        params[t].p_pq = p_pq;
        params[t].m = m;
        params[t].sub_dim = sub_dim;
        params[t].lut = &lut;
        pthread_create(&threads[t], nullptr, ivf_pq_worker_pthread, &params[t]);
    }
    for (int t = 0; t < num_threads; ++t) pthread_join(threads[t], nullptr);

    // 归并局部堆
    std::priority_queue<std::pair<float, uint32_t>> global_heap;
    for (int t = 0; t < num_threads; ++t) {
        auto& heap = params[t].local_heap;
        while (!heap.empty()) {
            auto item = heap.top(); heap.pop();
            if (global_heap.size() < k) global_heap.push(item);
            else if (item.first < global_heap.top().first) {
                global_heap.push(item); global_heap.pop();
            }
        }
    }
    return global_heap;
}

// OpenMP 多线程 IVF‑PQ：LUT 并行构建，候选索引动态调度
inline std::priority_queue<std::pair<float, uint32_t>> ivf_pq_search_omp(
    float* base,
    const float* query,
    const std::vector<std::vector<float>>& pq_codebook,
    const std::vector<std::vector<uint8_t>>& pq_codes,
    const std::vector<float>& ivf_centroids,
    const std::vector<std::vector<uint32_t>>& ivf_lists,
    size_t base_num,
    size_t vecdim,
    size_t k,
    size_t p_pq,
    int nprobe)
{
    int n_centroids = ivf_centroids.size() / vecdim;
    int m = pq_codebook.size();
    int sub_dim = vecdim / m;

    // 粗排（串行）
    std::vector<std::pair<float, int>> cent_dist(n_centroids);
    for (int c = 0; c < n_centroids; ++c) {
        const float* center = &ivf_centroids[c * vecdim];
        cent_dist[c] = {inner_product_neon_optB(center, query, vecdim), c};
    }
    std::partial_sort(cent_dist.begin(), cent_dist.begin() + nprobe, cent_dist.end());

    std::vector<uint32_t> cand_indices;
    for (int i = 0; i < nprobe; ++i) {
        int cid = cent_dist[i].second;
        cand_indices.insert(cand_indices.end(), ivf_lists[cid].begin(), ivf_lists[cid].end());
    }
    if (cand_indices.empty()) return {};

    // 并行构建 LUT（子空间级并行）
    std::vector<std::vector<float>> lut(m, std::vector<float>(256));
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < m; ++s) {
        const float* sub_q = query + s * sub_dim;
        const float* cb = pq_codebook[s].data();
        for (int c = 0; c < 256; ++c) {
            lut[s][c] = inner_product_neon_optB(cb + c * sub_dim, sub_q, sub_dim);
        }
    }

    // 并行处理候选索引（手动分块 + 动态调度避免静态划分不均）
    int num_threads = omp_get_max_threads();
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_heaps(num_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t chunk_size = (cand_indices.size() + num_threads - 1) / num_threads;
        size_t start = tid * chunk_size;
        size_t end = std::min(start + chunk_size, cand_indices.size());

        if (start < end) {
            std::vector<std::pair<float, uint32_t>> local_approx;
            local_approx.reserve(end - start);
            // 查表累加
            for (size_t idx = start; idx < end; ++idx) {
                uint32_t base_idx = cand_indices[idx];
                float dist = 0.0f;
                for (int s = 0; s < m; ++s) {
                    dist += lut[s][pq_codes[s][base_idx]];
                }
                local_approx.emplace_back(dist, base_idx);
            }
            // 取局部 top‑p_pq
            if (local_approx.size() > p_pq) {
                std::partial_sort(local_approx.begin(), local_approx.begin() + p_pq, local_approx.end());
                local_approx.resize(p_pq);
            } else {
                std::sort(local_approx.begin(), local_approx.end());
            }
            // 精排，存入局部堆
            auto& heap = local_heaps[tid];
            for (auto& item : local_approx) {
                float dis = inner_product_neon_optB(base + item.second * vecdim, query, vecdim);
                if (heap.size() < k) heap.push({dis, item.second});
                else if (dis < heap.top().first) { heap.push({dis, item.second}); heap.pop(); }
            }
        }
    }

    // 归并所有局部堆
    std::priority_queue<std::pair<float, uint32_t>> global_heap;
    for (auto& heap : local_heaps) {
        while (!heap.empty()) {
            auto item = heap.top(); heap.pop();
            if (global_heap.size() < k) global_heap.push(item);
            else if (item.first < global_heap.top().first) {
                global_heap.push(item); global_heap.pop();
            }
        }
    }
    return global_heap;
}

// 批量延迟测量函数（预热 + 正式计时）
double measure_latency_batch(
    std::function<std::priority_queue<std::pair<float, uint32_t>>(const float*)> search_func,
    const std::vector<const float*>& queries,
    size_t warmup_queries = 100)
{
    // 预热
    for (size_t i = 0; i < warmup_queries; ++i) {
        volatile auto res = search_func(queries[i % queries.size()]);
        (void)res;
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (const float* q : queries) {
        volatile auto res = search_func(q);
        (void)res;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - start).count();
    return us / queries.size();
}

// 召回率测量（使用单线程 baseline）
double measure_recall(
    float* base,
    const std::vector<const float*>& queries,
    const std::vector<const int*>& gt_ptrs,
    size_t base_number,
    size_t vecdim,
    size_t k,
    const std::vector<std::vector<float>>& pq_codebook,
    const std::vector<std::vector<uint8_t>>& pq_codes,
    const std::vector<float>& ivf_centroids,
    const std::vector<std::vector<uint32_t>>& ivf_lists,
    int nprobe,
    size_t p_pq)
{
    double total_rec = 0.0;
    for (size_t i = 0; i < queries.size(); ++i) {
        auto res = ivf_pq_search_singlethread(base, queries[i], pq_codebook, pq_codes,
                                              ivf_centroids, ivf_lists,
                                              vecdim, k, p_pq, nprobe);
        total_rec += compute_recall(std::move(res), gt_ptrs[i], k);
    }
    return total_rec / queries.size();
}

// 测试 Pthread 版本 IVF‑PQ 扩展性
void test_ivf_pq_pthread_scale(
    float* base,
    const std::vector<const float*>& queries,
    size_t base_number,
    size_t vecdim,
    size_t k,
    const std::vector<std::vector<float>>& pq_codebook,
    const std::vector<std::vector<uint8_t>>& pq_codes,
    const std::vector<float>& ivf_centroids,
    const std::vector<std::vector<uint32_t>>& ivf_lists,
    int nprobe,
    size_t p_pq,
    const std::vector<int>& thread_counts)
{
    std::cout << "\n=== IVF-PQ Pthread scaling (nprobe=" << nprobe
              << ", p_pq=" << p_pq << ") ===" << std::endl;
    std::cout << "Threads\tLatency(us)\tSpeedup" << std::endl;

    double base_lat = measure_latency_batch(
        [&](const float* q) {
            return ivf_pq_search_singlethread(base, q, pq_codebook, pq_codes,
                                              ivf_centroids, ivf_lists,
                                              vecdim, k, p_pq, nprobe);
        },
        queries
    );
    std::cout << "baseline\t" << base_lat << "\t1.00" << std::endl;

    for (int num_threads : thread_counts) {
        double lat = measure_latency_batch(
            [&](const float* q) {
                return ivf_pq_search_pthread(base, q, pq_codebook, pq_codes,
                                             ivf_centroids, ivf_lists,
                                             base_number, vecdim, k, p_pq, nprobe, num_threads);
            },
            queries
        );
        double speedup = base_lat / lat;
        std::cout << num_threads << "\t" << lat << "\t" << speedup << std::endl;
    }
}

// 测试 OpenMP 版本 IVF‑PQ 扩展性
void test_ivf_pq_omp_scale(
    float* base,
    const std::vector<const float*>& queries,
    size_t base_number,
    size_t vecdim,
    size_t k,
    const std::vector<std::vector<float>>& pq_codebook,
    const std::vector<std::vector<uint8_t>>& pq_codes,
    const std::vector<float>& ivf_centroids,
    const std::vector<std::vector<uint32_t>>& ivf_lists,
    int nprobe,
    size_t p_pq,
    const std::vector<int>& thread_counts)
{
    std::cout << "\n=== IVF-PQ OpenMP scaling (nprobe=" << nprobe
              << ", p_pq=" << p_pq << ") ===" << std::endl;
    std::cout << "Threads\tLatency(us)\tSpeedup" << std::endl;

    double base_lat = measure_latency_batch(
        [&](const float* q) {
            return ivf_pq_search_singlethread(base, q, pq_codebook, pq_codes,
                                              ivf_centroids, ivf_lists,
                                              vecdim, k, p_pq, nprobe);
        },
        queries
    );
    std::cout << "baseline\t" << base_lat << "\t1.00" << std::endl;

    for (int num_threads : thread_counts) {
        omp_set_num_threads(num_threads);
        double lat = measure_latency_batch(
            [&](const float* q) {
                return ivf_pq_search_omp(base, q, pq_codebook, pq_codes,
                                         ivf_centroids, ivf_lists,
                                         base_number, vecdim, k, p_pq, nprobe);
            },
            queries
        );
        double speedup = base_lat / lat;
        std::cout << num_threads << "\t" << lat << "\t" << speedup << std::endl;
    }
}

// 方法二数据结构（每个簇独立 PQ）
struct ClusterPQIndex {
    int n_centroids;
    int m;
    int sub_dim;
    // [cluster][m][256*sub_dim]
    std::vector<std::vector<std::vector<float>>> codebooks;
    // [cluster][m][cluster_size]
    std::vector<std::vector<std::vector<uint8_t>>> codes;
};

// 加载每个簇独立 PQ 索引（方法二）
inline ClusterPQIndex load_cluster_pq(
    const std::string& meta_path,
    const std::string& codebook_path,
    const std::string& codes_path)
{
    ClusterPQIndex idx;
    std::ifstream meta(meta_path);
    std::string line;
    getline(meta, line); idx.n_centroids = stoi(line.substr(line.find('=') + 1));
    getline(meta, line); idx.m = stoi(line.substr(line.find('=') + 1));
    getline(meta, line); idx.sub_dim = stoi(line.substr(line.find('=') + 1));
    getline(meta, line); // skip "cluster_sizes:"
    std::vector<int> cluster_sizes(idx.n_centroids);
    for (int i = 0; i < idx.n_centroids; ++i) {
        getline(meta, line);
        cluster_sizes[i] = stoi(line);
    }
    meta.close();

    idx.codebooks.resize(idx.n_centroids);
    idx.codes.resize(idx.n_centroids);
    std::ifstream cbf(codebook_path, std::ios::binary);
    std::ifstream cdf(codes_path, std::ios::binary);
    for (int cid = 0; cid < idx.n_centroids; ++cid) {
        idx.codebooks[cid].resize(idx.m);
        idx.codes[cid].resize(idx.m);
        int cluster_size = cluster_sizes[cid];
        if (cluster_size == 0) continue;
        for (int s = 0; s < idx.m; ++s) {
            idx.codebooks[cid][s].resize(256 * idx.sub_dim);
            cbf.read((char*)idx.codebooks[cid][s].data(), sizeof(float) * 256 * idx.sub_dim);
        }
        for (int s = 0; s < idx.m; ++s) {
            idx.codes[cid][s].resize(cluster_size);
            cdf.read((char*)idx.codes[cid][s].data(), cluster_size);
        }
    }
    cbf.close(); cdf.close();
    return idx;
}
// 单线程搜索（方法二：每个簇独立 PQ）
inline std::priority_queue<std::pair<float, uint32_t>>
ivf_pq_cluster_search(
    float* base,
    const float* query,
    const std::vector<float>& centroids,
    const std::vector<std::vector<uint32_t>>& ivf_lists,
    const ClusterPQIndex& cpq,
    size_t vecdim,
    size_t k,
    size_t p_pq,
    int nprobe)
{
    int n_centroids = cpq.n_centroids;
    int m = cpq.m;
    int sub_dim = cpq.sub_dim;
    std::vector<std::pair<float, int>> cent_dist(n_centroids);
    for (int c = 0; c < n_centroids; ++c) {
        const float* center = &centroids[c * vecdim];
        cent_dist[c] = {inner_product_neon_optB(center, query, vecdim), c};
    }
    std::partial_sort(cent_dist.begin(), cent_dist.begin() + nprobe, cent_dist.end());

    std::vector<std::pair<float, uint32_t>> approx;
    for (int pi = 0; pi < nprobe; ++pi) {
        int cid = cent_dist[pi].second;
        const auto& cluster_list = ivf_lists[cid];
        if (cluster_list.empty()) continue;
        std::vector<std::vector<float>> lut(m, std::vector<float>(256));
        for (int s = 0; s < m; ++s) {
            const float* sub_q = query + s * sub_dim;
            const float* cb = cpq.codebooks[cid][s].data();
            for (int c = 0; c < 256; ++c) {
                lut[s][c] = inner_product_neon_optB(cb + c * sub_dim, sub_q, sub_dim);
            }
        }
        for (size_t local_idx = 0; local_idx < cluster_list.size(); ++local_idx) {
            uint32_t global_idx = cluster_list[local_idx];
            float dist = 0.0f;
            for (int s = 0; s < m; ++s) {
                dist += lut[s][cpq.codes[cid][s][local_idx]];
            }
            approx.emplace_back(dist, global_idx);
        }
    }

    if (approx.size() > p_pq) {
        std::partial_sort(approx.begin(), approx.begin() + p_pq, approx.end());
        approx.resize(p_pq);
    } else {
        std::sort(approx.begin(), approx.end());
    }

    std::priority_queue<std::pair<float, uint32_t>> heap;
    for (auto& item : approx) {
        float dis = inner_product_neon_optB(base + item.second * vecdim, query, vecdim);
        if (heap.size() < k) heap.push({dis, item.second});
        else if (dis < heap.top().first) { heap.push({dis, item.second}); heap.pop(); }
    }
    return heap;
}

//  OpenMP 优化版搜索（方法二：每个簇独立 PQ）
inline std::priority_queue<std::pair<float, uint32_t>>
ivf_pq_cluster_search_omp(
    float* base,
    const float* query,
    const std::vector<float>& centroids,
    const std::vector<std::vector<uint32_t>>& ivf_lists,
    const ClusterPQIndex& cpq,
    size_t vecdim,
    size_t k,
    size_t p_pq,
    int nprobe)
{
    int n_centroids = cpq.n_centroids;
    int m = cpq.m;
    int sub_dim = cpq.sub_dim;
    std::vector<std::pair<float, int>> cent_dist(n_centroids);
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < n_centroids; ++c) {
        const float* center = &centroids[c * vecdim];
        cent_dist[c] = {inner_product_neon_optB(center, query, vecdim), c};
    }
    std::partial_sort(cent_dist.begin(), cent_dist.begin() + nprobe, cent_dist.end());

    std::vector<std::vector<std::pair<float, uint32_t>>> local_buffers(omp_get_max_threads());
    #pragma omp parallel for schedule(dynamic)
    for (int pi = 0; pi < nprobe; ++pi) {
        int tid = omp_get_thread_num();
        int cid = cent_dist[pi].second;
        const auto& cluster_list = ivf_lists[cid];
        if (cluster_list.empty()) continue;
        std::vector<std::vector<float>> lut(m, std::vector<float>(256));
        for (int s = 0; s < m; ++s) {
            const float* sub_q = query + s * sub_dim;
            const float* cb = cpq.codebooks[cid][s].data();
            for (int c = 0; c < 256; ++c) {
                lut[s][c] = inner_product_neon_optB(cb + c * sub_dim, sub_q, sub_dim);
            }
        }
        auto& local = local_buffers[tid];
        for (size_t local_idx = 0; local_idx < cluster_list.size(); ++local_idx) {
            uint32_t global_idx = cluster_list[local_idx];
            float dist = 0.0f;
            #pragma omp simd reduction(+:dist)
            for (int s = 0; s < m; ++s) {
                dist += lut[s][cpq.codes[cid][s][local_idx]];
            }
            local.emplace_back(dist, global_idx);
        }
    }

    std::vector<std::pair<float, uint32_t>> approx;
    for (auto& vec : local_buffers) {
        approx.insert(approx.end(), vec.begin(), vec.end());
    }
    if (approx.size() > p_pq) {
        std::partial_sort(approx.begin(), approx.begin() + p_pq, approx.end());
        approx.resize(p_pq);
    } else {
        std::sort(approx.begin(), approx.end());
    }

    std::priority_queue<std::pair<float, uint32_t>> heap;
    for (auto& item : approx) {
        float dis = inner_product_neon_optB(base + item.second * vecdim, query, vecdim);
        if (heap.size() < k) heap.push({dis, item.second});
        else if (dis < heap.top().first) { heap.push({dis, item.second}); heap.pop(); }
    }
    return heap;
}
struct FlatStdThreadParam {
    const float* base;
    const float* query;
    size_t start;
    size_t end;
    size_t vecdim;
    size_t local_p;
    std::priority_queue<std::pair<float, uint32_t>> local_heap;
};

void flat_worker_stdthread(FlatStdThreadParam* p) {
    for (size_t i = p->start; i < p->end; ++i) {
        float dis = inner_product_neon_optB(p->base + i * p->vecdim, p->query, p->vecdim);
        if (p->local_heap.size() < p->local_p) {
            p->local_heap.push({dis, i});
        } else if (dis < p->local_heap.top().first) {
            p->local_heap.push({dis, i});
            p->local_heap.pop();
        }
    }
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_stdthread(
    float* base, const float* query, size_t base_num, size_t vecdim, size_t k, size_t local_p, int num_threads) {
    std::vector<std::thread> threads;
    std::vector<FlatStdThreadParam> params(num_threads);
    size_t chunk = (base_num + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = std::min((t+1)*chunk, base_num);
        params[t] = {base, query, start, end, vecdim, local_p, {}};
        threads.emplace_back(flat_worker_stdthread, &params[t]);
    }
    for (auto& th : threads) th.join();

    std::priority_queue<std::pair<float, uint32_t>> global_heap;
    for (int t = 0; t < num_threads; ++t) {
        auto& local = params[t].local_heap;
        while (!local.empty()) {
            auto item = local.top(); local.pop();
            if (global_heap.size() < k) global_heap.push(item);
            else if (item.first < global_heap.top().first) {
                global_heap.push(item); global_heap.pop();
            }
        }
    }
    return global_heap;
}

#endif // MY_MULTITHREAD_H