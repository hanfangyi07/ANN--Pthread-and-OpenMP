#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
#include "my_simd.h"
#include <limits>
#include <algorithm>
#include <cfloat>  
#include <cstdint>
#include <cstdlib>
// 可以自行添加需要的头文件
#include "my_multithread.h"
#include "hnswlib/space_ip_neon.h"
#include <thread>
#include <random>
#include <mutex>
using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}
// 计算单个查询的召回率
// res: 搜索返回的优先队列（包含距离和索引）
// gt: 该查询的 ground truth 索引数组（长度至少为 k）
// k: 需要比对的 top-k 数量


// 测试 Flat‑SIMD 在不同线程数下的多线程性能（Pthread vs OpenMP）
void test_flat_multithread(float* base,
                           const float* queries,
                           size_t test_number,
                           size_t base_number,
                           size_t vecdim,
                           size_t k,
                           const std::vector<int>& thread_counts,
                           const int* gt,
                           size_t gt_k) {  // gt_k 通常等于 k (10)
    std::cout << "\n=== Flat-SIMD Multi-threading ===" << std::endl;
    std::cout << "Threads\tPthread_lat(us)\tPthread_recall\tOpenMP_lat(us)\tOpenMP_recall" << std::endl;

    for (int num_threads : thread_counts) {
        // ----- Pthread 版本 -----
        double pthread_lat_total = 0.0;
        double pthread_recall_total = 0.0;

        for (size_t q = 0; q < test_number; ++q) {
            const float* query = queries + q * vecdim;
            const int* gt_q = gt + q * gt_k;   // 每个查询的 ground truth 偏移

            auto start = std::chrono::high_resolution_clock::now();
            auto res = flat_search_pthread(base, query, base_number, vecdim, k, num_threads);
            auto end = std::chrono::high_resolution_clock::now();

            double lat = std::chrono::duration<double, std::micro>(end - start).count();
            float rec = compute_recall(std::move(res), gt_q, k);

            pthread_lat_total += lat;
            pthread_recall_total += rec;
        }

        double pthread_avg_lat = pthread_lat_total / test_number;
        double pthread_avg_rec = pthread_recall_total / test_number;

        // ----- OpenMP 版本 -----
        omp_set_num_threads(num_threads);
        double omp_lat_total = 0.0;
        double omp_recall_total = 0.0;

        for (size_t q = 0; q < test_number; ++q) {
            const float* query = queries + q * vecdim;
            const int* gt_q = gt + q * gt_k;

            auto start = std::chrono::high_resolution_clock::now();
            auto res = flat_search_omp(base, query, base_number, vecdim, k);
            auto end = std::chrono::high_resolution_clock::now();

            double lat = std::chrono::duration<double, std::micro>(end - start).count();
            float rec = compute_recall(std::move(res), gt_q, k);

            omp_lat_total += lat;
            omp_recall_total += rec;
        }

        double omp_avg_lat = omp_lat_total / test_number;
        double omp_avg_rec = omp_recall_total / test_number;

        std::cout << num_threads << "\t"
                  << std::fixed << std::setprecision(2) << pthread_avg_lat << "\t\t"
                  << pthread_avg_rec << "\t\t"
                  << omp_avg_lat << "\t\t"
                  << omp_avg_rec << std::endl;
    }
}
// 测试 PQ‑SIMD 在不同线程数下的多线程性能（Pthread vs OpenMP）
void test_pq_multithread(float* base,
                         const float* queries,
                         size_t test_number,
                         size_t base_number,
                         size_t vecdim,
                         size_t k,
                         const std::vector<int>& thread_counts,
                         const int* gt,
                         size_t gt_k,
                         const std::vector<std::vector<float>>& codebook,
                         const std::vector<std::vector<uint8_t>>& codes,
                         size_t p) {
    std::cout << "\n=== PQ-SIMD Multi-threading (m=8, p=" << p << ") ===" << std::endl;
    std::cout << "Threads\tPthread_lat(us)\tPthread_recall\tOpenMP_lat(us)\tOpenMP_recall" << std::endl;

    for (int num_threads : thread_counts) {
        // ----- Pthread 版本 -----
        double pthread_lat_total = 0.0;
        double pthread_recall_total = 0.0;

        for (size_t q = 0; q < test_number; ++q) {
            const float* query = queries + q * vecdim;
            const int* gt_q = gt + q * gt_k;

            auto start = std::chrono::high_resolution_clock::now();
            auto res = pq_search_pthread(base, query, codebook, codes, base_number, vecdim, k, p, num_threads);
            auto end = std::chrono::high_resolution_clock::now();

            double lat = std::chrono::duration<double, std::micro>(end - start).count();
            float rec = compute_recall(std::move(res), gt_q, k);

            pthread_lat_total += lat;
            pthread_recall_total += rec;
        }

        double pthread_avg_lat = pthread_lat_total / test_number;
        double pthread_avg_rec = pthread_recall_total / test_number;

        // ----- OpenMP 版本 -----
        omp_set_num_threads(num_threads);
        double omp_lat_total = 0.0;
        double omp_recall_total = 0.0;

        for (size_t q = 0; q < test_number; ++q) {
            const float* query = queries + q * vecdim;
            const int* gt_q = gt + q * gt_k;

            auto start = std::chrono::high_resolution_clock::now();
            auto res = pq_search_omp(base, query, codebook, codes, base_number, vecdim, k, p);
            auto end = std::chrono::high_resolution_clock::now();

            double lat = std::chrono::duration<double, std::micro>(end - start).count();
            float rec = compute_recall(std::move(res), gt_q, k);

            omp_lat_total += lat;
            omp_recall_total += rec;
        }

        double omp_avg_lat = omp_lat_total / test_number;
        double omp_avg_rec = omp_recall_total / test_number;

        std::cout << num_threads << "\t"
                  << std::fixed << std::setprecision(2) << pthread_avg_lat << "\t\t"
                  << pthread_avg_rec << "\t\t"
                  << omp_avg_lat << "\t\t"
                  << omp_avg_rec << std::endl;
    }
}

// 测试 Flat‑SIMD 中局部候选数 trade‑off（固定线程数，改变 local_p）
void test_flat_tradeoff(float* base, const float* queries, size_t test_number,
                        size_t base_number, size_t vecdim, size_t k,
                        const int* gt, size_t gt_k, int num_threads) {
    std::vector<size_t> local_ps = {5, 10, 20, 50, 100, 200};  // 可自行增删
    std::cout << "\n=== Flat-SIMD Local_p Trade-off (threads=" << num_threads << ") ===" << std::endl;
    std::cout << "local_p\tPthread_lat(us)\tPthread_recall\tOpenMP_lat(us)\tOpenMP_recall" << std::endl;
    for (size_t local_p : local_ps) {
        // Pthread
        double pthread_lat = 0.0, pthread_rec = 0.0;
        for (size_t q = 0; q < test_number; ++q) {
            const float* query = queries + q * vecdim;
            const int* gt_q = gt + q * gt_k;
            auto start = std::chrono::high_resolution_clock::now();
            auto res = flat_search_pthread(base, query, base_number, vecdim, k, local_p, num_threads);
            auto end = std::chrono::high_resolution_clock::now();
            pthread_lat += std::chrono::duration<double, std::micro>(end - start).count();
            pthread_rec += compute_recall(std::move(res), gt_q, k);
        }
        // OpenMP
        omp_set_num_threads(num_threads);
        double omp_lat = 0.0, omp_rec = 0.0;
        for (size_t q = 0; q < test_number; ++q) {
            const float* query = queries + q * vecdim;
            const int* gt_q = gt + q * gt_k;
            auto start = std::chrono::high_resolution_clock::now();
            auto res = flat_search_omp(base, query, base_number, vecdim, k, local_p);
            auto end = std::chrono::high_resolution_clock::now();
            omp_lat += std::chrono::duration<double, std::micro>(end - start).count();
            omp_rec += compute_recall(std::move(res), gt_q, k);
        }
        std::cout << local_p << "\t" << pthread_lat/test_number << "\t\t" << pthread_rec/test_number
                  << "\t\t" << omp_lat/test_number << "\t\t" << omp_rec/test_number << std::endl;
    }
}

// 测试 IVF‑SIMD 的单线程 baseline（不同 nprobe）及多线程扩展性（固定 nprobe=16）
void test_ivf_multithread(float* base,
                          const float* queries,
                          size_t test_number,
                          size_t base_number,
                          size_t vecdim,
                          size_t k,
                          const std::vector<int>& thread_counts,
                          const int* gt,
                          size_t gt_k,
                          const std::vector<float>& centroids,
                          const std::vector<std::vector<uint32_t>>& lists)
{
    // 先测试单线程 baseline（不同 nprobe）
    std::cout << "\n=== IVF-SIMD baseline (single-thread) ===" << std::endl;
    std::cout << "nprobe\tlatency(us)\trecall" << std::endl;
    std::vector<int> nprobes = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    for (int nprobe : nprobes) {
        double total_lat = 0.0, total_rec = 0.0;
        for (size_t q = 0; q < test_number; ++q) {
            const float* query = queries + q * vecdim;
            const int* gt_q = gt + q * gt_k;
            auto start = std::chrono::high_resolution_clock::now();
            auto res = ivf_search_simd(base, query, centroids, lists, vecdim, k, nprobe);
            auto end = std::chrono::high_resolution_clock::now();
            total_lat += std::chrono::duration<double, std::micro>(end - start).count();
            total_rec += compute_recall(std::move(res), gt_q, k);
        }
        std::cout << nprobe << "\t" << total_lat/test_number << "\t\t" << total_rec/test_number << std::endl;
    }

    // 多线程测试（固定 nprobe=16，可调）
    std::cout << "\n=== IVF-SIMD Multi-threading (nprobe=16) ===" << std::endl;
    std::cout << "threads\tPthread_lat(us)\tPthread_recall\tOpenMP_lat(us)\tOpenMP_recall" << std::endl;
    int nprobe_fixed = 16;
    for (int num_threads : thread_counts) {
        // Pthread
        double pthread_lat = 0.0, pthread_rec = 0.0;
        for (size_t q = 0; q < test_number; ++q) {
            const float* query = queries + q * vecdim;
            const int* gt_q = gt + q * gt_k;
            auto start = std::chrono::high_resolution_clock::now();
            auto res = ivf_search_pthread(base, query, centroids, lists, vecdim, k, nprobe_fixed, num_threads);
            auto end = std::chrono::high_resolution_clock::now();
            pthread_lat += std::chrono::duration<double, std::micro>(end - start).count();
            pthread_rec += compute_recall(std::move(res), gt_q, k);
        }
        // OpenMP
        omp_set_num_threads(num_threads);
        double omp_lat = 0.0, omp_rec = 0.0;
        for (size_t q = 0; q < test_number; ++q) {
            const float* query = queries + q * vecdim;
            const int* gt_q = gt + q * gt_k;
            auto start = std::chrono::high_resolution_clock::now();
            auto res = ivf_search_omp(base, query, centroids, lists, vecdim, k, nprobe_fixed);
            auto end = std::chrono::high_resolution_clock::now();
            omp_lat += std::chrono::duration<double, std::micro>(end - start).count();
            omp_rec += compute_recall(std::move(res), gt_q, k);
        }
        std::cout << num_threads << "\t"
                  << pthread_lat/test_number << "\t\t" << pthread_rec/test_number << "\t\t"
                  << omp_lat/test_number << "\t\t" << omp_rec/test_number << std::endl;
    }
}


// 构建 HNSW 索引（NEON 加速）
hnswlib::HierarchicalNSW<float>* build_hnsw_index_neon(
    float* base, size_t base_number, size_t vecdim, int M, int efConstruction)
{
    static hnswlib::InnerProductSpaceNEON space(vecdim);
    hnswlib::HierarchicalNSW<float>* idx = new hnswlib::HierarchicalNSW<float>(&space, base_number, M, efConstruction);
   
    for (size_t i = 0; i < base_number; ++i) {
        idx->addPoint(base + i * vecdim, i);
    }
    std::cerr << "HNSW index built with NEON distance." << std::endl;
    return idx;
}


// HNSW 多入口点并行搜索（多线程独立搜索，结果合并）
std::priority_queue<std::pair<float, hnswlib::labeltype>>
multipoint_parallel_search(
    hnswlib::HierarchicalNSW<float>* index,
    const float* query,
    size_t k,
    int num_threads)
{
   
    std::vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> thread_results(num_threads);
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            auto res = index->searchKnn(query, k);
            thread_results[t] = std::move(res);
        });
    }
    for (auto& th : threads) th.join();
    
    // 合并所有结果，取全局 top-k
    std::priority_queue<std::pair<float, hnswlib::labeltype>> merged;
    for (auto& pq : thread_results) {
        while (!pq.empty()) {
            auto item = pq.top(); pq.pop();
            if (merged.size() < k) {
                merged.push(item);
            } else if (item.first < merged.top().first) {
                merged.push(item);
                merged.pop();
            }
        }
    }
    return merged;
}

// 测试 Flat‑SIMD 在不同 base 规模下的扩展性（固定线程数）
void test_flat_different_sizes(
    float* base,                 // 完整 base 数组（100k 向量）
    const float* queries,
    size_t test_number,
    size_t vecdim,
    size_t k,
    const int* gt,
    size_t gt_k,
    int num_threads)             // 固定线程数，例如 4
{
    std::vector<size_t> base_sizes = {10000, 20000, 50000, 80000, 100000};
    std::cout << "\n=== Flat-SIMD scalability with different base sizes (threads=" << num_threads << ") ===" << std::endl;
    std::cout << "BaseSize\tPthread_lat(us)\tPthread_recall\tOpenMP_lat(us)\tOpenMP_recall" << std::endl;

    for (size_t cur_base_num : base_sizes) {
        // ----- Pthread 版本 -----
        double pthread_lat = 0.0;
        double pthread_rec = 0.0;
        for (size_t q = 0; q < test_number; ++q) {
            const float* query = queries + q * vecdim;
            const int* gt_q = gt + q * gt_k;
            auto start = std::chrono::high_resolution_clock::now();
            auto res = flat_search_pthread(base, query, cur_base_num, vecdim, k, num_threads);
            auto end = std::chrono::high_resolution_clock::now();
            pthread_lat += std::chrono::duration<double, std::micro>(end - start).count();
            pthread_rec += compute_recall(std::move(res), gt_q, k);
        }
        double pthread_avg_lat = pthread_lat / test_number;
        double pthread_avg_rec = pthread_rec / test_number;

        // ----- OpenMP 版本 -----
        omp_set_num_threads(num_threads);
        double omp_lat = 0.0;
        double omp_rec = 0.0;
        for (size_t q = 0; q < test_number; ++q) {
            const float* query = queries + q * vecdim;
            const int* gt_q = gt + q * gt_k;
            auto start = std::chrono::high_resolution_clock::now();
            auto res = flat_search_omp(base, query, cur_base_num, vecdim, k);
            auto end = std::chrono::high_resolution_clock::now();
            omp_lat += std::chrono::duration<double, std::micro>(end - start).count();
            omp_rec += compute_recall(std::move(res), gt_q, k);
        }
        double omp_avg_lat = omp_lat / test_number;
        double omp_avg_rec = omp_rec / test_number;

        std::cout << cur_base_num << "\t"
                  << std::fixed << std::setprecision(2) << pthread_avg_lat << "\t\t"
                  << pthread_avg_rec << "\t\t"
                  << omp_avg_lat << "\t\t"
                  << omp_avg_rec << std::endl;
    }
}

// 测试 IVF‑SIMD 中伪共享优化效果（对齐局部堆 vs 原版）
void test_false_sharing_optimization(
    float* base,
    const float* queries,
    size_t test_number,
    size_t base_number,
    size_t vecdim,
    size_t k,
    const int* gt,
    size_t gt_k,
    const std::vector<float>& centroids,
    const std::vector<std::vector<uint32_t>>& lists)
{
    int nprobe = 16;
    std::vector<int> thread_counts = {2, 4, 8};  // 测试几个典型线程数
    std::cout << "\n=== False Sharing Optimization Test (IVF-SIMD, nprobe=16) ===" << std::endl;
    std::cout << "Threads\tOriginal_lat(us)\tAligned_lat(us)\tImprovement(%)" << std::endl;

    for (int num_threads : thread_counts) {
        // 原版（无对齐）
        omp_set_num_threads(num_threads);
        double original_lat = 0.0;
        for (size_t q = 0; q < test_number; ++q) {
            const float* query = queries + q * vecdim;
            const int* gt_q = gt + q * gt_k;
            auto start = std::chrono::high_resolution_clock::now();
            auto res = ivf_search_omp(base, query, centroids, lists, vecdim, k, nprobe);
            auto end = std::chrono::high_resolution_clock::now();
            original_lat += std::chrono::duration<double, std::micro>(end - start).count();
        }
        original_lat /= test_number;

        // 对齐版
        double aligned_lat = 0.0;
        for (size_t q = 0; q < test_number; ++q) {
            const float* query = queries + q * vecdim;
            const int* gt_q = gt + q * gt_k;
            auto start = std::chrono::high_resolution_clock::now();
            auto res = ivf_search_omp_aligned(base, query, centroids, lists, vecdim, k, nprobe);
            auto end = std::chrono::high_resolution_clock::now();
            aligned_lat += std::chrono::duration<double, std::micro>(end - start).count();
        }
        aligned_lat /= test_number;

        double improvement = (original_lat - aligned_lat) / original_lat * 100.0;
        std::cout << num_threads << "\t" << original_lat << "\t\t" << aligned_lat << "\t\t" << improvement << std::endl;
    }
}
int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    size_t p = 2000;  // 粗排候选数
     
    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);


    // int n_centroids;
    // auto ivf_centroids = load_ivf_centroids("files/ivf_centroids.bin", n_centroids, vecdim);
    // auto ivf_index = load_ivf_inverted("files/ivf_inverted.bin");
    // std::cerr << "Loaded IVF index." << std::endl;
    // int nprobe = 16;   // 可调参数

    // 只测试前2000条查询
    test_number = 2000;
    const size_t k = 10;
    std::vector<SearchResult> results;
    results.resize(test_number);

//测试 std::thread 版本的 Flat‑SIMD 多线程性能
    //   std::vector<int> thread_counts = {1, 2, 3, 4, 6, 8};
    // for (int num_threads : thread_counts) {
    //     double total_lat = 0.0, total_rec = 0.0;
    //     for (size_t q = 0; q < test_number; ++q) {
    //         const float* query = test_query + q * vecdim;
    //         const int* gt_q = test_gt + q * test_gt_d;
    //         auto start = std::chrono::high_resolution_clock::now();
    //         auto res = flat_search_stdthread(base, query, base_number, vecdim, k, k, num_threads);
    //         auto end = std::chrono::high_resolution_clock::now();
    //         total_lat += std::chrono::duration<double, std::micro>(end - start).count();
    //         total_rec += compute_recall(std::move(res), gt_q, k);
    //     }
    //     std::cout << num_threads << "\t" << total_lat / test_number << "\t\t" << total_rec / test_number << std::endl;
    // }
    // test_flat_different_sizes(base, test_query, test_number, vecdim, k, test_gt, test_gt_d, 4);


//准备查询指针列表
// std::vector<const float*> query_ptrs;
// std::vector<const int*> gt_ptrs;
// for (size_t i = 0; i < test_number; ++i) {
//     query_ptrs.push_back(test_query + i * vecdim);
//     gt_ptrs.push_back(test_gt + i * test_gt_d);
// }

// 加载 IVF 索引
int n_centroids;
std::string cent_path = "files/ivf_centroids.bin";
std::string inv_path = "files/ivf_inverted.bin";
auto ivf_centroids = load_ivf_centroids(cent_path, n_centroids, vecdim);
auto ivf_index = load_ivf_inverted(inv_path);
std::cerr << "Loaded IVF index: " << n_centroids << " centroids" << std::endl;

// //伪共享优化测试
// test_false_sharing_optimization(base, test_query, test_number, base_number, vecdim, k,
//                                     test_gt, test_gt_d, ivf_centroids, ivf_index.lists);

 //PQ 索引加载
    int m = 8;                     // 子空间数，必须与离线聚类时一致
    int sub_dim = vecdim / m;      // 12
    // size_t p_pq = 1500;             // 粗排候选数
    // 加载码本和编码
    std::string codebook_path = "files/codebook.bin";
    std::string codes_path = "files/codes.bin";
   auto codebook = load_codebook(codebook_path, m, sub_dim);
   auto codes = load_codes(codes_path, m, base_number);
//    std::cerr << "Loaded PQ codebook and codes." << std::endl;

 // 加载方法2的簇级 PQ 索引
// auto cluster_pq = load_cluster_pq(q
//     "files/cluster_meta.txt",
//     "files/cluster_codebooks.bin",
//     "files/cluster_codes.bin"
// );
// std::cerr << "Loaded per-cluster PQ index." << std::endl;

// // ========== HNSW 测试 ==========
// int M = 16;
// int efConstruction = 150;
// auto hnsw_index = build_hnsw_index_neon(base, base_number, vecdim, M, efConstruction);
// if (!hnsw_index) {
//     std::cerr << "Failed to build HNSW index." << std::endl;
//     return -1;
// }
// hnsw_index->setEf(100);   // 设置查询时的 ef_search

// std::vector<int> hnsw_threads = {1, 2, 3,4,5,6,7,8};
// std::cout << "\n=== HNSW multi-entrypoint parallel search (NEON distance) ===" << std::endl;
// std::cout << "Threads\tLatency(us)\tRecall@10" << std::endl;

// for (int threads : hnsw_threads) {
//     double total_lat = 0.0, total_rec = 0.0;
//     for (size_t q = 0; q < test_number; ++q) {
//         auto start = std::chrono::high_resolution_clock::now();
//         auto res = multipoint_parallel_search(hnsw_index, test_query + q * vecdim, k, threads);
//         auto end = std::chrono::high_resolution_clock::now();
//         total_lat += std::chrono::duration<double, std::micro>(end - start).count();

//         std::priority_queue<std::pair<float, uint32_t>> res_u32;
//         while (!res.empty()) {
//             auto item = res.top();
//             res.pop();
//             res_u32.push(std::make_pair(item.first, static_cast<uint32_t>(item.second)));
//         }
//         total_rec += compute_recall(std::move(res_u32), test_gt + q * test_gt_d, k);
//     }
//     std::cout << threads << "\t" << total_lat / test_number << "\t" << total_rec / test_number << std::endl;
// }
// delete hnsw_index;

//IVF‑PQ 方法二（每个簇独立 PQ）的单线程 baseline 测试
// std::cout << "\n=== Method2: IVF + Per-Cluster PQ (single-thread baseline) ===" << std::endl;
// std::cout << "nprobe\tp_pq\tlatency(us)\trecall" << std::endl;

// std::vector<int> nprobes = {4, 8, 12, 16, 24, 32, 48};
// std::vector<size_t> p_pq_list = {100, 200, 400, 500, 800, 1000};

// for (int nprobe : nprobes) {
//     for (size_t p_pq_val : p_pq_list) {
//         // 测量延迟
//         double total_lat = 0.0;
//         for (size_t q = 0; q < test_number; ++q) {
//             auto start = std::chrono::high_resolution_clock::now();
//             auto res = ivf_pq_cluster_search(
//                 base, query_ptrs[q],
//                 ivf_centroids, ivf_index.lists,
//                 cluster_pq,
//                 vecdim, k, p_pq_val, nprobe
//             );
//             auto end = std::chrono::high_resolution_clock::now();
//             total_lat += std::chrono::duration<double, std::micro>(end - start).count();
//         }
//         double avg_lat = total_lat / test_number;

//         // 测量召回率
//         double total_rec = 0.0;
//         for (size_t q = 0; q < test_number; ++q) {
//             auto res = ivf_pq_cluster_search(
//                 base, query_ptrs[q],
//                 ivf_centroids, ivf_index.lists,
//                 cluster_pq,
//                 vecdim, k, p_pq_val, nprobe
//             );
//             total_rec += compute_recall(std::move(res), gt_ptrs[q], k);
//         }
//         double avg_rec = total_rec / test_number;

//         std::cout << nprobe << "\t" << p_pq_val << "\t" << avg_lat << "\t" << avg_rec << std::endl;
//     }
// }


//方法二（每个簇独立 PQ）的多线程 OpenMP 扩展性测试
// std::cout << "\n=== Method2: IVF + Per-Cluster PQ OpenMP scaling (nprobe=32, p_pq=1000) ===" << std::endl;
// std::cout << "Threads\tLatency(us)\tRecall" << std::endl;

// int fixed_nprobe_m2 = 32;
// size_t fixed_p_pq_m2 = 1000;
// std::vector<int> thread_counts_m2 = {1, 2, 3, 4, 5, 6, 7, 8};

// for (int threads : thread_counts_m2) {
//     omp_set_num_threads(threads);
//     double total_lat = 0.0;
//     double total_rec = 0.0;

//     for (size_t q = 0; q < test_number; ++q) {
//         auto start = std::chrono::high_resolution_clock::now();
//         auto res = ivf_pq_cluster_search_omp(
//             base, query_ptrs[q],
//             ivf_centroids, ivf_index.lists,
//             cluster_pq,
//             vecdim, k, fixed_p_pq_m2, fixed_nprobe_m2
//         );
//         auto end = std::chrono::high_resolution_clock::now();
//         total_lat += std::chrono::duration<double, std::micro>(end - start).count();
//         total_rec += compute_recall(std::move(res), gt_ptrs[q], k);
//     }

//     std::cout << threads << "\t" << total_lat / test_number << "\t" << total_rec / test_number << std::endl;
// }

// 测试 IVF‑PQ 方法一（全局 PQ + IVF 索引）的单线程 baseline 性能
// std::vector<int> nprobes = {4, 8, 12, 16, 24, 32, 48};
// std::vector<size_t> p_pq_list = {100, 200, 400, 500, 800, 1000};
// std::cout << "nprobe\tp_pq\tlatency(us)\trecall" << std::endl;
// for (int nprobe : nprobes) {
//     for (size_t p_pq : p_pq_list) {
//         double lat = measure_latency_batch(
//             [&](const float* q) {
//                 return ivf_pq_search_singlethread(base, q, codebook, codes,
//                                                   ivf_centroids, ivf_index.lists,
//                                                   vecdim, k, p_pq, nprobe);
//             },
//             query_ptrs
//         );
//         double rec = measure_recall(base, query_ptrs, gt_ptrs, base_number, vecdim, k,
//                                     codebook, codes, ivf_centroids, ivf_index.lists,
//                                     nprobe, p_pq);
//         std::cout << nprobe << "\t" << p_pq << "\t" << lat << "\t" << rec << std::endl;
//     }
// }


 //测试 IVF‑PQ 方法一（全局 PQ + IVF）的多线程扩展性
    // int fixed_nprobe = 32;
    // size_t fixed_p_pq = 1000;
    // std::vector<int> thread_counts = {1,2,3,4,5,6,7,8};

    // // Pthread 扩展性
    // test_ivf_pq_pthread_scale(base, query_ptrs, base_number, vecdim, k,
    //                           codebook, codes, ivf_centroids, ivf_index.lists,
    //                           fixed_nprobe, fixed_p_pq, thread_counts);

    // // OpenMP 扩展性
    // test_ivf_pq_omp_scale(base, query_ptrs, base_number, vecdim, k,
    //                       codebook, codes, ivf_centroids, ivf_index.lists,
    //                       fixed_nprobe, fixed_p_pq, thread_counts);

 
    // 加载 IVF 索引
    // int n_centroids;
    // std::string cent_path = "files/ivf_centroids.bin";
    // std::string inv_path = "files/ivf_inverted.bin";
    // auto ivf_centroids = load_ivf_centroids(cent_path, n_centroids, vecdim);
    // auto ivf_index = load_ivf_inverted(inv_path);
    // std::cerr << "Loaded IVF index: " << n_centroids << " centroids" << std::endl;


    // 多线程测试
    //Flat‑SIMD 多线程对比测试（Pthread vs OpenMP）
    // test_flat_multithread(base, test_query, test_number, base_number, vecdim,
    //                       k, thread_counts, test_gt, test_gt_d);   // 注意传递 test_gt_d
                        
//PQ‑SIMD 多线程对比测试（Pthread vs OpenMP）
    // test_pq_multithread(base, test_query, test_number, base_number, vecdim,
    //                     k, thread_counts, test_gt, test_gt_d,
    //                     codebook, codes, p_pq);
    //Flat‑SIMD 局部候选数 trade‑off 测试
    //  test_flat_tradeoff(base, test_query, test_number, base_number, vecdim,
    //                k, test_gt, test_gt_d, 4);   // 固定线程数=4
    //IVF‑SIMD 多线程测试（含单线程 baseline）
    //  test_ivf_multithread(base, test_query, test_number, base_number, vecdim, k,
    //                      thread_counts, test_gt, test_gt_d,
    //                      ivf_centroids, ivf_index.lists);

    // PQ‑SIMD 优化版测试（自带 OpenMP 并行）
// std::cout << "\n=== PQ-SIMD with built-in OpenMP (reuse optimized) ===" << std::endl;
// std::cout << "Threads\tLatency(us)\tRecall" << std::endl;
// for (int num_threads : thread_counts) {
//     omp_set_num_threads(num_threads);
//     double total_lat = 0.0, total_rec = 0.0;
//     for (size_t q = 0; q < test_number; ++q) {
//         const float* query = test_query + q * vecdim;
//         const int* gt_q = test_gt + q * test_gt_d;
//         auto start = std::chrono::high_resolution_clock::now();
//         auto res = pq_search_optimized(base, query, codebook, codes,
//                                        base_number, vecdim, k, p_pq);
//         auto end = std::chrono::high_resolution_clock::now();
//         double lat = std::chrono::duration<double, std::micro>(end - start).count();
//         float rec = compute_recall(std::move(res), gt_q, k);
//         total_lat += lat;
//         total_rec += rec;
//     }
//     std::cout << num_threads << "\t" << total_lat/test_number << "\t\t"
//               << total_rec/test_number << std::endl;
// }

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);
  
    const int nprobe = 32;
    const size_t p_pq = 1000;
    const int num_threads = 8;
    omp_set_num_threads(num_threads);
    //查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

     


        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
     auto res = ivf_pq_search_omp(base, test_query + i * vecdim,
                             codebook, codes,
                             ivf_centroids, ivf_index.lists, 
                             base_number, vecdim, k, p_pq, nprobe);
        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    return 0;
}
