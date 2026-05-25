#include "my_simd.h"
#include "my_multithread.h"   // 包含 ivf_pq_search_singlethread 等
#include <iostream>
#include <chrono>
#include <vector>

// 加载数据的辅助函数（复制自您原有的 LoadData）
template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d) {
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(size_t i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();
    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"\n";
    return data;
}

int main() {
    // 加载数据集和索引（与您的 main.cc 相同）
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    // 加载 PQ 码本和 IVF 索引（根据您需要剖析的算法选择加载哪些）
    int m = 8;
    int sub_dim = vecdim / m;
    std::string codebook_path = "files/codebook.bin";
    std::string codes_path = "files/codes.bin";
    auto codebook = load_codebook(codebook_path, m, sub_dim);
    auto codes = load_codes(codes_path, m, base_number);
    
    int n_centroids;
    std::string cent_path = "files/ivf_centroids.bin";
    std::string inv_path = "files/ivf_inverted.bin";
    auto ivf_centroids = load_ivf_centroids(cent_path, n_centroids, vecdim);
    auto ivf_index = load_ivf_inverted(inv_path);
    
    const size_t k = 10;
    const int nprobe = 32;
    const size_t p_pq = 1000;
    const size_t warmup = 100;
    const size_t measure = 1000;
    const int num_threads = 8;

    omp_set_num_threads(num_threads);

    // // 预热
    // for (size_t i = 0; i < warmup; ++i) {
    //     auto res = ivf_pq_search_singlethread(base, test_query + (i % test_number) * vecdim,
    //                                           codebook, codes, ivf_centroids, ivf_index.lists,
    //                                           vecdim, k, p_pq, nprobe);
    // }
    // // 正式测量（perf 会采集这些指令）
    // for (size_t i = 0; i < measure; ++i) {
    //     auto res = ivf_pq_search_singlethread(base, test_query + (i % test_number) * vecdim,
    //                                           codebook, codes, ivf_centroids, ivf_index.lists,
    //                                           vecdim, k, p_pq, nprobe);
    // }

    // std::cout << "Perf test completed." << std::endl;
    // 正式测量
    for (size_t i = 0; i < measure; ++i) {
        auto res = ivf_pq_search_omp(base, test_query + (i % test_number) * vecdim,
                                     codebook, codes, ivf_centroids, ivf_index.lists,
                                     base_number, vecdim, k, p_pq, nprobe);
    }
    std::cout << "Perf test for IVF-PQ OMP 8 threads completed." << std::endl;
    return 0;
}