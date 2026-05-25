#include "my_simd.h"
#include "my_multithread.h"
#include <iostream>
#include <chrono>
#include <vector>

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
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    // 加载 PQ 码本和 IVF 索引
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
    const size_t warmup = 200;
    const size_t measure = 2000;
    const int num_threads = 8;

    omp_set_num_threads(num_threads);

    // 预热
    for (size_t i = 0; i < warmup; ++i) {
        auto res = ivf_pq_search_omp(base, test_query + (i % test_number) * vecdim,
                                     codebook, codes, ivf_centroids, ivf_index.lists,
                                     base_number, vecdim, k, p_pq, nprobe);
    }
    // 正式测量
    for (size_t i = 0; i < measure; ++i) {
        auto res = ivf_pq_search_omp(base, test_query + (i % test_number) * vecdim,
                                     codebook, codes, ivf_centroids, ivf_index.lists,
                                     base_number, vecdim, k, p_pq, nprobe);
    }
    std::cout << "Perf test for IVF-PQ OMP 8 threads completed." << std::endl;
    return 0;
}