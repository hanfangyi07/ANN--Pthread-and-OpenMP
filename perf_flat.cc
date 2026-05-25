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

    const size_t k = 10;
    const int num_threads = 4;          // 与报告一致
    const size_t warmup = 200;          // 预热 200 次
    const size_t measure = 2000;        // 测量 2000 次（与报告一致）

    omp_set_num_threads(num_threads);

    // 预热
    for (size_t i = 0; i < warmup; ++i) {
        auto res = flat_search_omp(base, test_query + (i % test_number) * vecdim, 
                                   base_number, vecdim, k);
    }
    // 正式测量
    for (size_t i = 0; i < measure; ++i) {
        auto res = flat_search_omp(base, test_query + (i % test_number) * vecdim,
                                   base_number, vecdim, k);
    }

    std::cout << "Perf test for Flat-SIMD completed." << std::endl;
    return 0;
}