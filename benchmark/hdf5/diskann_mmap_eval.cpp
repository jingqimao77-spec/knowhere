#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <hdf5.h>

#include "knowhere/comp/index_param.h"
#include "knowhere/comp/local_file_manager.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/utils.h"
#include "knowhere/version.h"

namespace fs = std::filesystem;

namespace {
constexpr const char* kDatasetTrain = "train";
constexpr const char* kDatasetTest = "test";
constexpr const char* kDatasetNeighbors = "neighbors";
constexpr const char* kMetricType = "L2";
constexpr uint32_t kTopk = 100;
constexpr int64_t kMaxDegree = 56;
constexpr int64_t kBuildSearchListSize = 128;
constexpr int64_t kBeamwidth = 8;
constexpr double kBuildDramBudgetGb = 32.0;

struct FloatMatrix {
    uint32_t rows;
    uint32_t dim;
    std::vector<float> data;
};

struct IntMatrix {
    uint32_t rows;
    uint32_t dim;
    std::vector<int32_t> data;
};

void* ReadHdf5Dataset(const fs::path& path, const char* dataset_name, H5T_class_t dataset_class,
                      uint32_t& rows_out, uint32_t& dim_out) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error("failed to open hdf5 file: " + path.string());
    }
    hid_t dataset = H5Dopen2(file, dataset_name, H5P_DEFAULT);
    if (dataset < 0) {
        H5Fclose(file);
        throw std::runtime_error(std::string("failed to open hdf5 dataset: ") + dataset_name);
    }
    hid_t datatype = H5Dget_type(dataset);
    H5T_class_t klass = H5Tget_class(datatype);
    if (klass != dataset_class) {
        H5Tclose(datatype);
        H5Dclose(dataset);
        H5Fclose(file);
        throw std::runtime_error(std::string("unexpected dataset class for ") + dataset_name);
    }
    hid_t dataspace = H5Dget_space(dataset);
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(dataspace, dims, nullptr);
    rows_out = static_cast<uint32_t>(dims[0]);
    dim_out = static_cast<uint32_t>(dims[1]);

    hsize_t mem_dims[3] = {dims[0], dims[1], 1};
    hid_t memspace = H5Screate_simple(3, mem_dims, nullptr);
    hsize_t offset[2] = {0, 0};
    hsize_t count[2] = {dims[0], dims[1]};
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
    hsize_t mem_offset[3] = {0, 0, 0};
    hsize_t mem_count[3] = {dims[0], dims[1], 1};
    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, nullptr, mem_count, nullptr);

    void* output = nullptr;
    if (klass == H5T_FLOAT) {
        output = new float[dims[0] * dims[1]];
        H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, output);
    } else if (klass == H5T_INTEGER) {
        output = new int32_t[dims[0] * dims[1]];
        H5Dread(dataset, H5T_NATIVE_INT32, memspace, dataspace, H5P_DEFAULT, output);
    }

    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Fclose(file);
    return output;
}

FloatMatrix ReadFloatMatrix(const fs::path& path, const char* dataset_name) {
    uint32_t rows = 0;
    uint32_t dim = 0;
    auto* raw = static_cast<float*>(ReadHdf5Dataset(path, dataset_name, H5T_FLOAT, rows, dim));
    std::unique_ptr<float[]> holder(raw);
    FloatMatrix matrix{rows, dim, std::vector<float>(raw, raw + static_cast<size_t>(rows) * dim)};
    return matrix;
}

IntMatrix ReadIntMatrix(const fs::path& path, const char* dataset_name) {
    uint32_t rows = 0;
    uint32_t dim = 0;
    auto* raw = static_cast<int32_t*>(ReadHdf5Dataset(path, dataset_name, H5T_INTEGER, rows, dim));
    std::unique_ptr<int32_t[]> holder(raw);
    IntMatrix matrix{rows, dim, std::vector<int32_t>(raw, raw + static_cast<size_t>(rows) * dim)};
    return matrix;
}

void WriteRawDataToDisk(const fs::path& data_path, const FloatMatrix& train) {
    std::ofstream writer(data_path, std::ios::binary);
    if (!writer) {
        throw std::runtime_error("failed to open raw data output: " + data_path.string());
    }
    writer.write(reinterpret_cast<const char*>(&train.rows), sizeof(uint32_t));
    writer.write(reinterpret_cast<const char*>(&train.dim), sizeof(uint32_t));
    writer.write(reinterpret_cast<const char*>(train.data.data()), sizeof(float) * train.data.size());
}

size_t ReadStatusKb(const std::string& target) {
    std::ifstream in("/proc/self/status");
    std::string key;
    while (in >> key) {
        if (key == target) {
            size_t kb = 0;
            in >> kb;
            return kb;
        }
        std::string rest;
        std::getline(in, rest);
    }
    return 0;
}

float CalcRecallAtK(const int64_t* result_ids, const IntMatrix& gt, uint32_t nq, uint32_t topk) {
    size_t total_hits = 0;
    for (uint32_t i = 0; i < nq; ++i) {
        std::unordered_set<int32_t> gt_set;
        for (uint32_t j = 0; j < topk; ++j) {
            gt_set.insert(gt.data[static_cast<size_t>(i) * gt.dim + j]);
        }
        for (uint32_t j = 0; j < topk; ++j) {
            auto id = static_cast<int32_t>(result_ids[static_cast<size_t>(i) * topk + j]);
            total_hits += gt_set.count(id);
        }
    }
    return static_cast<float>(total_hits) / static_cast<float>(nq * topk);
}

knowhere::Json BuildConfig(const FloatMatrix& train, const fs::path& work_dir) {
    knowhere::Json conf;
    const auto index_prefix = (work_dir / "l2_index" / "l2").string();
    conf[knowhere::meta::DIM] = train.dim;
    conf[knowhere::meta::METRIC_TYPE] = kMetricType;
    conf["index_prefix"] = index_prefix;
    conf["data_path"] = (work_dir / "raw_data").string();
    conf["max_degree"] = kMaxDegree;
    conf["search_list_size"] = kBuildSearchListSize;
    conf["pq_code_budget_gb"] = sizeof(float) * static_cast<double>(train.dim) * static_cast<double>(train.rows) *
                                  0.125 / (1024.0 * 1024.0 * 1024.0);
    conf["build_dram_budget_gb"] = kBuildDramBudgetGb;
    return conf;
}

knowhere::Json SearchConfig(uint32_t dim, const fs::path& work_dir, bool use_mmap) {
    knowhere::Json conf;
    conf[knowhere::meta::DIM] = dim;
    conf[knowhere::meta::METRIC_TYPE] = kMetricType;
    conf[knowhere::meta::TOPK] = kTopk;
    conf["index_prefix"] = (work_dir / "l2_index" / "l2").string();
    conf["search_cache_budget_gb"] = 0.0;
    conf["beamwidth"] = kBeamwidth;
    conf["search_list_size"] = 2 * kTopk;
    conf["use_mmap"] = use_mmap;
    return conf;
}

knowhere::Index<knowhere::IndexNode> CreateDiskannIndex() {
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    return knowhere::IndexFactory::Instance().Create<knowhere::fp32>("DISKANN", version, diskann_index_pack).value();
}

int RunPrepare(const fs::path& dataset_path, const fs::path& work_dir) {
    fs::create_directories(work_dir / "l2_index");
    auto train = ReadFloatMatrix(dataset_path, kDatasetTrain);
    WriteRawDataToDisk(work_dir / "raw_data", train);

    auto conf = BuildConfig(train, work_dir);
    auto index = CreateDiskannIndex();
    knowhere::DataSetPtr ds_ptr = nullptr;
    auto status = index.Build(*ds_ptr, conf);
    if (status != knowhere::Status::success) {
        std::cerr << "build failed: " << static_cast<int>(status) << std::endl;
        return 2;
    }
    std::cout << "dataset=" << dataset_path.string() << "\n";
    std::cout << "work_dir=" << work_dir.string() << "\n";
    std::cout << "prepared_rows=" << train.rows << "\n";
    std::cout << "dim=" << train.dim << "\n";
    return 0;
}

int RunBench(const fs::path& dataset_path, const fs::path& work_dir, bool use_mmap) {
    auto query = ReadFloatMatrix(dataset_path, kDatasetTest);
    auto gt = ReadIntMatrix(dataset_path, kDatasetNeighbors);
    if (gt.dim < kTopk || gt.rows != query.rows) {
        std::cerr << "unexpected ground-truth shape" << std::endl;
        return 3;
    }

    auto conf = SearchConfig(query.dim, work_dir, use_mmap);
    auto index = CreateDiskannIndex();
    auto query_ds = knowhere::GenDataSet(query.rows, query.dim, query.data.data());

    knowhere::BinarySet binset;
    auto t0 = std::chrono::steady_clock::now();
    auto st = index.Deserialize(binset, conf);
    auto t1 = std::chrono::steady_clock::now();
    if (st != knowhere::Status::success) {
        std::cerr << "deserialize failed: " << static_cast<int>(st) << std::endl;
        return 4;
    }
    auto rss_after_load = ReadStatusKb("VmRSS:");

    auto s0 = std::chrono::steady_clock::now();
    auto res = index.Search(*query_ds, conf, nullptr);
    auto s1 = std::chrono::steady_clock::now();
    if (!res.has_value()) {
        std::cerr << "search failed: " << static_cast<int>(res.error()) << std::endl;
        return 5;
    }
    auto rss_after_search = ReadStatusKb("VmRSS:");
    auto hwm_kb = ReadStatusKb("VmHWM:");

    double deserialize_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double search_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();
    float recall = CalcRecallAtK(res.value()->GetIds(), gt, query.rows, kTopk);

    std::cout << "dataset=" << dataset_path.string() << "\n";
    std::cout << "query_rows=" << query.rows << "\n";
    std::cout << "topk=" << kTopk << "\n";
    std::cout << "use_mmap=" << (use_mmap ? 1 : 0) << "\n";
    std::cout << "deserialize_ms=" << deserialize_ms << "\n";
    std::cout << "search_ms=" << search_ms << "\n";
    std::cout << "recall_at_100=" << recall << "\n";
    std::cout << "rss_after_load_kb=" << rss_after_load << "\n";
    std::cout << "rss_after_search_kb=" << rss_after_search << "\n";
    std::cout << "vmhwm_kb=" << hwm_kb << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage:\n"
                  << "  diskann_mmap_eval prepare <dataset.hdf5> <work_dir>\n"
                  << "  diskann_mmap_eval bench <dataset.hdf5> <work_dir> <use_mmap 0|1>\n";
        return 1;
    }

    const std::string command = argv[1];
    const fs::path dataset_path = argv[2];
    const fs::path work_dir = argv[3];

    try {
        if (command == "prepare") {
            return RunPrepare(dataset_path, work_dir);
        }
        if (command == "bench") {
            if (argc < 5) {
                std::cerr << "bench requires use_mmap argument" << std::endl;
                return 1;
            }
            return RunBench(dataset_path, work_dir, std::stoi(argv[4]) != 0);
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 10;
    }

    std::cerr << "unknown command: " << command << std::endl;
    return 1;
}
