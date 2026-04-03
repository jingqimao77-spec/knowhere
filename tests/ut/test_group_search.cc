// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "diskann/grouped_result_collector.h"
#include "knowhere/comp/local_file_manager.h"
#include "knowhere/group_search.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "Missing the <filesystem> header."
#endif

namespace {

std::string kGroupSearchDir = fs::current_path().string() + "/group_search_test";
std::string kGroupSearchRawDataPath = kGroupSearchDir + "/raw_data";
std::string kGroupSearchIndexDir = kGroupSearchDir + "/index";
std::string kGroupSearchIndexPrefix = kGroupSearchIndexDir + "/group";

constexpr uint32_t kRows = 12;
constexpr uint32_t kDim = 8;
constexpr uint32_t kParentCount = 4;

void
WriteRawDataToDisk(const std::string& data_path, const float* raw_data, const uint32_t num, const uint32_t dim) {
    std::ofstream writer(data_path.c_str(), std::ios::binary);
    writer.write((char*)&num, sizeof(uint32_t));
    writer.write((char*)&dim, sizeof(uint32_t));
    writer.write((char*)raw_data, sizeof(float) * num * dim);
    writer.close();
}

std::vector<float>
BuildGroupedBaseVectors() {
    std::vector<float> data;
    data.reserve(kRows * kDim);
    for (uint32_t group = 0; group < kParentCount; ++group) {
        const float base = static_cast<float>(group * 10);
        for (uint32_t child = 0; child < 3; ++child) {
            const float value = base + static_cast<float>(child) * 0.1f;
            for (uint32_t dim = 0; dim < kDim; ++dim) {
                data.push_back(value);
            }
        }
    }
    return data;
}

std::vector<uint32_t>
BuildInternalToParentOrd() {
    return {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
}

knowhere::Json
BuildDiskannBuildJson() {
    knowhere::Json json;
    json["dim"] = kDim;
    json["metric_type"] = knowhere::metric::L2;
    json["k"] = 4;
    json["index_prefix"] = kGroupSearchIndexPrefix;
    json["data_path"] = kGroupSearchRawDataPath;
    json["max_degree"] = 32;
    json["search_list_size"] = 64;
    json["pq_code_budget_gb"] = sizeof(float) * kDim * kRows * 0.125 / (1024.0 * 1024.0 * 1024.0);
    json["search_cache_budget_gb"] = sizeof(float) * kDim * kRows * 0.125 / (1024.0 * 1024.0 * 1024.0);
    json["build_dram_budget_gb"] = 4.0;
    json["beamwidth"] = 4;
    return json;
}

knowhere::Json
BuildDiskannDeserializeJson() {
    knowhere::Json json;
    json["dim"] = kDim;
    json["metric_type"] = knowhere::metric::L2;
    json["k"] = 4;
    json["index_prefix"] = kGroupSearchIndexPrefix;
    json["search_cache_budget_gb"] = sizeof(float) * kDim * kRows * 0.125 / (1024.0 * 1024.0 * 1024.0);
    json["use_mmap"] = false;
    return json;
}

knowhere::Json
BuildDiskannSearchJson(uint32_t k) {
    knowhere::Json json;
    json["dim"] = kDim;
    json["metric_type"] = knowhere::metric::L2;
    json["k"] = k;
    json["index_prefix"] = kGroupSearchIndexPrefix;
    json["search_list_size"] = 64;
    json["beamwidth"] = 4;
    return json;
}

std::vector<int64_t>
CollectIds(const knowhere::DataSet& result) {
    std::vector<int64_t> ids;
    const int64_t* raw_ids = result.GetIds();
    const auto topk = result.GetDim();
    ids.reserve(topk);
    for (int64_t i = 0; i < topk; ++i) {
        if (raw_ids[i] == -1) {
            break;
        }
        ids.push_back(raw_ids[i]);
    }
    return ids;
}

}  // namespace

TEST_CASE("Group search dataset context can be read from const dataset", "[group_search]") {
    auto query = std::vector<float>(kDim, 0.0f);
    auto ds = knowhere::GenDataSet(1, kDim, query.data());
    auto internal_to_parent_ord = BuildInternalToParentOrd();

    ds->Set<uint64_t>(
        knowhere::group_search::kInternalToParentOrdAddressKey,
        static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(internal_to_parent_ord.data()))
    );
    ds->Set<uint64_t>(knowhere::group_search::kParentCountKey, kParentCount);

    const knowhere::DataSet& const_ds = *ds;
    REQUIRE(
        const_ds.Get<uint64_t>(knowhere::group_search::kInternalToParentOrdAddressKey)
        == static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(internal_to_parent_ord.data()))
    );
    REQUIRE(const_ds.Get<uint64_t>(knowhere::group_search::kParentCountKey) == kParentCount);
}

TEST_CASE("Grouped result collector keeps the best child for each parent", "[group_search]") {
    auto internal_to_parent_ord = BuildInternalToParentOrd();
    knowhere::group_search::Context ctx { internal_to_parent_ord.data(), kParentCount };
    diskann::GroupedResultCollector collector(3, ctx);

    collector.Observe(0, 0.5f);
    collector.Observe(1, 0.2f);
    collector.Observe(3, 0.3f);
    collector.Observe(4, 0.4f);
    collector.Observe(6, 0.6f);
    collector.Observe(7, 0.1f);

    const auto grouped_top = collector.MaterializeTopK();
    REQUIRE(grouped_top.size() == 3);
    REQUIRE(grouped_top[0].id == 7);
    REQUIRE(grouped_top[1].id == 1);
    REQUIRE(grouped_top[2].id == 3);
}

TEST_CASE("DiskANN grouped search returns one representative child per parent", "[diskann][group_search]") {
    fs::remove_all(kGroupSearchDir);
    REQUIRE_NOTHROW(fs::create_directory(kGroupSearchDir));
    REQUIRE_NOTHROW(fs::create_directory(kGroupSearchIndexDir));

    auto base = BuildGroupedBaseVectors();
    WriteRawDataToDisk(kGroupSearchRawDataPath, base.data(), kRows, kDim);

    const auto version = GenTestVersionList();
    std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    auto binary_set = knowhere::BinarySet();

    {
        knowhere::DataSet* ds_ptr = nullptr;
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>("DISKANN", version, diskann_index_pack).value();
        auto build_json = BuildDiskannBuildJson();
        REQUIRE(diskann.Build(*ds_ptr, build_json) == knowhere::Status::success);
        REQUIRE_NOTHROW(diskann.Serialize(binary_set));
    }

    auto query = std::vector<float>(kDim, 0.0f);
    auto query_ds = knowhere::GenDataSet(1, kDim, query.data());
    auto internal_to_parent_ord = BuildInternalToParentOrd();
    query_ds->Set<uint64_t>(
        knowhere::group_search::kInternalToParentOrdAddressKey,
        static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(internal_to_parent_ord.data()))
    );
    query_ds->Set<uint64_t>(knowhere::group_search::kParentCountKey, kParentCount);

    SECTION("ANN path") {
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>("DISKANN", version, diskann_index_pack).value();
        REQUIRE(diskann.Deserialize(binary_set, BuildDiskannDeserializeJson()) == knowhere::Status::success);

        auto res = diskann.Search(*query_ds, BuildDiskannSearchJson(4), nullptr);
        REQUIRE(res.has_value());

        const auto ids = CollectIds(*res.value());
        REQUIRE(ids.size() == 4);

        std::vector<uint32_t> groups;
        for (const auto id : ids) {
            groups.push_back(internal_to_parent_ord[static_cast<size_t>(id)]);
        }
        std::sort(groups.begin(), groups.end());
        groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
        REQUIRE(groups.size() == 4);
    }

    SECTION("Brute-force path") {
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>("DISKANN", version, diskann_index_pack).value();
        REQUIRE(diskann.Deserialize(binary_set, BuildDiskannDeserializeJson()) == knowhere::Status::success);

        auto res = diskann.Search(*query_ds, BuildDiskannSearchJson(7), nullptr);
        REQUIRE(res.has_value());

        const auto ids = CollectIds(*res.value());
        REQUIRE(ids.size() == 4);

        std::vector<uint32_t> groups;
        for (const auto id : ids) {
            groups.push_back(internal_to_parent_ord[static_cast<size_t>(id)]);
        }
        std::sort(groups.begin(), groups.end());
        groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
        REQUIRE(groups.size() == 4);
    }

    fs::remove_all(kGroupSearchDir);
}
