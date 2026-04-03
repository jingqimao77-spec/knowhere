// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "diskann/neighbor.h"
#include "knowhere/group_search.h"

namespace diskann {

class GroupedResultCollector {
   public:
    GroupedResultCollector(size_t k, const knowhere::group_search::Context& group_ctx) : k_(k), group_ctx_(group_ctx) {
    }

    bool
    Observe(unsigned child_internal_id, float dist) {
        const uint32_t group = group_ctx_.get_group(child_internal_id);
        if (group == knowhere::group_search::kInvalidGroup) {
            return false;
        }

        auto [it, inserted] = best_by_group_.try_emplace(group, Neighbor(child_internal_id, dist, true));
        if (inserted) {
            return true;
        }

        if (dist < it->second.distance) {
            it->second = Neighbor(child_internal_id, dist, true);
            return true;
        }

        return false;
    }

    std::vector<Neighbor>
    MaterializeAllSorted() const {
        std::vector<Neighbor> out;
        out.reserve(best_by_group_.size());
        for (const auto& entry : best_by_group_) {
            out.push_back(entry.second);
        }
        std::sort(out.begin(), out.end(), [](const Neighbor& left, const Neighbor& right) {
            return left.distance < right.distance;
        });
        return out;
    }

    std::vector<Neighbor>
    MaterializeTopK() const {
        auto out = MaterializeAllSorted();
        if (out.size() > k_) {
            out.resize(k_);
        }
        return out;
    }

    size_t
    UniqueGroupCount() const {
        return best_by_group_.size();
    }

   private:
    size_t k_;
    const knowhere::group_search::Context& group_ctx_;
    std::unordered_map<uint32_t, Neighbor> best_by_group_;
};

}  // namespace diskann
