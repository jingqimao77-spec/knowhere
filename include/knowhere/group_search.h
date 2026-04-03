#pragma once

#include <cstdint>
#include <limits>

namespace knowhere::group_search {

inline constexpr char kInternalToParentOrdAddressKey[] = "opensearch_group_search_internal_to_parent_ord_address";
inline constexpr char kParentCountKey[] = "opensearch_group_search_parent_count";
inline constexpr uint32_t kInvalidGroup = std::numeric_limits<uint32_t>::max();

struct Context {
    const uint32_t* internal_to_parent_ord = nullptr;
    uint32_t parent_count = 0;

    inline bool
    enabled() const {
        return internal_to_parent_ord != nullptr && parent_count > 0;
    }

    inline uint32_t
    get_group(uint32_t internal_id) const {
        if (!enabled()) {
            return kInvalidGroup;
        }
        return internal_to_parent_ord[internal_id];
    }
};

}  // namespace knowhere::group_search
