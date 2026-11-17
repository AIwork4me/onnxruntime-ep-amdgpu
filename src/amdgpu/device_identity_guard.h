// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <limits>

namespace gpu_ep {
namespace device_identity {

// Decides whether an ORT-reported AMD GPU candidate may safely be mapped to a
// HIP device ordinal through its "DxgiAdapterNumber" device metadata.
//
// "DxgiAdapterNumber" is a DXGI *enumeration position* (the index the adapter
// occupies in IDXGIFactory1::EnumAdapters1 order), not a device identity and
// not a HIP ordinal. On hosts with indirect display adapters (virtual display
// drivers, RDP indirect display), DXGI enumerates zero-output mirror adapters
// that copy the real GPU's vendor/device/subsystem id and can survive ORT's
// platform-level deduplication, so an EP factory can be handed a candidate
// whose DxgiAdapterNumber is >= hipGetDeviceCount(). Passing that number on as
// a memory-info device id reaches the HIP backends as hipSetDevice(n) and
// fails with hipErrorInvalidDevice (101).
//
// This header is deliberately pure C++ with no Windows, ORT, or HIP API calls
// so the decision logic can be unit tested standalone; the factory injects the
// HIP device count it queried (tests/device_identity_guard_test.cc).
enum class Decision {
    // The candidate maps to a usable HIP ordinal; use Verdict::device_index.
    kAccept,
    // "DxgiAdapterNumber" metadata is absent, so no ordinal can be derived
    // without guessing. (On Windows ORT device discovery always writes this
    // key for GPU adapters; absence is unexpected and not safely mappable.)
    kSkipMissingMetadata,
    // The metadata is not a canonical non-negative decimal integer that fits
    // in int: empty, signed, whitespace-padded, trailing garbage, or overflow.
    // std::atoi would silently truncate several of these shapes.
    kSkipInvalidMetadata,
    // The parsed adapter number is >= the HIP device count (or the count is
    // zero): it cannot be a valid HIP ordinal and would fail with HIP 101.
    kSkipOutOfRange,
    // A previously accepted candidate already claimed the same LUID: this
    // entry is a duplicate enumeration of the same physical adapter (e.g. an
    // indirect display mirror), not a second GPU.
    kSkipDuplicateLuid,
};

// One ORT hardware device candidate reduced to the identity metadata the
// guard needs. Both strings are borrowed and only read during the call;
// null means the metadata key is absent.
struct Candidate {
    const char* luid;                 // "LUID" metadata; may be null or empty
    const char* dxgi_adapter_number;  // "DxgiAdapterNumber" metadata; may be null
};

struct Verdict {
    Decision decision;
    int device_index;  // HIP ordinal to use; meaningful only for kAccept
};

// Stable, human-readable reason for logging and test diagnostics.
inline const char* ToString(Decision decision) {
    switch (decision) {
        case Decision::kAccept:
            return "accept";
        case Decision::kSkipMissingMetadata:
            return "skip: DxgiAdapterNumber metadata is missing";
        case Decision::kSkipInvalidMetadata:
            return "skip: DxgiAdapterNumber is not a valid non-negative integer";
        case Decision::kSkipOutOfRange:
            return "skip: DxgiAdapterNumber >= hipGetDeviceCount(), not a valid HIP ordinal";
        case Decision::kSkipDuplicateLuid:
            return "skip: duplicate LUID of an already accepted adapter";
    }
    return "unknown decision";
}

// Strict parse of a DxgiAdapterNumber value: non-null, non-empty, digits only
// ('0'-'9'), and representable as int. Rejects leading signs and whitespace
// and any trailing characters, all of which std::atoi tolerates by truncation.
inline bool ParseAdapterNumber(const char* value, int& out) {
    if (value == 0 || *value == '\0') {
        return false;
    }
    const int max_div_10 = (std::numeric_limits<int>::max)() / 10;
    const int max_mod_10 = (std::numeric_limits<int>::max)() % 10;
    int result = 0;
    for (const char* p = value; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        const int digit = *p - '0';
        if (result > max_div_10 || (result == max_div_10 && digit > max_mod_10)) {
            return false;  // would overflow int
        }
        result = result * 10 + digit;
    }
    out = result;
    return true;
}

// Evaluates one candidate against the HIP device count.
//
// hip_device_count semantics: >= 0 enables the range check against that
// count; < 0 means the count is unknown (hipGetDeviceCount failed) and only
// the metadata check runs - the caller decides how to report that case.
inline Verdict Evaluate(const Candidate& candidate, int hip_device_count) {
    int adapter_number = 0;
    if (candidate.dxgi_adapter_number == 0) {
        return Verdict{Decision::kSkipMissingMetadata, 0};
    }
    if (!ParseAdapterNumber(candidate.dxgi_adapter_number, adapter_number)) {
        return Verdict{Decision::kSkipInvalidMetadata, 0};
    }
    if (hip_device_count >= 0 && adapter_number >= hip_device_count) {
        return Verdict{Decision::kSkipOutOfRange, adapter_number};
    }
    return Verdict{Decision::kAccept, adapter_number};
}

namespace detail {
inline bool StrEq(const char* a, const char* b) {
    if (a == b) {
        return true;
    }
    if (a == 0 || b == 0) {
        return false;
    }
    while (*a != '\0' && *b != '\0' && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b;
}
}  // namespace detail

// Evaluates every candidate and additionally deduplicates by LUID: at most one
// accepted candidate per non-empty LUID, the first in list order. Two DXGI
// entries carrying the same LUID are the same physical adapter seen twice
// (direct + indirect display mirror), so the second must not be registered as
// a separate HIP device. Candidates with an absent or empty LUID never
// collide because their identity is unknown. Verdicts are written to out[i]
// for candidates[i]; out must have space for count entries.
inline void EvaluateAll(const Candidate* candidates, std::size_t count, int hip_device_count,
    Verdict* out) {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = Evaluate(candidates[i], hip_device_count);
        if (out[i].decision != Decision::kAccept) {
            continue;
        }
        const char* luid = candidates[i].luid;
        if (luid == 0 || *luid == '\0') {
            continue;  // unknown identity: cannot collide with anything
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (out[j].decision == Decision::kAccept && detail::StrEq(candidates[j].luid, luid)) {
                out[i] = Verdict{Decision::kSkipDuplicateLuid, out[i].device_index};
                break;
            }
        }
    }
}

}  // namespace device_identity
}  // namespace gpu_ep
