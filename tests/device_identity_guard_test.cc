// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// PATCH_PREPARED_UNBUILT
//
// Status: this test was authored as part of the Windows DxgiAdapterNumber ->
// HIP ordinal guard patch but has NOT been compiled or executed: the authoring
// host has no MSVC/clang, Windows SDK, or HIP SDK installed (native builds of
// this repository are blocked there). It is written to compile standalone
// with any C++11 compiler, without the repository's build system:
//
//   MSVC:  cl /nologo /W4 /EHsc /std:c++14 tests\device_identity_guard_test.cc
//   gcc:   g++ -std=c++11 -Wall -Wextra -o device_identity_guard_test tests/device_identity_guard_test.cc
//   run:   ./device_identity_guard_test   (exit 0 = all checks passed)
//
// The file includes ONLY src/amdgpu/device_identity_guard.h (itself free of
// Windows/ORT/HIP API calls) plus <cstdio>; no other repository dependencies.

#include <cstdio>

#include "../src/amdgpu/device_identity_guard.h"

namespace di = gpu_ep::device_identity;

static int g_failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

static void CheckDecision(const char* case_name, const di::Verdict& verdict, di::Decision expected) {
    if (verdict.decision != expected) {
        std::printf("FAILED %s: expected %s, got %s\n", case_name, di::ToString(expected),
            di::ToString(verdict.decision));
        ++g_failures;
    }
}

// 1. The motivating mirror case: real card at DXGI 0, indirect display mirror
//    at DXGI 1, hipGetDeviceCount() == 1. Only the real card may register.
static void TestMirrorCaseSingleGpu() {
    const di::Candidate real{"72346", "0"};
    const di::Candidate mirror{"118045", "1"};
    const int hip_count = 1;

    CheckDecision("mirror/real", di::Evaluate(real, hip_count), di::Decision::kAccept);
    CheckDecision("mirror/mirror", di::Evaluate(mirror, hip_count), di::Decision::kSkipOutOfRange);

    const di::Verdict v = di::Evaluate(real, hip_count);
    CHECK(v.device_index == 0);
}

// 2. Order permutation: permuting the candidate list must not change which
//    LUIDs get accepted, and every accepted verdict must carry its own
//    candidate's ordinal.
static void TestOrderPermutation() {
    const di::Candidate candidates[] = {
        {"A", "0"},  // real GPU
        {"B", "1"},  // mirror (would be out of range with 1 HIP device)
        {"C", "2"},  // second mirror
    };
    const int hip_count = 1;
    di::Verdict verdicts[3];

    di::EvaluateAll(candidates, 3, hip_count, verdicts);
    const bool accepted_original[] = {
        verdicts[0].decision == di::Decision::kAccept,
        verdicts[1].decision == di::Decision::kAccept,
        verdicts[2].decision == di::Decision::kAccept,
    };

    // All 6 permutations of the same three candidates.
    const int permutations[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    for (int p = 0; p < 6; ++p) {
        const di::Candidate permuted[3] = {
            candidates[permutations[p][0]], candidates[permutations[p][1]], candidates[permutations[p][2]]};
        di::Verdict permuted_verdicts[3];
        di::EvaluateAll(permuted, 3, hip_count, permuted_verdicts);
        for (int j = 0; j < 3; ++j) {
            const bool accepted = permuted_verdicts[j].decision == di::Decision::kAccept;
            CHECK(accepted == accepted_original[permutations[p][j]]);
        }
    }
}

// 3. Duplicate LUIDs: two entries claiming the same physical adapter. With
//    room for two HIP devices both entries pass the range check, but only the
//    first may be accepted; the second must be deduplicated by LUID.
static void TestDuplicateLuids() {
    const int hip_count = 2;
    const di::Candidate duplicates[] = {{"A", "0"}, {"A", "1"}, {"B", "1"}};
    di::Verdict verdicts[3];
    di::EvaluateAll(duplicates, 3, hip_count, verdicts);

    CheckDecision("dup/first", verdicts[0], di::Decision::kAccept);
    CheckDecision("dup/second-same-luid", verdicts[1], di::Decision::kSkipDuplicateLuid);
    CheckDecision("dup/other-luid", verdicts[2], di::Decision::kAccept);
    CHECK(verdicts[0].device_index == 0);
    CHECK(verdicts[2].device_index == 1);

    // Only an ACCEPTED candidate claims its LUID: if the first duplicate is
    // itself invalid (out of range), the second is still considered.
    const di::Candidate reversed[] = {{"A", "5"}, {"A", "1"}};
    di::Verdict reversed_verdicts[2];
    di::EvaluateAll(reversed, 2, hip_count, reversed_verdicts);
    CheckDecision("dup/first-out-of-range", reversed_verdicts[0], di::Decision::kSkipOutOfRange);
    CheckDecision("dup/second-then-accepted", reversed_verdicts[1], di::Decision::kAccept);

    // Absent or empty LUIDs never collide (identity unknown).
    const di::Candidate anonymous[] = {{"", "0"}, {0, "1"}};
    di::Verdict anonymous_verdicts[2];
    di::EvaluateAll(anonymous, 2, hip_count, anonymous_verdicts);
    CheckDecision("anon/first", anonymous_verdicts[0], di::Decision::kAccept);
    CheckDecision("anon/second", anonymous_verdicts[1], di::Decision::kAccept);
}

// 4. Missing DxgiAdapterNumber metadata: no ordinal can be derived without
//    guessing, independent of the HIP count.
static void TestMissingMetadata() {
    const di::Candidate missing_luid{"A", 0};
    CheckDecision("missing/no-metadata", di::Evaluate(missing_luid, 1), di::Decision::kSkipMissingMetadata);
    CheckDecision("missing/no-metadata-many-gpus", di::Evaluate(missing_luid, 8),
        di::Decision::kSkipMissingMetadata);
    CheckDecision("missing/no-metadata-unknown-count", di::Evaluate(missing_luid, -1),
        di::Decision::kSkipMissingMetadata);
}

// 5. Negative / overflow / non-numeric values: all must be rejected by the
//    strict parser, never silently truncated the way std::atoi does.
static void TestInvalidValues() {
    const int hip_count = 100;  // generous: everything here fails on parse, not range
    const char* bad_values[] = {
        "",     " 1",   "1 ",   "\t1",  "+1",   "-1",   "-0",   "1x",   "x1",
        "0x1",  "1.0",  "2147483648",  "99999999999999999999",  "1,000",
    };
    for (const char* bad : bad_values) {
        const di::Verdict v = di::Evaluate(di::Candidate{"A", bad}, hip_count);
        CheckDecision("invalid-value", v, di::Decision::kSkipInvalidMetadata);
    }
    // Boundary values that must parse.
    const di::Verdict zero = di::Evaluate(di::Candidate{"A", "0"}, hip_count);
    const di::Verdict int_max = di::Evaluate(di::Candidate{"A", "2147483647"}, hip_count);
    CheckDecision("boundary/zero", zero, di::Decision::kAccept);
    CheckDecision("boundary/int-max-in-range", int_max, di::Decision::kAccept);
    CHECK(int_max.device_index == 2147483647);

    // ParseAdapterNumber direct checks.
    int parsed = -1;
    CHECK(di::ParseAdapterNumber("42", parsed) && parsed == 42);
    CHECK(!di::ParseAdapterNumber(0, parsed));
    CHECK(!di::ParseAdapterNumber("", parsed));
    CHECK(!di::ParseAdapterNumber("-1", parsed));
    CHECK(!di::ParseAdapterNumber("2147483648", parsed));
}

// 6. DXGI=0 with HIP count 1: the normal single-GPU host must keep working.
static void TestSingleGpuPass() {
    const di::Verdict v = di::Evaluate(di::Candidate{"A", "0"}, 1);
    CheckDecision("single-gpu", v, di::Decision::kAccept);
    CHECK(v.device_index == 0);
}

// 7. Two real GPUs, both valid: distinct LUIDs, ordinals 0 and 1, HIP count 2.
static void TestTwoRealGpus() {
    const di::Candidate gpus[] = {{"A", "0"}, {"B", "1"}};
    di::Verdict verdicts[2];
    di::EvaluateAll(gpus, 2, 2, verdicts);
    CheckDecision("two-gpus/first", verdicts[0], di::Decision::kAccept);
    CheckDecision("two-gpus/second", verdicts[1], di::Decision::kAccept);
    CHECK(verdicts[0].device_index == 0);
    CHECK(verdicts[1].device_index == 1);
}

// 8. HIP count 0 (no usable HIP device): nothing may register.
static void TestZeroHipDevices() {
    CheckDecision("zero-hip/dxgi0", di::Evaluate(di::Candidate{"A", "0"}, 0), di::Decision::kSkipOutOfRange);
    CheckDecision("zero-hip/dxgi1", di::Evaluate(di::Candidate{"A", "1"}, 0), di::Decision::kSkipOutOfRange);

    const di::Candidate candidates[] = {{"A", "0"}, {"B", "1"}};
    di::Verdict verdicts[2];
    di::EvaluateAll(candidates, 2, 0, verdicts);
    CheckDecision("zero-hip/batch-0", verdicts[0], di::Decision::kSkipOutOfRange);
    CheckDecision("zero-hip/batch-1", verdicts[1], di::Decision::kSkipOutOfRange);
}

// 9. Unknown HIP count (-1, hipGetDeviceCount failed): the range check is
//    disabled but the metadata checks still apply.
static void TestUnknownHipCount() {
    CheckDecision("unknown-count/valid", di::Evaluate(di::Candidate{"A", "1"}, -1), di::Decision::kAccept);
    CheckDecision("unknown-count/huge", di::Evaluate(di::Candidate{"A", "99"}, -1), di::Decision::kAccept);
    CheckDecision("unknown-count/garbage", di::Evaluate(di::Candidate{"A", "x"}, -1),
        di::Decision::kSkipInvalidMetadata);
}

int main() {
    TestMirrorCaseSingleGpu();
    TestOrderPermutation();
    TestDuplicateLuids();
    TestMissingMetadata();
    TestInvalidValues();
    TestSingleGpuPass();
    TestTwoRealGpus();
    TestZeroHipDevices();
    TestUnknownHipCount();

    if (g_failures == 0) {
        std::printf("device_identity_guard_test: all checks passed\n");
        return 0;
    }
    std::printf("device_identity_guard_test: %d check(s) failed\n", g_failures);
    return 1;
}
