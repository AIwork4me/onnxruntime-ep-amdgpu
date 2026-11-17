// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Status (R2.1): this test is compiled and executed by the fork-only CI
// workflow .github/workflows/guard-helper-cpp.yml (Ubuntu g++/clang++ with
// -Wall -Wextra -Werror, Windows MSVC cl /W4 /WX). It exercises ONLY the
// pure header src/amdgpu/device_identity_guard.h; it is NOT a gpu_factory.cc
// integration test and requires no HIP SDK. The native Windows AMD EP build
// remains PATCH_NATIVE_BUILD_BLOCKED_TOOLCHAIN on the authoring host (no
// MSVC/HIP SDK installed there), so factory-level behavior is unverified.
//
//   MSVC:  cl /nologo /W4 /WX /EHsc /std:c++14 tests\device_identity_guard_test.cc
//   gcc:   g++ -std=c++11 -Wall -Wextra -Werror -o device_identity_guard_test
//          tests/device_identity_guard_test.cc
//   run:   ./device_identity_guard_test   (exit 0 = all checks passed)
//
// Guard contract notes the tests encode:
//  * gpu_factory.cc currently wires the per-candidate Evaluate() only;
//    EvaluateAll()'s LUID deduplication is NOT used by the factory, so these
//    EvaluateAll tests verify the helper in isolation, not shipping behavior.
//  * A bounds guard cannot establish physical identity: a DxgiAdapterNumber
//    inside [0, hip_count) is a *range* proof only. It cannot prove the
//    ordinal refers to the same physical adapter the candidate describes, so
//    two distinct LUIDs aliasing one ordinal, or a valid-range ordinal that
//    points at the wrong HIP GPU, remain possible. See TestGuardLimits().

#include <cstdio>
#include <limits>

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
//    candidate's ordinal. (EvaluateAll helper semantics, not factory wiring.)
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
//    (EvaluateAll helper semantics; the factory does not use LUID dedup.)
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
    const di::Candidate anonymous[] = {{"", "0"}, {nullptr, "1"}};
    di::Verdict anonymous_verdicts[2];
    di::EvaluateAll(anonymous, 2, hip_count, anonymous_verdicts);
    CheckDecision("anon/first", anonymous_verdicts[0], di::Decision::kAccept);
    CheckDecision("anon/second", anonymous_verdicts[1], di::Decision::kAccept);

    // Same LUID reported with different DXGI indexes (adapter re-enumerated):
    // still one physical adapter; the first wins, the second is deduplicated
    // even though both indexes are individually in range.
    const di::Candidate renumbered[] = {{"A", "0"}, {"A", "1"}, {"A", "2"}};
    di::Verdict renumbered_verdicts[3];
    di::EvaluateAll(renumbered, 3, 3, renumbered_verdicts);
    CheckDecision("dup/renumbered-first", renumbered_verdicts[0], di::Decision::kAccept);
    CheckDecision("dup/renumbered-second", renumbered_verdicts[1], di::Decision::kSkipDuplicateLuid);
    CheckDecision("dup/renumbered-third", renumbered_verdicts[2], di::Decision::kSkipDuplicateLuid);
}

// 4. Missing DxgiAdapterNumber metadata: no ordinal can be derived without
//    guessing, independent of the HIP count.
static void TestMissingMetadata() {
    const di::Candidate missing_luid{"A", nullptr};
    CheckDecision("missing/no-metadata", di::Evaluate(missing_luid, 1), di::Decision::kSkipMissingMetadata);
    CheckDecision("missing/no-metadata-many-gpus", di::Evaluate(missing_luid, 8),
        di::Decision::kSkipMissingMetadata);
    CheckDecision("missing/no-metadata-unknown-count", di::Evaluate(missing_luid, -1),
        di::Decision::kSkipMissingMetadata);
}

// 5. Negative / overflow / non-numeric values: all must be rejected by the
//    strict parser, never silently truncated the way std::atoi does.
//    INT_MAX *parses* (tested via ParseAdapterNumber) but is out of range for
//    any realistic hip_count, so Evaluate must classify it kSkipOutOfRange —
//    never kAccept. (This assertion was wrong in the original patch draft and
//    is the deterministic failure R2.1 fixed.)
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

    // Boundary: zero is a legal ordinal when devices exist.
    const di::Verdict zero = di::Evaluate(di::Candidate{"A", "0"}, hip_count);
    CheckDecision("boundary/zero", zero, di::Decision::kAccept);

    // Boundary: INT_MAX parses as an int, but 2147483647 >= 100 devices is
    // never a valid HIP ordinal -> kSkipOutOfRange, never kAccept.
    const di::Verdict int_max = di::Evaluate(di::Candidate{"A", "2147483647"}, hip_count);
    CheckDecision("boundary/int-max-out-of-range", int_max, di::Decision::kSkipOutOfRange);
    CHECK(int_max.device_index == 2147483647);  // the rejected ordinal is echoed, not used

    // INT_MAX parsing proven independently of the range decision.
    int parsed = -1;
    CHECK(di::ParseAdapterNumber("2147483647", parsed) && parsed == 2147483647);
    CHECK(parsed == (std::numeric_limits<int>::max)());

    // ParseAdapterNumber direct checks: canonical decimals only.
    CHECK(di::ParseAdapterNumber("42", parsed) && parsed == 42);
    CHECK(di::ParseAdapterNumber("0", parsed) && parsed == 0);
    CHECK(!di::ParseAdapterNumber(nullptr, parsed));
    CHECK(!di::ParseAdapterNumber("", parsed));
    CHECK(!di::ParseAdapterNumber("-1", parsed));
    CHECK(!di::ParseAdapterNumber("2147483648", parsed));
    CHECK(!di::ParseAdapterNumber("1 ", parsed));
    CHECK(!di::ParseAdapterNumber(" 1", parsed));
    CHECK(di::ParseAdapterNumber("007", parsed) && parsed == 7);  // leading zeros parse
}

// 5b. Exact range edges: count-1 is the last accepted ordinal, count is the
//     first rejected one, on both sides of the boundary.
static void TestRangeEdges() {
    CheckDecision("edge/last", di::Evaluate(di::Candidate{"A", "99"}, 100), di::Decision::kAccept);
    CheckDecision("edge/first-rejected", di::Evaluate(di::Candidate{"A", "100"}, 100),
        di::Decision::kSkipOutOfRange);
    CheckDecision("edge/one-gpu-zero", di::Evaluate(di::Candidate{"A", "0"}, 1), di::Decision::kAccept);
    CheckDecision("edge/one-gpu-one", di::Evaluate(di::Candidate{"A", "1"}, 1),
        di::Decision::kSkipOutOfRange);
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
//    disabled (fail-open with warning at the factory's discretion) but the
//    metadata checks still apply. Documented policy risk: on such hosts a
//    mirror adapter with a large DxgiAdapterNumber would be accepted by this
//    helper; the factory logs that bounds validation is disabled.
static void TestUnknownHipCount() {
    CheckDecision("unknown-count/valid", di::Evaluate(di::Candidate{"A", "1"}, -1), di::Decision::kAccept);
    CheckDecision("unknown-count/huge", di::Evaluate(di::Candidate{"A", "99"}, -1), di::Decision::kAccept);
    CheckDecision("unknown-count/int-max", di::Evaluate(di::Candidate{"A", "2147483647"}, -1),
        di::Decision::kAccept);  // range check disabled: parse-only gate
    CheckDecision("unknown-count/garbage", di::Evaluate(di::Candidate{"A", "x"}, -1),
        di::Decision::kSkipInvalidMetadata);
    CheckDecision("unknown-count/missing", di::Evaluate(di::Candidate{"A", nullptr}, -1),
        di::Decision::kSkipMissingMetadata);
}

// 10. Documented limits of a bounds-only guard: these MUST currently accept,
//     because a range check cannot see physical identity. If any of these
//     ever starts failing, the guard grew identity verification - update the
//     factory wiring notes accordingly.
static void TestGuardLimits() {
    // Two distinct LUIDs aliasing the same ordinal: the guard accepts both;
    // only the factory (via EvaluateAll-style dedup or identity checks, not
    // currently wired) could disambiguate.
    const di::Verdict a = di::Evaluate(di::Candidate{"A", "0"}, 4);
    const di::Verdict b = di::Evaluate(di::Candidate{"B", "0"}, 4);
    CheckDecision("limits/distinct-luids-same-ordinal-a", a, di::Decision::kAccept);
    CheckDecision("limits/distinct-luids-same-ordinal-b", b, di::Decision::kAccept);
    CHECK(a.device_index == 0 && b.device_index == 0);

    // A valid-range ordinal that points at the wrong physical HIP GPU is
    // indistinguishable from the right one by range alone.
    CheckDecision("limits/wrong-gpu-possible", di::Evaluate(di::Candidate{"A", "2"}, 4),
        di::Decision::kAccept);
}

int main() {
    TestMirrorCaseSingleGpu();
    TestOrderPermutation();
    TestDuplicateLuids();
    TestMissingMetadata();
    TestInvalidValues();
    TestRangeEdges();
    TestSingleGpuPass();
    TestTwoRealGpus();
    TestZeroHipDevices();
    TestUnknownHipCount();
    TestGuardLimits();

    if (g_failures == 0) {
        std::printf("device_identity_guard_test: all checks passed\n");
        return 0;
    }
    std::printf("device_identity_guard_test: %d check(s) failed\n", g_failures);
    return 1;
}
