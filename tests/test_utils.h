// Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root.

#pragma once

#include <cmath>
#include <cstdio>

namespace lightgp {
namespace test {

inline int& pass_count() { static int c = 0; return c; }
inline int& fail_count() { static int c = 0; return c; }

inline bool approx_eq(float a, float b, float atol, float rtol) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

#define LIGHTGP_CHECK(cond) do { \
    if (cond) { ::lightgp::test::pass_count()++; } \
    else { \
        ::lightgp::test::fail_count()++; \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)

#define LIGHTGP_CHECK_NEAR(a, b, tol) do { \
    float lightgp_check_a = (a), lightgp_check_b = (b), lightgp_check_t = (tol); \
    if (::lightgp::test::approx_eq(lightgp_check_a, lightgp_check_b, lightgp_check_t, lightgp_check_t)) { \
        ::lightgp::test::pass_count()++; \
    } else { \
        ::lightgp::test::fail_count()++; \
        std::fprintf(stderr, "FAIL %s:%d: %s ~ %s (got %.7g vs %.7g)\n", \
                     __FILE__, __LINE__, #a, #b, lightgp_check_a, lightgp_check_b); \
    } \
} while (0)

}  // namespace test
}  // namespace lightgp
