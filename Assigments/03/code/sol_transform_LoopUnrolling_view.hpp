#include "utilities.hpp"
#include <random>
#include <iostream>
#include <numeric>
#include <cassert>
#include <iomanip>
#include <vector>
#include <deque>
#include <complex>
#include <algorithm>
#include <functional>
#include <ranges>
#include <execution>
#include <omp.h>
#include <xsimd/xsimd.hpp>

#include "UNROLLFACTOR.h"

#pragma once

using namespace std;

template <typename R = array<char, 1>, typename L = function<void(void)>>
class transform_LoopUnrolling_view
{

private:
    static array<char, 1> p_one;
    stringstream p_log;
    R &p_loop_state;
    L p_loop_action;

public:
    transform_LoopUnrolling_view(R &loop_state = p_one, L loop_action = []() {}) : p_loop_state(loop_state), p_loop_action(loop_action) { p_log << setprecision(3); }
    string get_log() { return p_log.str(); }

    using Index = int;
    using Int = int32_t;
    using Real = float;
    using CReal = complex<Real>;
    template <typename T>
    using Container = vector<T, allocator<T>>;
    template <typename T>
    using AlignedContainer = std::vector<T, xsimd::default_allocator<T>>;
    static constexpr auto stExec = execution::unseq;     // single-threaded execution policy
    static constexpr auto mtExec = execution::par_unseq; // multi-threaded execution policy
    decltype(std::views::iota(0, 1)) V;                  // Used for memory-independent input
    std::vector<Real> W;                                 // Output container with modulo indexing

    static constexpr Index default_n = 3;
    static constexpr Index default_m = 2;
    static constexpr Index default_N = 24;
    static constexpr Index default_Nout = 20;

    auto benchTransformOmpSimd(Index N = default_N)
    {
        // Define scaling factor
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;

        // Define input and output: V is an iota view; W is a 256-element vector reused modulo
        V = views::iota(0, N);       // generates 0,1,2,...,N-1
        W = vector<Real>(256, 1.0f); // initialize W with arbitrary value

        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
#pragma omp simd
            for (Index i = 0; i < N; i++)
            {
                W[i % 256] = a * V[i] + W[i % 256];
            }
            p_loop_action();
        }
        p_log << "OmpIndex \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformDirectiveUnroll(Index N = default_N)
    {
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;

        V = views::iota(0, N);
        W = vector<Real>(256, 1.0f);

        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
#pragma omp simd
#pragma unroll
            for (Index i = 0; i < N; i++)
            {
                W[i % 256] = a * V[i] + W[i % 256];
            }
            p_loop_action();
        }
        p_log << "DirectiveUnroll \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformDirectiveUnrollFactor64(Index N = default_N)
    {
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;

        V = views::iota(0, N);
        W = vector<Real>(256, 1.0f);

        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
#pragma omp simd
#pragma unroll(unroll_factor)
            for (Index i = 0; i < N; i++)
            {
                W[i % 256] = a * V[i] + W[i % 256];
            }
            p_loop_action();
        }
        p_log << "DirectiveUnrollFactor64 \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformUnrollManual(Index N = default_N)
    {
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;
        N = std::max(1, N); // Ensure N is at least 1
        auto rem = N % unroll_factor;

        // Initialize V with increasing values and W as reusable container with size 256
        V = views::iota(0, N);
        W = vector<Real>(256, 1.0f);

        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            // Process full blocks of unroll_factor elements
            for (Index i = 0; i < N - rem; i += unroll_factor)
            {
                if (i + 0 < N)
                    W[(i + 0) % W.size()] = a * V[i + 0] + W[(i + 0) % W.size()];
                if (i + 1 < N)
                    W[(i + 1) % W.size()] = a * V[i + 1] + W[(i + 1) % W.size()];
                if (i + 2 < N)
                    W[(i + 2) % W.size()] = a * V[i + 2] + W[(i + 2) % W.size()];
                if (i + 3 < N)
                    W[(i + 3) % W.size()] = a * V[i + 3] + W[(i + 3) % W.size()];
                if (i + 4 < N)
                    W[(i + 4) % W.size()] = a * V[i + 4] + W[(i + 4) % W.size()];
                if (i + 5 < N)
                    W[(i + 5) % W.size()] = a * V[i + 5] + W[(i + 5) % W.size()];
                if (i + 6 < N)
                    W[(i + 6) % W.size()] = a * V[i + 6] + W[(i + 6) % W.size()];
                if (i + 7 < N)
                    W[(i + 7) % W.size()] = a * V[i + 7] + W[(i + 7) % W.size()];
                if (i + 8 < N)
                    W[(i + 8) % W.size()] = a * V[i + 8] + W[(i + 8) % W.size()];
                if (i + 9 < N)
                    W[(i + 9) % W.size()] = a * V[i + 9] + W[(i + 9) % W.size()];
                if (i + 10 < N)
                    W[(i + 10) % W.size()] = a * V[i + 10] + W[(i + 10) % W.size()];
                if (i + 11 < N)
                    W[(i + 11) % W.size()] = a * V[i + 11] + W[(i + 11) % W.size()];
                if (i + 12 < N)
                    W[(i + 12) % W.size()] = a * V[i + 12] + W[(i + 12) % W.size()];
                if (i + 13 < N)
                    W[(i + 13) % W.size()] = a * V[i + 13] + W[(i + 13) % W.size()];
                if (i + 14 < N)
                    W[(i + 14) % W.size()] = a * V[i + 14] + W[(i + 14) % W.size()];
                if (i + 15 < N)
                    W[(i + 15) % W.size()] = a * V[i + 15] + W[(i + 15) % W.size()];
                if (i + 16 < N)
                    W[(i + 16) % W.size()] = a * V[i + 16] + W[(i + 16) % W.size()];
                if (i + 17 < N)
                    W[(i + 17) % W.size()] = a * V[i + 17] + W[(i + 17) % W.size()];
                if (i + 18 < N)
                    W[(i + 18) % W.size()] = a * V[i + 18] + W[(i + 18) % W.size()];
                if (i + 19 < N)
                    W[(i + 19) % W.size()] = a * V[i + 19] + W[(i + 19) % W.size()];
                if (i + 20 < N)
                    W[(i + 20) % W.size()] = a * V[i + 20] + W[(i + 20) % W.size()];
                if (i + 21 < N)
                    W[(i + 21) % W.size()] = a * V[i + 21] + W[(i + 21) % W.size()];
                if (i + 22 < N)
                    W[(i + 22) % W.size()] = a * V[i + 22] + W[(i + 22) % W.size()];
                if (i + 23 < N)
                    W[(i + 23) % W.size()] = a * V[i + 23] + W[(i + 23) % W.size()];
                if (i + 24 < N)
                    W[(i + 24) % W.size()] = a * V[i + 24] + W[(i + 24) % W.size()];
                if (i + 25 < N)
                    W[(i + 25) % W.size()] = a * V[i + 25] + W[(i + 25) % W.size()];
                if (i + 26 < N)
                    W[(i + 26) % W.size()] = a * V[i + 26] + W[(i + 26) % W.size()];
                if (i + 27 < N)
                    W[(i + 27) % W.size()] = a * V[i + 27] + W[(i + 27) % W.size()];
                if (i + 28 < N)
                    W[(i + 28) % W.size()] = a * V[i + 28] + W[(i + 28) % W.size()];
                if (i + 29 < N)
                    W[(i + 29) % W.size()] = a * V[i + 29] + W[(i + 29) % W.size()];
                if (i + 30 < N)
                    W[(i + 30) % W.size()] = a * V[i + 30] + W[(i + 30) % W.size()];
                if (i + 31 < N)
                    W[(i + 31) % W.size()] = a * V[i + 31] + W[(i + 31) % W.size()];
                if (i + 32 < N)
                    W[(i + 32) % W.size()] = a * V[i + 32] + W[(i + 32) % W.size()];
                if (i + 33 < N)
                    W[(i + 33) % W.size()] = a * V[i + 33] + W[(i + 33) % W.size()];
                if (i + 34 < N)
                    W[(i + 34) % W.size()] = a * V[i + 34] + W[(i + 34) % W.size()];
                if (i + 35 < N)
                    W[(i + 35) % W.size()] = a * V[i + 35] + W[(i + 35) % W.size()];
                if (i + 36 < N)
                    W[(i + 36) % W.size()] = a * V[i + 36] + W[(i + 36) % W.size()];
                if (i + 37 < N)
                    W[(i + 37) % W.size()] = a * V[i + 37] + W[(i + 37) % W.size()];
                if (i + 38 < N)
                    W[(i + 38) % W.size()] = a * V[i + 38] + W[(i + 38) % W.size()];
                if (i + 39 < N)
                    W[(i + 39) % W.size()] = a * V[i + 39] + W[(i + 39) % W.size()];
                if (i + 40 < N)
                    W[(i + 40) % W.size()] = a * V[i + 40] + W[(i + 40) % W.size()];
                if (i + 41 < N)
                    W[(i + 41) % W.size()] = a * V[i + 41] + W[(i + 41) % W.size()];
                if (i + 42 < N)
                    W[(i + 42) % W.size()] = a * V[i + 42] + W[(i + 42) % W.size()];
                if (i + 43 < N)
                    W[(i + 43) % W.size()] = a * V[i + 43] + W[(i + 43) % W.size()];
                if (i + 44 < N)
                    W[(i + 44) % W.size()] = a * V[i + 44] + W[(i + 44) % W.size()];
                if (i + 45 < N)
                    W[(i + 45) % W.size()] = a * V[i + 45] + W[(i + 45) % W.size()];
                if (i + 46 < N)
                    W[(i + 46) % W.size()] = a * V[i + 46] + W[(i + 46) % W.size()];
                if (i + 47 < N)
                    W[(i + 47) % W.size()] = a * V[i + 47] + W[(i + 47) % W.size()];
                if (i + 48 < N)
                    W[(i + 48) % W.size()] = a * V[i + 48] + W[(i + 48) % W.size()];
                if (i + 49 < N)
                    W[(i + 49) % W.size()] = a * V[i + 49] + W[(i + 49) % W.size()];
                if (i + 50 < N)
                    W[(i + 50) % W.size()] = a * V[i + 50] + W[(i + 50) % W.size()];
                if (i + 51 < N)
                    W[(i + 51) % W.size()] = a * V[i + 51] + W[(i + 51) % W.size()];
                if (i + 52 < N)
                    W[(i + 52) % W.size()] = a * V[i + 52] + W[(i + 52) % W.size()];
                if (i + 53 < N)
                    W[(i + 53) % W.size()] = a * V[i + 53] + W[(i + 53) % W.size()];
                if (i + 54 < N)
                    W[(i + 54) % W.size()] = a * V[i + 54] + W[(i + 54) % W.size()];
                if (i + 55 < N)
                    W[(i + 55) % W.size()] = a * V[i + 55] + W[(i + 55) % W.size()];
                if (i + 56 < N)
                    W[(i + 56) % W.size()] = a * V[i + 56] + W[(i + 56) % W.size()];
                if (i + 57 < N)
                    W[(i + 57) % W.size()] = a * V[i + 57] + W[(i + 57) % W.size()];
                if (i + 58 < N)
                    W[(i + 58) % W.size()] = a * V[i + 58] + W[(i + 58) % W.size()];
                if (i + 59 < N)
                    W[(i + 59) % W.size()] = a * V[i + 59] + W[(i + 59) % W.size()];
                if (i + 60 < N)
                    W[(i + 60) % W.size()] = a * V[i + 60] + W[(i + 60) % W.size()];
                if (i + 61 < N)
                    W[(i + 61) % W.size()] = a * V[i + 61] + W[(i + 61) % W.size()];
                if (i + 62 < N)
                    W[(i + 62) % W.size()] = a * V[i + 62] + W[(i + 62) % W.size()];
                if (i + 63 < N)
                    W[(i + 63) % W.size()] = a * V[i + 63] + W[(i + 63) % W.size()];
            }

            // Handle remaining elements
            for (Index i = N - rem; i < N; ++i)
            {
                W[i % W.size()] = a * V[i] + W[i % W.size()];
            }

            p_loop_action();
        }
        p_log << "UnrollManual \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformUnrollLoopPeeling(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;
        auto rem = N % unroll_factor;

        // Initialize input as an iota view and output as a fixed-size vector
        V = std::views::iota(0, N);       // values: 0,1,2,...,N-1
        W = std::vector<Real>(256, 1.0f); // reusable buffer of length 256

        Index Nout = std::min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            // Manually unrolled loop with explicit lines and modulo indexing
#pragma omp simd
            for (Index i = 0; i < N - rem; i += unroll_factor)
            {
                if (i + 0 < N)
                    W[(i + 0) % 256] = a * V[i + 0] + W[(i + 0) % 256];
                if (i + 1 < N)
                    W[(i + 1) % 256] = a * V[i + 1] + W[(i + 1) % 256];
                if (i + 2 < N)
                    W[(i + 2) % 256] = a * V[i + 2] + W[(i + 2) % 256];
                if (i + 3 < N)
                    W[(i + 3) % 256] = a * V[i + 3] + W[(i + 3) % 256];
                if (i + 4 < N)
                    W[(i + 4) % 256] = a * V[i + 4] + W[(i + 4) % 256];
                if (i + 5 < N)
                    W[(i + 5) % 256] = a * V[i + 5] + W[(i + 5) % 256];
                if (i + 6 < N)
                    W[(i + 6) % 256] = a * V[i + 6] + W[(i + 6) % 256];
                if (i + 7 < N)
                    W[(i + 7) % 256] = a * V[i + 7] + W[(i + 7) % 256];
                if (i + 8 < N)
                    W[(i + 8) % 256] = a * V[i + 8] + W[(i + 8) % 256];
                if (i + 9 < N)
                    W[(i + 9) % 256] = a * V[i + 9] + W[(i + 9) % 256];
                if (i + 10 < N)
                    W[(i + 10) % 256] = a * V[i + 10] + W[(i + 10) % 256];
                if (i + 11 < N)
                    W[(i + 11) % 256] = a * V[i + 11] + W[(i + 11) % 256];
                if (i + 12 < N)
                    W[(i + 12) % 256] = a * V[i + 12] + W[(i + 12) % 256];
                if (i + 13 < N)
                    W[(i + 13) % 256] = a * V[i + 13] + W[(i + 13) % 256];
                if (i + 14 < N)
                    W[(i + 14) % 256] = a * V[i + 14] + W[(i + 14) % 256];
                if (i + 15 < N)
                    W[(i + 15) % 256] = a * V[i + 15] + W[(i + 15) % 256];
                if (i + 16 < N)
                    W[(i + 16) % 256] = a * V[i + 16] + W[(i + 16) % 256];
                if (i + 17 < N)
                    W[(i + 17) % 256] = a * V[i + 17] + W[(i + 17) % 256];
                if (i + 18 < N)
                    W[(i + 18) % 256] = a * V[i + 18] + W[(i + 18) % 256];
                if (i + 19 < N)
                    W[(i + 19) % 256] = a * V[i + 19] + W[(i + 19) % 256];
                if (i + 20 < N)
                    W[(i + 20) % 256] = a * V[i + 20] + W[(i + 20) % 256];
                if (i + 21 < N)
                    W[(i + 21) % 256] = a * V[i + 21] + W[(i + 21) % 256];
                if (i + 22 < N)
                    W[(i + 22) % 256] = a * V[i + 22] + W[(i + 22) % 256];
                if (i + 23 < N)
                    W[(i + 23) % 256] = a * V[i + 23] + W[(i + 23) % 256];
                if (i + 24 < N)
                    W[(i + 24) % 256] = a * V[i + 24] + W[(i + 24) % 256];
                if (i + 25 < N)
                    W[(i + 25) % 256] = a * V[i + 25] + W[(i + 25) % 256];
                if (i + 26 < N)
                    W[(i + 26) % 256] = a * V[i + 26] + W[(i + 26) % 256];
                if (i + 27 < N)
                    W[(i + 27) % 256] = a * V[i + 27] + W[(i + 27) % 256];
                if (i + 28 < N)
                    W[(i + 28) % 256] = a * V[i + 28] + W[(i + 28) % 256];
                if (i + 29 < N)
                    W[(i + 29) % 256] = a * V[i + 29] + W[(i + 29) % 256];
                if (i + 30 < N)
                    W[(i + 30) % 256] = a * V[i + 30] + W[(i + 30) % 256];
                if (i + 31 < N)
                    W[(i + 31) % 256] = a * V[i + 31] + W[(i + 31) % 256];
                if (i + 32 < N)
                    W[(i + 32) % 256] = a * V[i + 32] + W[(i + 32) % 256];
                if (i + 33 < N)
                    W[(i + 33) % 256] = a * V[i + 33] + W[(i + 33) % 256];
                if (i + 34 < N)
                    W[(i + 34) % 256] = a * V[i + 34] + W[(i + 34) % 256];
                if (i + 35 < N)
                    W[(i + 35) % 256] = a * V[i + 35] + W[(i + 35) % 256];
                if (i + 36 < N)
                    W[(i + 36) % 256] = a * V[i + 36] + W[(i + 36) % 256];
                if (i + 37 < N)
                    W[(i + 37) % 256] = a * V[i + 37] + W[(i + 37) % 256];
                if (i + 38 < N)
                    W[(i + 38) % 256] = a * V[i + 38] + W[(i + 38) % 256];
                if (i + 39 < N)
                    W[(i + 39) % 256] = a * V[i + 39] + W[(i + 39) % 256];
                if (i + 40 < N)
                    W[(i + 40) % 256] = a * V[i + 40] + W[(i + 40) % 256];
                if (i + 41 < N)
                    W[(i + 41) % 256] = a * V[i + 41] + W[(i + 41) % 256];
                if (i + 42 < N)
                    W[(i + 42) % 256] = a * V[i + 42] + W[(i + 42) % 256];
                if (i + 43 < N)
                    W[(i + 43) % 256] = a * V[i + 43] + W[(i + 43) % 256];
                if (i + 44 < N)
                    W[(i + 44) % 256] = a * V[i + 44] + W[(i + 44) % 256];
                if (i + 45 < N)
                    W[(i + 45) % 256] = a * V[i + 45] + W[(i + 45) % 256];
                if (i + 46 < N)
                    W[(i + 46) % 256] = a * V[i + 46] + W[(i + 46) % 256];
                if (i + 47 < N)
                    W[(i + 47) % 256] = a * V[i + 47] + W[(i + 47) % 256];
                if (i + 48 < N)
                    W[(i + 48) % 256] = a * V[i + 48] + W[(i + 48) % 256];
                if (i + 49 < N)
                    W[(i + 49) % 256] = a * V[i + 49] + W[(i + 49) % 256];
                if (i + 50 < N)
                    W[(i + 50) % 256] = a * V[i + 50] + W[(i + 50) % 256];
                if (i + 51 < N)
                    W[(i + 51) % 256] = a * V[i + 51] + W[(i + 51) % 256];
                if (i + 52 < N)
                    W[(i + 52) % 256] = a * V[i + 52] + W[(i + 52) % 256];
                if (i + 53 < N)
                    W[(i + 53) % 256] = a * V[i + 53] + W[(i + 53) % 256];
                if (i + 54 < N)
                    W[(i + 54) % 256] = a * V[i + 54] + W[(i + 54) % 256];
                if (i + 55 < N)
                    W[(i + 55) % 256] = a * V[i + 55] + W[(i + 55) % 256];
                if (i + 56 < N)
                    W[(i + 56) % 256] = a * V[i + 56] + W[(i + 56) % 256];
                if (i + 57 < N)
                    W[(i + 57) % 256] = a * V[i + 57] + W[(i + 57) % 256];
                if (i + 58 < N)
                    W[(i + 58) % 256] = a * V[i + 58] + W[(i + 58) % 256];
                if (i + 59 < N)
                    W[(i + 59) % 256] = a * V[i + 59] + W[(i + 59) % 256];
                if (i + 60 < N)
                    W[(i + 60) % 256] = a * V[i + 60] + W[(i + 60) % 256];
                if (i + 61 < N)
                    W[(i + 61) % 256] = a * V[i + 61] + W[(i + 61) % 256];
                if (i + 62 < N)
                    W[(i + 62) % 256] = a * V[i + 62] + W[(i + 62) % 256];
                if (i + 63 < N)
                    W[(i + 63) % 256] = a * V[i + 63] + W[(i + 63) % 256];
            }

#pragma omp simd
            for (Index i = N - rem; i < N; ++i)
            {
                auto wpos = i % 256;
                W[wpos] = a * V[i] + W[wpos];
            }

            p_loop_action();
        }

        p_log << "UnrollLoopPeeling\t" << std::views::take(W, Nout) << '\n';
        return std::tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
    auto benchTransformUnrollLoopPeelingDirective(Index N = default_N)
    {
        Real a = -1.0f;
        using batch = xsimd::batch<Real>;
        constexpr auto simd_width = batch::size;

        constexpr auto unroll_factor = UNROLLFACTOR;
        static_assert(unroll_factor % simd_width == 0, "Unroll factor must be divisible by SIMD width");

        N = N % unroll_factor ? N : N + 1;
        auto rem = N % unroll_factor;

        // Initialize V as iota-view and W as fixed-size vector
        V = std::views::iota(0, N);       // input: 0, 1, 2, ..., N-1
        W = std::vector<Real>(256);       // output buffer with modulo access
        std::iota(W.begin(), W.end(), 2); // fill with 2, 3, 4, ...

        Index Nout = std::min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            batch a_vec(a);

#pragma omp simd
            for (Index i = 0; i < N - rem; i += unroll_factor)
            {
#pragma unroll
                for (Index j = 0; j < unroll_factor; j += simd_width)
                {
                    // Load V and W
                    std::array<Real, simd_width> temp_v;
                    for (Index k = 0; k < simd_width; ++k)
                        temp_v[k] = V[i + j + k];

                    batch v_vec = batch::load_unaligned(temp_v.data());
                    std::array<Real, simd_width> temp_w;
                    for (Index k = 0; k < simd_width; ++k)
                        temp_w[k] = W[(i + j + k) % 256];

                    batch w_vec = batch::load_unaligned(temp_w.data());

                    // SIMD-Berechnung
                    w_vec = a_vec * v_vec + w_vec;

                    // zurückspeichern
                    w_vec.store_unaligned(temp_w.data());
                    for (Index k = 0; k < simd_width; ++k)
                        W[(i + j + k) % 256] = temp_w[k];
                }
            }

#pragma omp simd
            for (Index i = N - rem; i < N; i++)
            {
                W[i % 256] = a * V[i] + W[i % 256];
            }

            p_loop_action();
        }

        p_log << "UnrollLoopPeelingDirective\t" << views::take(W, Nout) << '\n';
        return std::tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
};
template <typename R, typename L>
array<char, 1> transform_LoopUnrolling_view<R, L>::p_one = {0};
