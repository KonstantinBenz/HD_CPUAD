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

        // TODO
        V = views::iota(0, N);       // values: 0,1,2,...,N-1
        W = vector<Real>(256, 1.0f); // fixed-size output buffer
        // --------
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
#pragma omp simd
            for (Index i = 0; i < N - rem; i += unroll_factor)
            {
                W[i + 0] = a * V[i + 0] + W[i + 0];
                W[i + 1] = a * V[i + 1] + W[i + 1];
                W[i + 2] = a * V[i + 2] + W[i + 2];
                W[i + 3] = a * V[i + 3] + W[i + 3];
                W[i + 4] = a * V[i + 4] + W[i + 4];
                W[i + 5] = a * V[i + 5] + W[i + 5];
                W[i + 6] = a * V[i + 6] + W[i + 6];
                W[i + 7] = a * V[i + 7] + W[i + 7];
                W[i + 8] = a * V[i + 8] + W[i + 8];
                W[i + 9] = a * V[i + 9] + W[i + 9];
                W[i + 10] = a * V[i + 10] + W[i + 10];
                W[i + 11] = a * V[i + 11] + W[i + 11];
                W[i + 12] = a * V[i + 12] + W[i + 12];
                W[i + 13] = a * V[i + 13] + W[i + 13];
                W[i + 14] = a * V[i + 14] + W[i + 14];
                W[i + 15] = a * V[i + 15] + W[i + 15];
                W[i + 16] = a * V[i + 16] + W[i + 16];
                W[i + 17] = a * V[i + 17] + W[i + 17];
                W[i + 18] = a * V[i + 18] + W[i + 18];
                W[i + 19] = a * V[i + 19] + W[i + 19];
                W[i + 20] = a * V[i + 20] + W[i + 20];
                W[i + 21] = a * V[i + 21] + W[i + 21];
                W[i + 22] = a * V[i + 22] + W[i + 22];
                W[i + 23] = a * V[i + 23] + W[i + 23];
                W[i + 24] = a * V[i + 24] + W[i + 24];
                W[i + 25] = a * V[i + 25] + W[i + 25];
                W[i + 26] = a * V[i + 26] + W[i + 26];
                W[i + 27] = a * V[i + 27] + W[i + 27];
                W[i + 28] = a * V[i + 28] + W[i + 28];
                W[i + 29] = a * V[i + 29] + W[i + 29];
                W[i + 30] = a * V[i + 30] + W[i + 30];
                W[i + 31] = a * V[i + 31] + W[i + 31];
                W[i + 32] = a * V[i + 32] + W[i + 32];
                W[i + 33] = a * V[i + 33] + W[i + 33];
                W[i + 34] = a * V[i + 34] + W[i + 34];
                W[i + 35] = a * V[i + 35] + W[i + 35];
                W[i + 36] = a * V[i + 36] + W[i + 36];
                W[i + 37] = a * V[i + 37] + W[i + 37];
                W[i + 38] = a * V[i + 38] + W[i + 38];
                W[i + 39] = a * V[i + 39] + W[i + 39];
                W[i + 40] = a * V[i + 40] + W[i + 40];
                W[i + 41] = a * V[i + 41] + W[i + 41];
                W[i + 42] = a * V[i + 42] + W[i + 42];
                W[i + 43] = a * V[i + 43] + W[i + 43];
                W[i + 44] = a * V[i + 44] + W[i + 44];
                W[i + 45] = a * V[i + 45] + W[i + 45];
                W[i + 46] = a * V[i + 46] + W[i + 46];
                W[i + 47] = a * V[i + 47] + W[i + 47];
                W[i + 48] = a * V[i + 48] + W[i + 48];
                W[i + 49] = a * V[i + 49] + W[i + 49];
                W[i + 50] = a * V[i + 50] + W[i + 50];
                W[i + 51] = a * V[i + 51] + W[i + 51];
                W[i + 52] = a * V[i + 52] + W[i + 52];
                W[i + 53] = a * V[i + 53] + W[i + 53];
                W[i + 54] = a * V[i + 54] + W[i + 54];
                W[i + 55] = a * V[i + 55] + W[i + 55];
                W[i + 56] = a * V[i + 56] + W[i + 56];
                W[i + 57] = a * V[i + 57] + W[i + 57];
                W[i + 58] = a * V[i + 58] + W[i + 58];
                W[i + 59] = a * V[i + 59] + W[i + 59];
                W[i + 60] = a * V[i + 60] + W[i + 60];
                W[i + 61] = a * V[i + 61] + W[i + 61];
                W[i + 62] = a * V[i + 62] + W[i + 62];
                W[i + 63] = a * V[i + 63] + W[i + 63];
            }
#pragma omp simd
            for (Index i = N - rem; i < N; i++)
            {
                W[i] = a * V[i] + W[i];
            }
            p_loop_action();
        }
        p_log << "UnrollLoopPeeling\t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
    auto benchTransformUnrollLoopPeelingDirective(Index N = default_N)
    {
        Real a = -1.0f;
        constexpr auto unroll_factor = UNROLLFACTOR;
        constexpr auto simd_width = 8; // As specified in the task

        // Define the batch type explicitly
        using batch = xsimd::batch<Real>;

        // Static assert to ensure unroll_factor is divisible by SIMD batch size
        static_assert(unroll_factor % batch::size == 0, "Unroll factor must be divisible by SIMD width");

        N = std::max(1, N); // Ensure N is at least 1
        auto rem = N % unroll_factor;

        // Initialize V as iota-view (0, 1, 2, ..., N-1)
        auto V = std::views::iota(0, N);

        // Initialize W as a fixed-size vector with initial values
        auto W = std::vector<Real>(256, 1.0f);

        Index Nout = std::min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            // Create a constant SIMD batch for 'a'
            batch a_vec(a);

// Use directive-based unrolling with SIMD
#pragma omp simd
            for (Index i = 0; i < N - rem; i += unroll_factor)
            {
#pragma unroll
                for (Index j = 0; j < unroll_factor; j += simd_width)
                {
                    // Create arrays to hold values before loading them into SIMD batches
                    std::array<Real, simd_width> v_array;
                    std::array<Real, simd_width> w_array;

                    // Fill arrays with values
                    for (Index k = 0; k < simd_width; ++k)
                    {
                        v_array[k] = static_cast<Real>(V[i + j + k]);
                        w_array[k] = W[(i + j + k) % 256];
                    }

                    // Load arrays into SIMD batches
                    batch v_vec = batch::load_unaligned(v_array.data());
                    batch w_vec = batch::load_unaligned(w_array.data());

                    // Perform SIMD computation
                    w_vec = a_vec * v_vec + w_vec;

                    // Store results back to temporary array
                    std::array<Real, simd_width> result_array;
                    w_vec.store_unaligned(result_array.data());

                    // Copy results back to W with modulo indexing
                    for (Index k = 0; k < simd_width; ++k)
                    {
                        W[(i + j + k) % 256] = result_array[k];
                    }
                }
            }

            // Handle remainder elements
            for (Index i = N - rem; i < N; ++i)
            {
                W[i % 256] = a * V[i] + W[i % 256];
            }

            p_loop_action();
        }

        p_log << "DirectiveUnrollPeelingSIMD" << unroll_factor << " \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
};
template <typename R, typename L>
array<char, 1> transform_LoopUnrolling_view<R, L>::p_one = {0};
