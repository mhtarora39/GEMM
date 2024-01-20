#include <immintrin.h>
#include <iostream>
#include <chrono>

// https://en.wikichip.org/wiki/amd/microarchitectures/zen_2
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <pthread.h>
#include <unistd.h>
// #include <stdatomic.h>

#ifndef LOOP_COUNT
#define LOOP_COUNT 1
#endif

#ifndef N
#define N 64
#endif

// aligned?
float A[N * N] __attribute__((aligned(64)));
float B[N * N] __attribute__((aligned(64)));
float C[N * N] __attribute__((aligned(64)));
float val[N * N] __attribute__((aligned(64)));

__m256 *Am = (__m256 *)A;
__m256 *Bm = (__m256 *)B;
__m256 *Cm = (__m256 *)C;

float Bf[N * N] __attribute__((aligned(64)));
__m256 *Bfm = (__m256 *)Bf;

#define TIME_IT(func, ...)                                                                     \
  [&]() -> decltype(auto) {                                                                    \
    std::chrono::nanoseconds duration;                                                         \
    auto start_time = std::chrono::high_resolution_clock::now();                               \
    func(__VA_ARGS__);                                                                         \
    auto end_time = std::chrono::high_resolution_clock::now();                                 \
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);    \
    for (int i = 0; i < LOOP_COUNT - 1; i++)                                                   \
    {                                                                                          \
      start_time = std::chrono::high_resolution_clock::now();                                  \
      func(__VA_ARGS__);                                                                       \
      end_time = std::chrono::high_resolution_clock::now();                                    \
      duration += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time); \
    }                                                                                          \
    auto time = (duration.count() / LOOP_COUNT);                                               \
    double val = N;                                                                            \
    double gflop = 2 * val * val * val;                                                        \
    double gflops = (gflop / time);                                                            \
    std::cout << "Time taken by " #func ": " << time << " ns."                                 \
              << "GFLOPs/s " << gflops << std::endl;                                           \
  }()

#define BLOCK 64
#define BLOCK_Y 16
#define BLOCK_X 16

void matmul(int sy, int ey)
{
  // 136.77 GFLOPS on single core numpy
  // 4.9 GHz is max boost for 5950X
  // 32 FLOPS/cycle (16 FMAs, aka 2x 8 single wide / 32 byte FMAs)
  // theoretical max is 156.8 GFLOPS, we see 150
  // multicore theo max = 2508.8 GFLOPS, we see 1501.434299

  // Bf = (y/8, k, 8)
  for (int y = sy; y < ey; y += BLOCK_Y)
  {
    for (int x = 0; x < N; x += BLOCK * BLOCK_X)
    {

      __m256 acc[BLOCK_Y][BLOCK_X] = {};
      for (int k = 0; k < N; k++)
      {
        for (int iy = 0; iy < BLOCK_Y; iy++)
        {
          __m256 ta = _mm256_broadcast_ss(&A[(y + iy) * N + k]);
          for (int ix = 0; ix < BLOCK_X; ix++)
          {
            acc[iy][ix] = _mm256_fmadd_ps(ta, Bfm[((x + ix * BLOCK) * N + k * 8) / 8], acc[iy][ix]);
          }
        }
      }

      for (int iy = 0; iy < BLOCK_Y; iy++)
      {
        for (int ix = 0; ix < BLOCK_X; ix++)
        {
          Cm[((y + iy) * N + x + ix * BLOCK) / 8] = acc[iy][ix];
        }
      }
    }
  }
}

int test()
{
  return 1 + 2;
}

int main()
{
  TIME_IT(matmul, 0, N);
  std::cout << C[0] << std::endl;
}