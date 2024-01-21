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

#ifdef DEBUG
// #define BLOCK 8
// #define BLOCK_Y 4
// #define BLOCK_X 2
#define BLOCK 
#define BLOCK_Y 
#define BLOCK_X 4
#else
// #define BLOCK 8
// #define BLOCK_Y 4
// #define BLOCK_X 2
#define BLOCK 8
#define BLOCK_Y 4
#define BLOCK_X 2
#endif

// g++ ./gemm.cpp -O3 -DN=128 -DDEBUG -mavx -march=native
// aligned?
float A[N * N] __attribute__((aligned(64)));
float B[N * N] __attribute__((aligned(64)));
float C[N * N] __attribute__((aligned(64)));
// float val[N * N] __attribute__((aligned(64)));

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

void matmul_custom()
{
}

void generate_mat()
{
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      A[i * N + j] = i * N + j;
      B[i * N + j] = i * N + j;
      C[i * N + j] = 0;

    }
  }
}

void display_mat()
{
#ifdef DEBUG
  std::cout << "*******************************\n";
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      std::cout << C[i * N + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "*******************************\n";
#endif
}

void matmul(int sy, int ey)
{
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
  generate_mat();

  // preswizzle
  for (int y = 0; y < N; y+=8) {
    for (int x = 0; x < N; x++) {
      for (int iy = 0; iy < 8; iy++) {
        //y = 1 , x = 1 
        Bf[y*N + x*8 + iy] = B[(y+iy)*N + x];
      }
    }
  }
  std::cout << "BLOCK_X " << BLOCK_X << std::endl;
  TIME_IT(matmul, 0, N);
  display_mat();
}