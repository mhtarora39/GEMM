# Matrix Multiplication with AVX2 Intrinsics and Cache Awareness

This C++ code demonstrates matrix multiplication using AVX2 intrinsics to leverage SIMD instructions for improved performance. The implementation is also optimized for cache awareness to enhance memory locality. The matrix multiplication is based on the [Zen 2 microarchitecture](https://en.wikichip.org/wiki/amd/microarchitectures/zen_2) and is designed to exploit the parallelism offered by AVX2.

## Requirements

- C++ compiler with AVX2 support
- [GCC](https://gcc.gnu.org/), [Clang](https://clang.llvm.org/), or [Microsoft Visual C++](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

## Usage

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
