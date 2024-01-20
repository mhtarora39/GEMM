import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import time
import sys

def matrix_multiply(A, B):
    return np.dot(A, B)

def measure_gflops(matrix_size, num_iterations=100):
    A = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    B = np.random.rand(matrix_size, matrix_size).astype(np.float32)

    for i in range(matrix_size):
        for j in range(matrix_size):
            A[i][j] = i*matrix_size + j
            B[i][j] = i*matrix_size + j

    total_time = 0.0
    for _ in range(num_iterations):
        start_time = time.time()
        result = matrix_multiply(A, B)
        print(result)
        end_time = time.time()
        total_time += end_time - start_time

    average_time = total_time / num_iterations
    flops = 2.0 * matrix_size**3  # Assuming a single multiplication requires 2 floating-point operations
    gflops = (flops / average_time) / 1e9

    return gflops

# Adjust matrix size and number of iterations as needed
matrix_size = int(sys.argv[1])
print(matrix_size)
num_iterations = 1

gflops = measure_gflops(matrix_size, num_iterations)
print(f"Matrix size: {matrix_size}x{matrix_size}")
print(f"Number of iterations: {num_iterations}")
print(f"Average GFLOPS: {gflops:.2f} GFLOPS")
