# 🧮 OpenCL Matrix Multiplication Profiler

A GPU-accelerated matrix multiplication project using OpenCL and C++, built to explore how different workgroup (block) sizes affect kernel execution performance. This project benchmarks various `BLOCK_SIZE` values to find optimal GPU usage patterns for a fixed matrix size.

## 🚀 Features

- Written in **C++** with OpenCL 1.2
- Executes a parallel matrix multiplication kernel on the GPU
- Measures and logs kernel execution time using OpenCL event profiling
- Tests a range of `BLOCK_SIZE` values: `1, 2, 4, 8, 10, 20`
- Helps visualize how block size affects GPU performance

## 📸 Sample Output

OpenCL Matrix Multiplication Profiling
Matrix size: 40x40

BLOCK_SIZE,Kernel_Time_ms
1,1.234
2,0.845
4,0.562
8,0.490
10,0.500
20,0.800

## 🛠️ Build & Run

### Requirements

- C++ compiler (e.g. `g++`)
- OpenCL development libraries installed
- CMake (optional)

### Compile Manually

g++ main.cpp -o matmul -lOpenCL

### Run
./matmul

Make sure the OpenCL kernel file matrix_multiplication.cl is in the same directory.

