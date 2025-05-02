# 🧮 OpenCL Matrix Multiplication Profiler

A GPU-accelerated matrix multiplication project using OpenCL and C++, built to explore how different workgroup (block) sizes affect kernel execution performance. This project benchmarks various `BLOCK_SIZE` values to find optimal GPU usage patterns for a fixed matrix size.

## 🚀 Features

- Written in **C++** with OpenCL 1.2
- Executes a parallel matrix multiplication kernel on the GPU
- Measures and logs kernel execution time using OpenCL event profiling
- Tests a range of `BLOCK_SIZE` values: `1, 2, 4, 8, 10, 20`
- Helps visualize how block size affects GPU performance

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


## 💡 Notes
The matrix size is fixed at 40×40 but can be easily changed via the N constant.

All profiling is done using OpenCL's CL_PROFILING_COMMAND_START and CL_PROFILING_COMMAND_END.

## 📘 What I Learned
Fundamentals of GPU parallelism and workgroup sizing

How to manage OpenCL contexts, devices, queues, and memory buffers

Use of profiling events to analyze kernel performance

Insights into memory access patterns and their effects on execution time

## 📫 Contact
For questions or improvements, feel free to open an issue or pull request!


You can place the `.cpp` file as `main.cpp` and the kernel as `matrix_multiplication.cl`. Once it's in a GitHub repo, name the repo something like:

opencl-matrix-multiplication

Let me know if you want a `CMakeLists.txt`, a graph of the timings, or a badge-styled project header image.








