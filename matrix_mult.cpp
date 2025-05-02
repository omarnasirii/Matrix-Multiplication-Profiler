#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>

constexpr unsigned int N = 40;
const std::vector<size_t> BLOCK_SIZES = {1, 2, 4, 8, 10, 20};

// Utility: Error checker
void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error during operation '" << operation << "': " << err << std::endl;
        exit(1);
    }
}

// Utility: Load kernel source from file
std::string loadKernelSource(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to load kernel file: " << filename << std::endl;
        exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

// Utility: Print OpenCL build log
void printBuildLog(cl_program program, cl_device_id device) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    std::cout << "Build Log:\n" << log.data() << std::endl;
}

int main() {
    std::cout << "OpenCL Matrix Multiplication Profiling\n";
    std::cout << "Matrix size: " << N << "x" << N << "\n";

    // Allocate host matrices
    float* A = (float*)malloc(N * N * sizeof(float));
    float* B = (float*)malloc(N * N * sizeof(float));
    float* C = (float*)malloc(N * N * sizeof(float));

    int data = 0;
    for (int i = 0; i < N * N; i++) {
        A[i] = B[i] = static_cast<float>(data);
        C[i] = 0.0f;
        data++;
    }

    // Set up OpenCL platform, device, context, and queue
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    checkError(clGetPlatformIDs(1, &platform, nullptr), "Getting platform");
    checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "Getting device");

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "Creating context");

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkError(err, "Creating command queue");

    // Load and compile kernel
    std::string source = loadKernelSource("matrix_multiplication.cl");
    const char* src = source.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &src, nullptr, &err);
    checkError(err, "Creating program");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        printBuildLog(program, device);
        checkError(err, "Building program");
    }

    cl_kernel kernel = clCreateKernel(program, "matMul", &err);
    checkError(err, "Creating kernel");

    std::cout << "\nBLOCK_SIZE,Kernel_Time_ms\n";

    // Run for different BLOCK_SIZE values
    for (size_t BLOCK_SIZE : BLOCK_SIZES) {
        // Create buffers
        cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, A, &err);
        checkError(err, "Creating buffer A");

        cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, B, &err);
        checkError(err, "Creating buffer B");

        cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N, nullptr, &err);
        checkError(err, "Creating buffer C");

        // Set kernel arguments
        checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA), "Set kernel arg 0");
        checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB), "Set kernel arg 1");
        checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC), "Set kernel arg 2");
        checkError(clSetKernelArg(kernel, 3, sizeof(unsigned int), &N), "Set kernel arg 3");

        // Set up execution dimensions
        size_t globalSize[2] = {N, N};
        size_t localSize[2] = {BLOCK_SIZE, BLOCK_SIZE};

        cl_event event;
        checkError(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, localSize, 0, nullptr, &event), "Launching kernel");
        clWaitForEvents(1, &event);

        // Profile kernel execution
        cl_ulong start, end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
        double time_ms = (end - start) / 1e6;

        // Print timing result
        std::cout << BLOCK_SIZE << "," << time_ms << "\n";

        // Cleanup for this iteration
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufC);
        clReleaseEvent(event);
    }

    // Final cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(A); free(B); free(C);

    return 0;
}
