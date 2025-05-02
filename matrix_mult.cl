__kernel void matMul(
    __global float *A,
    __global float *B,
    __global float *C,
    const unsigned int N
) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    float sum = 0.0;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}