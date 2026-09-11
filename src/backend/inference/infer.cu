#include <cstdio>
#include <cstdlib>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "layers.h"

namespace {

constexpr int kTileSize = 16;

void check_cuda(
    cudaError_t error,
    const char* expression,
    const char* file,
    int line
) {
    if (error != cudaSuccess) {
        std::fprintf(
            stderr,
            "CUDA error at %s:%d while evaluating %s: %s (%s=%d)\n",
            file,
            line,
            expression,
            cudaGetErrorString(error),
            cudaGetErrorName(error),
            static_cast<int>(error)
        );
        std::abort();
    }
}

#define CHECK_CUDA(expression) \
    check_cuda((expression), #expression, __FILE__, __LINE__)

}  // namespace

extern "C" void* upload_cuda(void* host, size_t size) {
    void* device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, size));
    CHECK_CUDA(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
    return device;
}

extern "C" void* allocate_cuda_zeroed(size_t size) {
    void* device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, size));
    CHECK_CUDA(cudaMemset(device, 0, size));
    return device;
}

extern "C" void zero_cuda(void* device, size_t size) {
    CHECK_CUDA(cudaMemset(device, 0, size));
}

extern "C" void free_cuda(void* device) {
    if (device == nullptr) {
        return;
    }
    CHECK_CUDA(cudaFree(device));
}

__global__ void matmul_tiled(
    const float* x,
    const __nv_bfloat16* weights,
    float* out,
    int n,
    int m,
    int batch_size
) {
    __shared__ float x_tile[kTileSize][kTileSize];
    __shared__ float weight_tile[kTileSize][kTileSize];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int batch = blockIdx.y * kTileSize + ty;
    const int output = blockIdx.x * kTileSize + tx;

    float sum = 0.0f;
    for (int tile_start = 0; tile_start < m; tile_start += kTileSize) {
        const int x_column = tile_start + tx;
        x_tile[ty][tx] = batch < batch_size && x_column < m
            ? x[static_cast<size_t>(batch) * m + x_column]
            : 0.0f;

        const int weight_column = tile_start + ty;
        weight_tile[ty][tx] = output < n && weight_column < m
            ? __bfloat162float(
                weights[static_cast<size_t>(output) * m + weight_column]
            )
            : 0.0f;

        __syncthreads();

        
        for (int j = 0; j < kTileSize; ++j) {
            sum += x_tile[ty][j] * weight_tile[j][tx];
        }

        __syncthreads();
    }

    if (batch < batch_size && output < n) {
        out[static_cast<size_t>(batch) * n + output] = sum;
    }
}

void matmul_gpu(
    float* out,
    const float* x,
    const std::bfloat16_t* weights,
    int n,
    int m,
    int batch_size
) {
    if (n <= 0 || m <= 0 || batch_size <= 0) {
        return;
    }

    static_assert(sizeof(std::bfloat16_t) == sizeof(__nv_bfloat16));

    const size_t x_bytes =
        static_cast<size_t>(batch_size) * m * sizeof(float);
    const size_t weight_bytes =
        static_cast<size_t>(n) * m * sizeof(std::bfloat16_t);
    const size_t output_bytes =
        static_cast<size_t>(batch_size) * n * sizeof(float);

    float* device_x = nullptr;
    __nv_bfloat16* device_weights = nullptr;
    float* device_output = nullptr;
    CHECK_CUDA(cudaMalloc(&device_x, x_bytes));
    CHECK_CUDA(cudaMalloc(&device_weights, weight_bytes));
    CHECK_CUDA(cudaMalloc(&device_output, output_bytes));

    CHECK_CUDA(cudaMemcpy(
        device_x,
        x,
        x_bytes,
        cudaMemcpyHostToDevice
    ));
    CHECK_CUDA(cudaMemcpy(
        device_weights,
        weights,
        weight_bytes,
        cudaMemcpyHostToDevice
    ));

    const dim3 threads(kTileSize, kTileSize);
    const dim3 grid(
        (n + kTileSize - 1) / kTileSize,
        (batch_size + kTileSize - 1) / kTileSize
    );
    matmul_tiled<<<grid, threads>>>(
        device_x,
        device_weights,
        device_output,
        n,
        m,
        batch_size
    );
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(
        out,
        device_output,
        output_bytes,
        cudaMemcpyDeviceToHost
    ));
    // could also just leak memory
    CHECK_CUDA(cudaFree(device_output));
    CHECK_CUDA(cudaFree(device_weights));
    CHECK_CUDA(cudaFree(device_x));
}

void rope_gpu(float *out, int d, int head_dim, int pos, float theta, int rotary_dim) {

}
__device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}
#undef CHECK_CUDA
