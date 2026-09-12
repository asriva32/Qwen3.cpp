#include <cstdio>
#include <cstdlib>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "layers.h"

// TODO: split kernels into prefill and decode

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

static int WARP_SIZE = 0;
static int MAX_THREADS_PER_BLOCK = 0;

extern "C" void set_cuda_device(int device) {
  CHECK_CUDA(cudaSetDevice(device));
  CHECK_CUDA(cudaDeviceGetAttribute(&WARP_SIZE, cudaDevAttrWarpSize, device));
  CHECK_CUDA(cudaDeviceGetAttribute(&MAX_THREADS_PER_BLOCK, cudaDevAttrMaxThreadsPerBlock, device));
}

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

__device__ inline float to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
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

        const int weight_output = blockIdx.x * kTileSize + ty;
        const int weight_column = tile_start + tx;
        weight_tile[tx][ty] = weight_output < n && weight_column < m
            ? to_float(
                weights[static_cast<size_t>(weight_output) * m + weight_column]
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
    const dim3 threads(kTileSize, kTileSize);
    const dim3 grid(
        (n + kTileSize - 1) / kTileSize,
        (batch_size + kTileSize - 1) / kTileSize
    );
    matmul_tiled<<<grid, threads>>>(
        x,
        reinterpret_cast<const __nv_bfloat16*>(weights),
        out,
        n,
        m,
        batch_size
    );
    CHECK_CUDA(cudaGetLastError());
}

__global__
void rmsnorm(
    const float * x,
    const __nv_bfloat16 *weights,
    float *out,
    float eps,
    int n,
    int batch_size
) {
    // n * row is the inner
    int row = blockIdx.x;
    int tx = threadIdx.x;

    if (row >= batch_size){
        return;
    }

    const size_t offset = row * n;

    float sum_sq = 0.0f;

    for (int col = tx; col < n; col += blockDim.x) {
        sum_sq += x[offset + col] * x[offset + col];
    }

    extern __shared__ float shared_sum[];
    shared_sum[tx] = sum_sq;
    __syncthreads();

    // blockDim.x must be a power of two.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride){
            shared_sum[tx] += shared_sum[tx + stride];
        }
        __syncthreads();
    }

    const float inv_rms =
        rsqrtf(shared_sum[0] / static_cast<float>(n) + eps);

    for (int col = tx; col < n; col += blockDim.x) {
        const float weight = to_float(weights[col]);
        out[offset + col] = x[offset + col] * inv_rms * weight;
    }
}

void rmsnorm_gpu(
    float *out,
    const float *x,
    const std::bfloat16_t *weights,
    float eps,
    int n,
    int batch_size
) {
    int threads = 256;
    size_t shared_bytes = threads * sizeof(float);
    rmsnorm<<<batch_size, threads, shared_bytes>>>(
        x, reinterpret_cast<const __nv_bfloat16*>(weights), out, eps, n, batch_size);
    CHECK_CUDA(cudaGetLastError());
}

// can fuse rope and cache updates
__device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}
#undef CHECK_CUDA
