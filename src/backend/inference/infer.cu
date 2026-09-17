#include <cstdio>
#include <cstdlib>
#include <cfloat>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "layers.h"

// TODO: split kernels into prefill and decode

inline constexpr int kTileSize = 16;

namespace {
void check_cuda(cudaError_t result, const char* expression, const char* file, int line) {
    if (result == cudaSuccess) {
        return;
    }
    std::fprintf(
        stderr,
        "CUDA error at %s:%d while evaluating %s: %s\n",
        file,
        line,
        expression,
        cudaGetErrorString(result)
    );
    std::abort();
}
}  // namespace

#define CUDA_CHECK(expression) do {                \
    check_cuda((expression), #expression, __FILE__, __LINE__); \
} while(0)

extern "C" void set_cuda_device(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

extern "C" void* upload_cuda(void* host, size_t size) {
    void* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, size));
    CUDA_CHECK(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
    return device;
}

extern "C" void* allocate_cuda_zeroed(size_t size) {
    void* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, size));
    CUDA_CHECK(cudaMemset(device, 0, size));
    return device;
}

extern "C" void zero_cuda(void* device, size_t size) {
    CUDA_CHECK(cudaMemset(device, 0, size));
}

extern "C" void free_cuda(void* device) {
    if (device == nullptr) {
        return;
    }
    CUDA_CHECK(cudaFree(device));
}

extern "C" void download_cuda(void* host, const void* device, size_t size) {
    CUDA_CHECK(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
}

extern "C" void synchronize_cuda() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

__device__ inline float to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__device__ inline __nv_bfloat16 to_bfloat(float val) {
    return __float2bfloat16(val);
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

__global__ void matvec_warp(
    const float *x,
    const __nv_bfloat16 *weights,
    float *out,
    int n,
    int m
) {
    const int warp_in_block = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    const int output = blockIdx.x * (blockDim.x / warpSize) + warp_in_block;
    if (output >= n) {
        return;
    }

    const auto *weight_row = weights + static_cast<size_t>(output) * m;
    float sum = 0.0f;
    for (int column = lane; column < m; column += warpSize) {
        sum += x[column] * to_float(weight_row[column]);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        out[output] = sum;
    }
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

__device__ inline void rotate_rope_pair(
    float& first,
    float& second,
    int rotary_index,
    int pos,
    float theta,
    int rotary_dim
) {
    const auto freq = 1.0f / powf(
        theta,
        static_cast<float>(rotary_index) / static_cast<float>(rotary_dim)
    );
    const auto angle = static_cast<float>(pos) * freq;
    float sine;
    float cosine;
    sincosf(angle, &sine, &cosine);

    const auto original_first = first;
    first = original_first * cosine - second * sine;
    second = original_first * sine + second * cosine;
}

__global__ void rotate_sink_tokens_kernel(
    __nv_bfloat16 *cache_k,
    size_t num_sink,
    size_t kv_dim,
    int head_dim,
    float rope_theta,
    int rotary_dim
) {
    const auto pair_index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto pairs_per_head = static_cast<size_t>(rotary_dim / 2);
    const auto num_kv_heads = kv_dim / static_cast<size_t>(head_dim);
    const auto pairs_per_sink = num_kv_heads * pairs_per_head;
    const auto total_pairs = num_sink * pairs_per_sink;

    if (pair_index >= total_pairs) {
        return;
    }

    const auto sink = pair_index / pairs_per_sink;
    const auto pair_in_sink = pair_index % pairs_per_sink;
    const auto kv_head = pair_in_sink / pairs_per_head;
    const auto rope_pair = pair_in_sink % pairs_per_head;
    const auto element = sink * kv_dim + kv_head * static_cast<size_t>(head_dim) + 2 * rope_pair;

    auto first = to_float(cache_k[element]);
    auto second = to_float(cache_k[element + 1]);
    rotate_rope_pair(
        first,
        second,
        static_cast<int>(2 * rope_pair),
        1,
        rope_theta,
        rotary_dim
    );
    cache_k[element] = to_bfloat(first);
    cache_k[element + 1] = to_bfloat(second);
}

__global__ void qk_norm_rope_cache_kernel(
    float *q,
    float *k,
    const float *v,
    __nv_bfloat16 *cache_k,
    __nv_bfloat16 *cache_v,
    const __nv_bfloat16 *q_norm,
    const __nv_bfloat16 *k_norm,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    int kv_pos,
    float norm_eps,
    float theta,
    int rotary_dim,
    int batch_size
) {
    const int row = blockIdx.x;
    const int q_rows = batch_size * n_heads;
    const bool is_query = row < q_rows;
    const int row_in_type = is_query ? row : row - q_rows;
    const int heads = is_query ? n_heads : n_kv_heads;
    const int token = row_in_type / heads;
    const int head = row_in_type % heads;
    const int token_dim = heads * head_dim;
    float *values = (is_query ? q : k) +
        static_cast<size_t>(token) * token_dim + head * head_dim;
    const auto *norm = is_query ? q_norm : k_norm;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        sum_sq += values[i] * values[i];
    }
    extern __shared__ float shared_sum[];
    shared_sum[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(
        shared_sum[0] / static_cast<float>(head_dim) + norm_eps
    );
    const int position = pos + token;

    for (int pair = threadIdx.x; pair < head_dim / 2; pair += blockDim.x) {
        const int first_index = 2 * pair;
        const int second_index = first_index + 1;
        float first = values[first_index] * inverse_rms *
            to_float(norm[first_index]);
        float second = values[second_index] * inverse_rms *
            to_float(norm[second_index]);
        if (first_index < rotary_dim) {
            rotate_rope_pair(
                first, second, first_index, position, theta, rotary_dim
            );
        }
        values[first_index] = first;
        values[second_index] = second;

        if (!is_query) {
            const int kv_dim = n_kv_heads * head_dim;
            const auto cache_base =
                (static_cast<size_t>(kv_pos) + token) * kv_dim +
                head * head_dim;
            cache_k[cache_base + first_index] = to_bfloat(first);
            cache_k[cache_base + second_index] = to_bfloat(second);
            const auto value_base = static_cast<size_t>(token) * kv_dim +
                head * head_dim;
            cache_v[cache_base + first_index] =
                to_bfloat(v[value_base + first_index]);
            cache_v[cache_base + second_index] =
                to_bfloat(v[value_base + second_index]);
        }
    }

    if ((head_dim & 1) != 0 && threadIdx.x == 0) {
        const int last = head_dim - 1;
        const float normalized = values[last] * inverse_rms * to_float(norm[last]);
        values[last] = normalized;
        if (!is_query) {
            const int kv_dim = n_kv_heads * head_dim;
            const auto cache_index =
                (static_cast<size_t>(kv_pos) + token) * kv_dim +
                head * head_dim + last;
            cache_k[cache_index] = to_bfloat(normalized);
            cache_v[cache_index] = to_bfloat(
                v[static_cast<size_t>(token) * kv_dim + head * head_dim + last]
            );
        }
    }
}

__global__ void attention_scores_kernel(
    float *scores,
    const float *q,
    const __nv_bfloat16 *k,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int kv_len_start,
    int max_seq_len,
    int batch_size
) {
    const int row = blockIdx.x;
    const int batch = row / n_heads;
    const int head = row % n_heads;
    if (batch >= batch_size) {
        return;
    }
    const int kv_head = head / (n_heads / n_kv_heads);
    const int kv_len = kv_len_start + batch;
    const int cache_stride = n_kv_heads * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float *query = q + static_cast<size_t>(row) * head_dim;
    float *row_scores = scores + static_cast<size_t>(row) * max_seq_len;

    for (int token = threadIdx.x; token < kv_len; token += blockDim.x) {
        const auto *key = k + static_cast<size_t>(token) * cache_stride +
            kv_head * head_dim;
        float sum = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            sum += query[i] * to_float(key[i]);
        }
        row_scores[token] = sum * scale;
    }
}

__global__ void softmax_rows_kernel(
    float *scores,
    int n_heads,
    int kv_len_start,
    int max_seq_len,
    int batch_size
) {
    const int row = blockIdx.x;
    const int batch = row / n_heads;
    if (batch >= batch_size) {
        return;
    }
    const int kv_len = kv_len_start + batch;
    float *row_scores = scores + static_cast<size_t>(row) * max_seq_len;
    extern __shared__ float scratch[];

    float local_max = -FLT_MAX;
    for (int token = threadIdx.x; token < kv_len; token += blockDim.x) {
        local_max = fmaxf(local_max, row_scores[token]);
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] =
                fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float maximum = scratch[0];

    float local_sum = 0.0f;
    for (int token = threadIdx.x; token < kv_len; token += blockDim.x) {
        const float value = expf(row_scores[token] - maximum);
        row_scores[token] = value;
        local_sum += value;
    }
    scratch[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inverse_sum = 1.0f / scratch[0];
    for (int token = threadIdx.x; token < kv_len; token += blockDim.x) {
        row_scores[token] *= inverse_sum;
    }
}

__global__ void attention_output_kernel(
    float *out,
    const float *scores,
    const __nv_bfloat16 *v,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int kv_len_start,
    int max_seq_len,
    int batch_size
) {
    const auto index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto count =
        static_cast<size_t>(batch_size) * n_heads * head_dim;
    if (index >= count) {
        return;
    }
    const int component = index % head_dim;
    const int row = index / head_dim;
    const int batch = row / n_heads;
    const int head = row % n_heads;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int kv_len = kv_len_start + batch;
    const int cache_stride = n_kv_heads * head_dim;
    const float *row_scores = scores + static_cast<size_t>(row) * max_seq_len;

    float sum = 0.0f;
    for (int token = 0; token < kv_len; ++token) {
        const auto value_index = static_cast<size_t>(token) * cache_stride +
            kv_head * head_dim + component;
        sum += row_scores[token] * to_float(v[value_index]);
    }
    out[index] = sum;
}

__global__ void silu_multiply_kernel(float *first, const float *second, size_t count) {
    const auto index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = first[index];
        first[index] = value / (1.0f + expf(-value)) * second[index];
    }
}

__global__ void add_kernel(float *destination, const float *source, size_t count) {
    const auto index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        destination[index] += source[index];
    }
}

__global__ void embedding_kernel(
    float *out,
    const __nv_bfloat16 *embedding_row,
    int dim
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dim) {
        out[index] = to_float(embedding_row[index]);
    }
}

__global__ void argmax_kernel(
    std::int32_t *out,
    const float *values,
    int count
) {
    __shared__ float maxima[256];
    __shared__ int indices[256];
    float local_max = -FLT_MAX;
    int local_index = 0;
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        const float value = values[index];
        if (value > local_max || (value == local_max && index < local_index)) {
            local_max = value;
            local_index = index;
        }
    }
    maxima[threadIdx.x] = local_max;
    indices[threadIdx.x] = local_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_max = maxima[threadIdx.x + stride];
            const int other_index = indices[threadIdx.x + stride];
            if (other_max > maxima[threadIdx.x] ||
                (other_max == maxima[threadIdx.x] &&
                 other_index < indices[threadIdx.x])) {
                maxima[threadIdx.x] = other_max;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *out = indices[0];
    }
}

__device__
float silu(float x) {
    return x / (1.0f + expf(-x));
}

void matmul_gpu(
    float* out,
    const float* x,
    const std::bfloat16_t* weights,
    int n,
    int m,
    int batch_size
) {
    if (batch_size == 1) {
        constexpr int threads = 256;
        constexpr int outputs_per_block = threads / 32;
        const int blocks = (n + outputs_per_block - 1) / outputs_per_block;
        matvec_warp<<<blocks, threads>>>(
            x,
            reinterpret_cast<const __nv_bfloat16*>(weights),
            out,
            n,
            m
        );
    } else {
        const dim3 block(kTileSize, kTileSize);
        const dim3 grid(
            (n + kTileSize - 1) / kTileSize,
            (batch_size + kTileSize - 1) / kTileSize
        );
        matmul_tiled<<<grid, block>>>(
            x,
            reinterpret_cast<const __nv_bfloat16*>(weights),
            out,
            n,
            m,
            batch_size
        );
    }
    CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaGetLastError());
}

// I'm ignoring sink tokens in prefill
// this is decode only for now
void rotate_sink_tokens(
    std::bfloat16_t *cache_k,
    size_t num_sink,
    size_t kv_dim,
    int head_dim,
    float rope_theta,
    int rotary_dim
) {
    if (num_sink == 0 || rotary_dim == 0) {
        return;
    }

    constexpr int threads = 256;
    const auto num_kv_heads = kv_dim / static_cast<size_t>(head_dim);
    const auto total_pairs =
        num_sink * num_kv_heads * static_cast<size_t>(rotary_dim / 2);
    const auto blocks = static_cast<unsigned int>(
        (total_pairs + threads - 1) / threads
    );

    rotate_sink_tokens_kernel<<<blocks, threads>>>(
        reinterpret_cast<__nv_bfloat16*>(cache_k),
        num_sink,
        kv_dim,
        head_dim,
        rope_theta,
        rotary_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

void qk_norm_rope_and_update_cache(
    float *q,
    float *k,
    const float *v,
    std::bfloat16_t *cache_k,
    std::bfloat16_t *cache_v,
    const std::bfloat16_t *q_norm,
    const std::bfloat16_t *k_norm,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    int kv_pos,
    float norm_eps,
    float theta,
    int rotary_dim,
    int batch_size
) {
    constexpr int threads = 256;
    const int blocks = batch_size * (n_heads + n_kv_heads);
    qk_norm_rope_cache_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        q,
        k,
        v,
        reinterpret_cast<__nv_bfloat16*>(cache_k),
        reinterpret_cast<__nv_bfloat16*>(cache_v),
        reinterpret_cast<const __nv_bfloat16*>(q_norm),
        reinterpret_cast<const __nv_bfloat16*>(k_norm),
        n_heads,
        n_kv_heads,
        head_dim,
        pos,
        kv_pos,
        norm_eps,
        theta,
        rotary_dim,
        batch_size
    );
    CUDA_CHECK(cudaGetLastError());
}

void attn_gpu(
    float *out,
    float *atth,
    const float *q,
    const std::bfloat16_t *k,
    const std::bfloat16_t *v,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int kv_len_start,
    int max_seq_len,
    int batch_size
) {
    constexpr int threads = 256;
    const int rows = batch_size * n_heads;
    attention_scores_kernel<<<rows, threads>>>(
        atth,
        q,
        reinterpret_cast<const __nv_bfloat16*>(k),
        head_dim,
        n_heads,
        n_kv_heads,
        kv_len_start,
        max_seq_len,
        batch_size
    );
    CUDA_CHECK(cudaGetLastError());
    softmax_rows_kernel<<<rows, threads, threads * sizeof(float)>>>(
        atth, n_heads, kv_len_start, max_seq_len, batch_size
    );
    CUDA_CHECK(cudaGetLastError());
    const auto count = static_cast<size_t>(rows) * head_dim;
    const auto blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    attention_output_kernel<<<blocks, threads>>>(
        out,
        atth,
        reinterpret_cast<const __nv_bfloat16*>(v),
        head_dim,
        n_heads,
        n_kv_heads,
        kv_len_start,
        max_seq_len,
        batch_size
    );
    CUDA_CHECK(cudaGetLastError());
}

void ffn_gpu(
    float *out,
    float *lin1,
    float *lin2,
    const float *x,
    const std::bfloat16_t *w1,
    const std::bfloat16_t *w2,
    const std::bfloat16_t *w3,
    int hidden_dim,
    int dim,
    int batch_size
) {
    matmul_gpu(lin1, x, w1, hidden_dim, dim, batch_size);
    matmul_gpu(lin2, x, w3, hidden_dim, dim, batch_size);
    constexpr int threads = 256;
    const auto count = static_cast<size_t>(batch_size) * hidden_dim;
    const auto blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    silu_multiply_kernel<<<blocks, threads>>>(lin1, lin2, count);
    CUDA_CHECK(cudaGetLastError());
    matmul_gpu(out, lin1, w2, dim, hidden_dim, batch_size);
}

void add_gpu(float *destination, const float *source, size_t count) {
    if (count == 0) {
        return;
    }
    constexpr int threads = 256;
    const auto blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    add_kernel<<<blocks, threads>>>(destination, source, count);
    CUDA_CHECK(cudaGetLastError());
}

void embedding_gpu(
    float *out,
    const std::bfloat16_t *embedding_row,
    int dim
) {
    constexpr int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    embedding_kernel<<<blocks, threads>>>(
        out,
        reinterpret_cast<const __nv_bfloat16*>(embedding_row),
        dim
    );
    CUDA_CHECK(cudaGetLastError());
}

void argmax_gpu(std::int32_t *out, const float *values, int count) {
    argmax_kernel<<<1, 256>>>(out, values, count);
    CUDA_CHECK(cudaGetLastError());
}
#undef CUDA_CHECK
