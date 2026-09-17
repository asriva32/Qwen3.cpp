#include <cuda_runtime.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <stdfloat>
#include <string>
#include <vector>
#include "layers.h"

namespace {

constexpr int kSkipped = 77;

void CheckCuda(cudaError_t error, const char* expression) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA call failed: ") + expression + ": " +
            cudaGetErrorString(error)
        );
    }
}

#define CHECK_CUDA(expression) CheckCuda((expression), #expression)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        CHECK_CUDA(cudaMalloc(&data_, count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }

private:
    T* data_ = nullptr;
};

template <typename T>
void CopyToDevice(DeviceBuffer<T>& destination, const T* source, size_t count) {
    CHECK_CUDA(cudaMemcpy(
        destination.data(),
        source,
        count * sizeof(T),
        cudaMemcpyHostToDevice
    ));
}

template <typename T>
std::vector<T> CopyFromDevice(const DeviceBuffer<T>& source, size_t count) {
    std::vector<T> values(count);
    CHECK_CUDA(cudaMemcpy(
        values.data(),
        source.data(),
        count * sizeof(T),
        cudaMemcpyDeviceToHost
    ));
    return values;
}

void ExpectNear(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float absolute_tolerance,
    float relative_tolerance,
    const char* test_name
) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error(std::string(test_name) + " output size mismatch");
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        const float difference = std::abs(expected[i] - actual[i]);
        const float tolerance = absolute_tolerance +
            relative_tolerance * std::abs(expected[i]);
        if (!std::isfinite(actual[i]) || difference > tolerance) {
            throw std::runtime_error(
                std::string(test_name) + " mismatch at element " +
                std::to_string(i) + ": CPU=" + std::to_string(expected[i]) +
                ", GPU=" + std::to_string(actual[i]) +
                ", tolerance=" + std::to_string(tolerance)
            );
        }
    }
}

std::vector<float> MakeInput(size_t count) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        const int centered = static_cast<int>((i * 17 + 5) % 31) - 15;
        values[i] = static_cast<float>(centered) * 0.0625f;
    }
    return values;
}

std::vector<std::uint16_t> MakeWeights(size_t count) {
    std::vector<std::uint16_t> values(count);
    for (size_t i = 0; i < count; ++i) {
        const int centered = static_cast<int>((i * 11 + 3) % 23) - 11;
        const float value =
            1.0f + static_cast<float>(centered) * 0.03125f;
        values[i] = static_cast<std::uint16_t>(
            std::bit_cast<std::uint32_t>(value) >> 16
        );
    }
    return values;
}

std::vector<std::uint16_t> MakeBfloatInput(size_t count) {
    std::vector<std::uint16_t> values(count);
    for (size_t i = 0; i < count; ++i) {
        const int centered = static_cast<int>((i * 13 + 2) % 19) - 9;
        const float value = static_cast<float>(centered) * 0.125f;
        values[i] = static_cast<std::uint16_t>(
            std::bit_cast<std::uint32_t>(value) >> 16
        );
    }
    return values;
}

std::vector<float> BfloatToFloat(const std::vector<std::uint16_t>& values) {
    std::vector<float> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = std::bit_cast<float>(
            static_cast<std::uint32_t>(values[i]) << 16
        );
    }
    return result;
}

void TestRmsNorm(int n, int batch_size) {
    constexpr float eps = 1.0e-6f;
    const auto input = MakeInput(static_cast<size_t>(n) * batch_size);
    const auto weights = MakeWeights(n);
    std::vector<float> expected(input.size());
    std::vector<float> actual(input.size());

    rmsnorm_cpu(
        expected.data(), input.data(),
        reinterpret_cast<const std::bfloat16_t*>(weights.data()),
        eps, n, batch_size
    );

    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<std::uint16_t> device_weights(n);
    DeviceBuffer<float> device_output(actual.size());
    CopyToDevice(device_input, input.data(), input.size());
    CopyToDevice(device_weights, weights.data(), n);

    rmsnorm_gpu(
        device_output.data(),
        device_input.data(),
        reinterpret_cast<const std::bfloat16_t*>(device_weights.data()),
        eps,
        n,
        batch_size
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(
        actual.data(),
        device_output.data(),
        actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    ExpectNear(expected, actual, 2.0e-5f, 2.0e-5f, "rmsnorm");
}

void TestMatmul(int n, int m, int batch_size) {
    const auto input = MakeInput(static_cast<size_t>(m) * batch_size);
    const size_t weight_count = static_cast<size_t>(n) * m;
    const auto weights = MakeWeights(weight_count);
    std::vector<float> expected(static_cast<size_t>(n) * batch_size);
    std::vector<float> actual(expected.size());

    matmul_cpu(
        expected.data(), input.data(),
        reinterpret_cast<const std::bfloat16_t*>(weights.data()),
        n, m, batch_size
    );

    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<std::uint16_t> device_weights(weight_count);
    DeviceBuffer<float> device_output(actual.size());
    CopyToDevice(device_input, input.data(), input.size());
    CopyToDevice(device_weights, weights.data(), weight_count);

    matmul_gpu(
        device_output.data(),
        device_input.data(),
        reinterpret_cast<const std::bfloat16_t*>(device_weights.data()),
        n,
        m,
        batch_size
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(
        actual.data(),
        device_output.data(),
        actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    ExpectNear(expected, actual, 1.0e-4f, 1.0e-4f, "matmul");
}

void TestRopeAndCache() {
    constexpr int batch_size = 2;
    constexpr int head_dim = 4;
    constexpr int q_dim = 8;
    constexpr int kv_dim = 4;
    constexpr int max_seq_len = 5;
    constexpr int position = 2;
    constexpr int cache_position = 1;
    constexpr float theta = 10000.0f;

    const auto q = MakeInput(batch_size * q_dim);
    const auto k = MakeInput(batch_size * kv_dim);
    const auto v = MakeInput(batch_size * kv_dim);
    const auto q_norm = MakeWeights(head_dim);
    const auto k_norm = MakeWeights(head_dim);
    auto expected_q = q;
    auto expected_k = k;
    for (int token = 0; token < batch_size; ++token) {
        for (int head = 0; head < q_dim / head_dim; ++head) {
            auto *row = expected_q.data() + token * q_dim + head * head_dim;
            rmsnorm_cpu(
                row, row,
                reinterpret_cast<const std::bfloat16_t*>(q_norm.data()),
                1.0e-6f, head_dim
            );
        }
        for (int head = 0; head < kv_dim / head_dim; ++head) {
            auto *row = expected_k.data() + token * kv_dim + head * head_dim;
            rmsnorm_cpu(
                row, row,
                reinterpret_cast<const std::bfloat16_t*>(k_norm.data()),
                1.0e-6f, head_dim
            );
        }
        rope_cpu(
            expected_q.data() + token * q_dim,
            q_dim, head_dim, position + token, theta, head_dim
        );
        rope_cpu(
            expected_k.data() + token * kv_dim,
            kv_dim, head_dim, position + token, theta, head_dim
        );
    }

    DeviceBuffer<float> device_q(q.size());
    DeviceBuffer<float> device_k(k.size());
    DeviceBuffer<float> device_v(v.size());
    DeviceBuffer<std::uint16_t> device_cache_k(max_seq_len * kv_dim);
    DeviceBuffer<std::uint16_t> device_cache_v(max_seq_len * kv_dim);
    DeviceBuffer<std::uint16_t> device_q_norm(head_dim);
    DeviceBuffer<std::uint16_t> device_k_norm(head_dim);
    CopyToDevice(device_q, q.data(), q.size());
    CopyToDevice(device_k, k.data(), k.size());
    CopyToDevice(device_v, v.data(), v.size());
    CopyToDevice(device_q_norm, q_norm.data(), q_norm.size());
    CopyToDevice(device_k_norm, k_norm.data(), k_norm.size());
    CHECK_CUDA(cudaMemset(device_cache_k.data(), 0, max_seq_len * kv_dim * sizeof(std::uint16_t)));
    CHECK_CUDA(cudaMemset(device_cache_v.data(), 0, max_seq_len * kv_dim * sizeof(std::uint16_t)));

    qk_norm_rope_and_update_cache(
        device_q.data(),
        device_k.data(),
        device_v.data(),
        reinterpret_cast<std::bfloat16_t*>(device_cache_k.data()),
        reinterpret_cast<std::bfloat16_t*>(device_cache_v.data()),
        reinterpret_cast<const std::bfloat16_t*>(device_q_norm.data()),
        reinterpret_cast<const std::bfloat16_t*>(device_k_norm.data()),
        q_dim / head_dim, kv_dim / head_dim, head_dim,
        position, cache_position, 1.0e-6f, theta, head_dim, batch_size
    );

    ExpectNear(expected_q, CopyFromDevice(device_q, q.size()), 2.0e-5f, 2.0e-5f, "rope q");
    ExpectNear(expected_k, CopyFromDevice(device_k, k.size()), 2.0e-5f, 2.0e-5f, "rope k");

    const auto cache_k = BfloatToFloat(
        CopyFromDevice(device_cache_k, max_seq_len * kv_dim)
    );
    const auto cache_v = BfloatToFloat(
        CopyFromDevice(device_cache_v, max_seq_len * kv_dim)
    );
    for (int token = 0; token < batch_size; ++token) {
        for (int i = 0; i < kv_dim; ++i) {
            const auto cache_index = (cache_position + token) * kv_dim + i;
            const float expected_cache_k = static_cast<float>(
                static_cast<std::bfloat16_t>(expected_k[token * kv_dim + i])
            );
            const float expected_cache_v = static_cast<float>(
                static_cast<std::bfloat16_t>(v[token * kv_dim + i])
            );
            if (cache_k[cache_index] != expected_cache_k ||
                cache_v[cache_index] != expected_cache_v) {
                throw std::runtime_error("KV cache update mismatch");
            }
        }
    }
}

void TestArgmax() {
    std::vector<float> values(1027, -4.0f);
    values[17] = 9.0f;
    values[901] = 9.0f;
    DeviceBuffer<float> device_values(values.size());
    DeviceBuffer<std::int32_t> device_output(1);
    CopyToDevice(device_values, values.data(), values.size());
    argmax_gpu(device_output.data(), device_values.data(), values.size());
    const auto actual = CopyFromDevice(device_output, 1);
    if (actual[0] != 17) {
        throw std::runtime_error("GPU argmax did not select the first maximum");
    }
}

void TestRotateSinkTokens() {
    constexpr int num_sink = 2;
    constexpr int kv_dim = 8;
    constexpr int head_dim = 4;
    constexpr int rotary_dim = 4;
    constexpr float theta = 10000.0f;
    auto cache = MakeBfloatInput(num_sink * kv_dim);
    auto expected = BfloatToFloat(cache);
    for (int sink = 0; sink < num_sink; ++sink) {
        rope_cpu(
            expected.data() + sink * kv_dim,
            kv_dim, head_dim, 1, theta, rotary_dim
        );
        for (int i = 0; i < kv_dim; ++i) {
            expected[sink * kv_dim + i] = static_cast<float>(
                static_cast<std::bfloat16_t>(expected[sink * kv_dim + i])
            );
        }
    }

    DeviceBuffer<std::uint16_t> device_cache(cache.size());
    CopyToDevice(device_cache, cache.data(), cache.size());
    rotate_sink_tokens(
        reinterpret_cast<std::bfloat16_t*>(device_cache.data()),
        num_sink, kv_dim, head_dim, theta, rotary_dim
    );
    const auto actual = BfloatToFloat(
        CopyFromDevice(device_cache, cache.size())
    );
    ExpectNear(expected, actual, 0.0f, 0.0f, "sink rotation");
}

void TestAttention() {
    constexpr int batch_size = 2;
    constexpr int head_dim = 4;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int kv_len_start = 2;
    constexpr int max_seq_len = 4;
    constexpr int q_dim = n_heads * head_dim;
    constexpr int kv_dim = n_kv_heads * head_dim;
    const auto q = MakeInput(batch_size * q_dim);
    const auto k = MakeBfloatInput(max_seq_len * kv_dim);
    const auto v = MakeBfloatInput(max_seq_len * kv_dim);
    std::vector<float> expected(batch_size * q_dim);
    std::vector<float> expected_scores(
        batch_size * n_heads * max_seq_len
    );
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < n_heads; ++head) {
            const int kv_head = head / (n_heads / n_kv_heads);
            attn_cpu(
                expected.data() + batch * q_dim + head * head_dim,
                expected_scores.data() +
                    (batch * n_heads + head) * max_seq_len,
                q.data() + batch * q_dim + head * head_dim,
                reinterpret_cast<const std::bfloat16_t*>(k.data()) +
                    kv_head * head_dim,
                reinterpret_cast<const std::bfloat16_t*>(v.data()) +
                    kv_head * head_dim,
                head_dim, n_kv_heads, kv_len_start + batch
            );
        }
    }

    DeviceBuffer<float> device_q(q.size());
    DeviceBuffer<std::uint16_t> device_k(k.size());
    DeviceBuffer<std::uint16_t> device_v(v.size());
    DeviceBuffer<float> device_scores(expected_scores.size());
    DeviceBuffer<float> device_output(expected.size());
    CopyToDevice(device_q, q.data(), q.size());
    CopyToDevice(device_k, k.data(), k.size());
    CopyToDevice(device_v, v.data(), v.size());
    attn_gpu(
        device_output.data(), device_scores.data(), device_q.data(),
        reinterpret_cast<const std::bfloat16_t*>(device_k.data()),
        reinterpret_cast<const std::bfloat16_t*>(device_v.data()),
        head_dim, n_heads, n_kv_heads, kv_len_start, max_seq_len, batch_size
    );
    ExpectNear(
        expected,
        CopyFromDevice(device_output, expected.size()),
        2.0e-5f, 2.0e-5f, "attention"
    );
}

void TestFfnAndAdd() {
    constexpr int batch_size = 3;
    constexpr int dim = 5;
    constexpr int hidden_dim = 7;
    const auto input = MakeInput(batch_size * dim);
    const auto w1 = MakeWeights(hidden_dim * dim);
    const auto w2 = MakeWeights(dim * hidden_dim);
    const auto w3 = MakeWeights(hidden_dim * dim);
    std::vector<float> expected(batch_size * dim);
    std::vector<float> expected_lin1(batch_size * hidden_dim);
    std::vector<float> expected_lin2(batch_size * hidden_dim);
    ffn_cpu(
        expected.data(), expected_lin1.data(), expected_lin2.data(),
        input.data(), reinterpret_cast<const std::bfloat16_t*>(w1.data()),
        reinterpret_cast<const std::bfloat16_t*>(w2.data()),
        reinterpret_cast<const std::bfloat16_t*>(w3.data()),
        hidden_dim, dim, batch_size
    );

    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<float> device_output(expected.size());
    DeviceBuffer<float> device_lin1(expected_lin1.size());
    DeviceBuffer<float> device_lin2(expected_lin2.size());
    DeviceBuffer<std::uint16_t> device_w1(w1.size());
    DeviceBuffer<std::uint16_t> device_w2(w2.size());
    DeviceBuffer<std::uint16_t> device_w3(w3.size());
    CopyToDevice(device_input, input.data(), input.size());
    CopyToDevice(device_w1, w1.data(), w1.size());
    CopyToDevice(device_w2, w2.data(), w2.size());
    CopyToDevice(device_w3, w3.data(), w3.size());
    ffn_gpu(
        device_output.data(), device_lin1.data(), device_lin2.data(),
        device_input.data(),
        reinterpret_cast<const std::bfloat16_t*>(device_w1.data()),
        reinterpret_cast<const std::bfloat16_t*>(device_w2.data()),
        reinterpret_cast<const std::bfloat16_t*>(device_w3.data()),
        hidden_dim, dim, batch_size
    );
    ExpectNear(
        expected,
        CopyFromDevice(device_output, expected.size()),
        2.0e-4f, 2.0e-4f, "ffn"
    );

    add_gpu(device_output.data(), device_input.data(), input.size());
    auto expected_added = expected;
    for (size_t i = 0; i < input.size(); ++i) {
        expected_added[i] += input[i];
    }
    ExpectNear(
        expected_added,
        CopyFromDevice(device_output, expected.size()),
        2.0e-4f, 2.0e-4f, "residual add"
    );
}

void TestEmbedding() {
    constexpr int dim = 19;
    const auto embedding = MakeBfloatInput(dim);
    const auto expected = BfloatToFloat(embedding);
    DeviceBuffer<std::uint16_t> device_embedding(dim);
    DeviceBuffer<float> device_output(dim);
    CopyToDevice(device_embedding, embedding.data(), embedding.size());
    embedding_gpu(
        device_output.data(),
        reinterpret_cast<const std::bfloat16_t*>(device_embedding.data()),
        dim
    );
    ExpectNear(
        expected, CopyFromDevice(device_output, dim),
        0.0f, 0.0f, "embedding"
    );
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t device_error = cudaGetDeviceCount(&device_count);
    if (device_error != cudaSuccess || device_count == 0) {
        std::cout << "Skipping CUDA kernel tests: "
                  << (device_error == cudaSuccess
                          ? "no CUDA device found"
                          : cudaGetErrorString(device_error))
                  << '\n';
        return kSkipped;
    }

    try {
        CHECK_CUDA(cudaSetDevice(0));

        // Cover a partial warp-sized row, multiple batches, and multiple passes
        // through the RMSNorm input loop.
        TestRmsNorm(13, 1);
        TestRmsNorm(37, 3);
        TestRmsNorm(513, 2);

        // Cover exact tiles as well as partial input, output, and batch tiles.
        TestMatmul(16, 16, 1);
        TestMatmul(19, 23, 3);
        TestMatmul(7, 33, 17);
        TestRopeAndCache();
        TestRotateSinkTokens();
        TestAttention();
        TestFfnAndAdd();
        TestEmbedding();
        TestArgmax();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    std::cout << "CUDA kernels match CPU references\n";
    return 0;
}

#undef CHECK_CUDA
