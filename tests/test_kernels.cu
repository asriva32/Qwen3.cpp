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
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    std::cout << "CUDA RMSNorm and matmul kernels match CPU references\n";
    return 0;
}

#undef CHECK_CUDA
