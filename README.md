# Qwen3.cpp

Qwen3.cpp is an experimental C++23 CPU inference engine for Qwen3. It builds a
standalone native executable without a scripting runtime or language bindings.

> [!IMPORTANT]
> This project is alpha software. It currently supports Linux on x86-64 CPUs
> with AVX2 and FMA, and Qwen3 models converted to the project's BF16 format.

## Requirements

- Linux or WSL2 on an x86-64 CPU with AVX2 and FMA
- GCC 14 or newer, or a compatible recent Clang/libstdc++ toolchain
- CMake 3.20 or newer
- An OpenMP development runtime

## Build and test

```sh
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake
ctest --test-dir build/cmake --output-on-failure
```

Add `-DQWEN3_NATIVE_ARCH=ON` to optimize for the build machine. The resulting
binary may then require CPU features unavailable on other machines.

## Run inference

The native interface includes a byte-level BPE tokenizer and applies Qwen's
chat template by default:

```sh
./build/cmake/qwen3 \
  --model Qwen3.bin \
  --prompt "Hello" \
  --context-length 512 \
  --max-tokens 128 \
  --temperature 0.6 \
  --threads 8
```

The executable writes decoded text followed by prefill and decode statistics.
Use `--raw` to skip the chat template, `--tokens` to supply exact input token
IDs, or `--no-eos` to run exactly the requested number of decode steps.

Temperature sampling is enabled by default at the Qwen-recommended value of
`0.6`. Use `--seed N` for reproducible output or `--greedy` for argmax decoding.

Run `./build/cmake/qwen3 --help` for all options.

## Current limitations

- Qwen3 BF16 models only
- The dependency-free tokenizer implements Qwen's ranked byte-level BPE; its
  built-in Unicode category handling is intentionally lightweight
- AVX2/FMA x86-64 CPUs only
- Linux/WSL2 only

## Benchmarks

On an AMD Ryzen AI 9 HX 370 with 32 GB DDR5-7500, using Qwen3 0.6B BF16 and
eight threads:

| Engine | Prefill | Decode |
| --- | ---: | ---: |
| llama.cpp | 303.27 tok/s | 39.42 tok/s |
| Qwen3.cpp | 43.20 tok/s | 40.62 tok/s |

## License

MIT. See [LICENSE](LICENSE).
