# Qwen3.cpp
Single file inference of Qwen3 0.6B
Currently has a dependency on https://github.com/mlc-ai/tokenizers-cpp (probably need rust installed)
Will try to remove this dependency in the future

## Requirements

- Use Linux/WSL2
- Need gcc version that supports C++ 23
- Need rust installed

## Build

```sh
cmake -S . -B build/cmake
cmake --build build/cmake
```

Run the loader test with:

```sh
ctest --test-dir build/cmake --output-on-failure
```

Run the CLI with:

```sh
./build/cmake/qwen3 Qwen3.bin "Hello" 512
```

Arguments are the model path, prompt, and runtime context length. Generation
stops at EOS or after 512 generated tokens by default. Set a different limit
with:

```sh
./build/cmake/qwen3 Qwen3.bin "Hello" 512 --max-tokens 128
```

For repeatable throughput measurements, `--benchmark N` performs exactly `N`
decode steps and ignores EOS:

```sh
./build/cmake/qwen3 Qwen3.bin "Hello" 512 --benchmark 64
```

Add `--raw` to either command to skip Qwen's chat template. The CLI reports
prefill and decode throughput plus the reason generation stopped.

To specify number of threads to use, `--threads N` uses N threads. It is recommended to play around with this number for maximum performance.

## Comparision with llama.cpp
CPU: AMD Ryzen AI 9 HX 370
RAM: 32GB DDR5 at 7500 MT/s
Tested both using bfloat16 on 8 threads with 5 repetitions:

- llama.cpp
    - Prefill: 44.82 tok/s
    - Decode: 301.61 tok/s
- Qwen3.cpp
    - Prefill: 40.53 tok/s
    - Decode: 39.91 tok/s

Prefill speed significantly worse because it isn't batched but decode achieves similar or better performance when tuned for my system.