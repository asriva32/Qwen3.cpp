# Qwen3.cpp
Single file inference of Qwen3 0.6B
Currently has a dependency on https://github.com/mlc-ai/tokenizers-cpp (probably need rust installed)
Will try to remove this dependency in the future
Or take tokens from hugging face tokenizer
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

Arguments are the model path, prompt, and runtime context length. Add `--raw`
as the fourth argument to skip Qwen's chat template. Generation continues until
the model emits its EOS token. The CLI reports prefill and decode throughput.

Inference currently uses FP32 weights, a single CPU thread, and greedy sampling.
The runtime context defaults to 512 tokens so the KV caches do not allocate the
model's full 40,960-token context.
