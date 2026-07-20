#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$root_dir/build/cmake"

if [[ -d "$HOME/.cargo/bin" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

cmake -S "$root_dir" -B "$build_dir"
cmake --build "$build_dir" --target test_loader
ctest --test-dir "$build_dir" --output-on-failure
