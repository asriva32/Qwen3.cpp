#!/usr/bin/env python3
"""
Convert a Hugging Face Qwen3 checkpoint into a simple binary file for this repo.

The converter is intentionally similar in spirit to yalm's convert.py:
  - read config.json and tokenizer.json from a Hugging Face model directory
  - normalize model metadata into one compact header
  - normalize Hugging Face tensor names into runtime-oriented names
  - append tokenizer.json and compatibility tokenizer bytes

File format, little endian:
  char[8]   magic: b"QWEN3CP\\0"
  uint32    format version, currently 1
  uint64    metadata JSON byte length
  uint8[]   UTF-8 metadata JSON
  uint64    tensor count

  Repeated tensor records:
    uint32  tensor name byte length
    uint8[] tensor name UTF-8
    uint32  dtype enum: 1=f32, 2=f16, 3=bf16, 4=u8, 5=i32
    uint32  number of dimensions
    uint64[] dimensions
    uint64  raw data byte length
    uint8[] raw contiguous row-major tensor bytes
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import safetensors
import torch


MAGIC = b"QWEN3CP\0"
VERSION = 1
SUPPORTED_ARCHITECTURES = {"Qwen3ForCausalLM"}
SUPPORTED_DTYPES = {"fp32", "fp16", "bf16"}
DTYPE_ENUM = {
    torch.float32: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
    torch.uint8: 4,
    torch.int32: 5,
}


@dataclass(frozen=True)
class Metadata:
    arch: str
    dtype: str
    dim: int
    hidden_dim: int
    head_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    max_seq_len: int
    bos_token_id: int
    eos_token_id: int
    rope_theta: float
    rotary_dim: int
    norm_eps: float
    act_type: str
    tie_word_embeddings: bool
    attention_bias: bool
    qk_norm: bool

    @classmethod
    def from_config(cls, config: dict, dtype: str) -> "Metadata":
        arch = config["architectures"][0]
        if arch not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Architecture {arch} is not supported; expected one of "
                f"{sorted(SUPPORTED_ARCHITECTURES)}"
            )
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of {sorted(SUPPORTED_DTYPES)}")
        if config.get("attention_bias", False):
            raise ValueError("attention_bias=true is not supported by this converter")
        if config.get("hidden_act") != "silu":
            raise ValueError(f"Expected Qwen3 hidden_act=silu, got {config.get('hidden_act')}")
        if config.get("rope_scaling") is not None:
            raise ValueError("rope_scaling is not supported yet")

        head_dim = int(config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]))
        return cls(
            arch=arch,
            dtype=dtype,
            dim=int(config["hidden_size"]),
            hidden_dim=int(config["intermediate_size"]),
            head_dim=head_dim,
            n_layers=int(config["num_hidden_layers"]),
            n_heads=int(config["num_attention_heads"]),
            n_kv_heads=int(config["num_key_value_heads"]),
            vocab_size=int(config["vocab_size"]),
            max_seq_len=int(config["max_position_embeddings"]),
            bos_token_id=int(config["bos_token_id"]),
            eos_token_id=int(config["eos_token_id"]),
            rope_theta=float(config.get("rope_theta", 10000.0)),
            rotary_dim=head_dim,
            norm_eps=float(config["rms_norm_eps"]),
            act_type=str(config["hidden_act"]),
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
            attention_bias=bool(config.get("attention_bias", False)),
            qk_norm=True,
        )

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode("utf-8")


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """ByteLevel tokenizer byte-to-unicode table used by GPT-2/Qwen tokenizers."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(n) for n in cs)))


def load_token_bytes(tokenizer_path: Path, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return concatenated NUL-terminated token bytes and int32 start offsets."""
    tokenizer = json.loads(tokenizer_path.read_text())
    vocab = tokenizer["model"]["vocab"]
    if len(vocab) > vocab_size:
        raise ValueError(f"tokenizer vocab has {len(vocab)} entries, config vocab_size={vocab_size}")

    tokens = [""] * vocab_size
    for token, idx in vocab.items():
        tokens[int(idx)] = token
    for token in tokenizer.get("added_tokens", []):
        idx = int(token["id"])
        if 0 <= idx < vocab_size:
            tokens[idx] = token["content"]

    use_byte_level_decode = not tokenizer["model"].get("byte_fallback", False)
    byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    offsets: list[int] = []
    data = bytearray()
    for token in tokens:
        offsets.append(len(data))
        if use_byte_level_decode:
            raw = bytes(byte_decoder.get(ch, 0) for ch in token)
        else:
            raw = token.replace("\u2581", " ").encode("utf-8")
        raw = raw.replace(b"\0", b"\7")
        data.extend(raw)
        data.append(0)

    return (
        torch.tensor(data, dtype=torch.uint8),
        torch.tensor(offsets, dtype=torch.int32),
    )


def load_file_bytes(path: Path) -> torch.Tensor:
    return torch.tensor(list(path.read_bytes()), dtype=torch.uint8)


def model_files(input_dir: Path) -> list[Path]:
    index_path = input_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        names = sorted(set(index["weight_map"].values()))
        return [input_dir / name for name in names]

    files = sorted(input_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {input_dir}")
    return files


class SafeTensorStore:
    def __init__(self, paths: Iterable[Path]):
        self._tensors: dict[str, tuple[Path, str]] = {}
        for path in paths:
            with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in self._tensors:
                        raise ValueError(f"Duplicate tensor {key} in {path}")
                    self._tensors[key] = (path, key)

    def get(self, name: str) -> torch.Tensor:
        try:
            path, key = self._tensors[name]
        except KeyError as exc:
            raise KeyError(f"Missing tensor {name}") from exc
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def has(self, name: str) -> bool:
        return name in self._tensors


def target_dtype(dtype: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype]


def convert_weight(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    return tensor.to(target_dtype(dtype)).contiguous()


def tensor_plan(metadata: Metadata) -> list[tuple[str, str]]:
    plan: list[tuple[str, str]] = [
        ("model.embed.weight", "model.embed_tokens.weight"),
    ]
    for layer in range(metadata.n_layers):
        src_prefix = f"model.layers.{layer}"
        dst_prefix = f"model.layers.{layer}"
        plan.extend(
            [
                (f"{dst_prefix}.attn.norm.weight", f"{src_prefix}.input_layernorm.weight"),
                (f"{dst_prefix}.attn.q_norm.weight", f"{src_prefix}.self_attn.q_norm.weight"),
                (f"{dst_prefix}.attn.k_norm.weight", f"{src_prefix}.self_attn.k_norm.weight"),
                (f"{dst_prefix}.attn.wq.weight", f"{src_prefix}.self_attn.q_proj.weight"),
                (f"{dst_prefix}.attn.wk.weight", f"{src_prefix}.self_attn.k_proj.weight"),
                (f"{dst_prefix}.attn.wv.weight", f"{src_prefix}.self_attn.v_proj.weight"),
                (f"{dst_prefix}.attn.wo.weight", f"{src_prefix}.self_attn.o_proj.weight"),
                (f"{dst_prefix}.mlp.norm.weight", f"{src_prefix}.post_attention_layernorm.weight"),
                (f"{dst_prefix}.mlp.w1.weight", f"{src_prefix}.mlp.gate_proj.weight"),
                (f"{dst_prefix}.mlp.w2.weight", f"{src_prefix}.mlp.down_proj.weight"),
                (f"{dst_prefix}.mlp.w3.weight", f"{src_prefix}.mlp.up_proj.weight"),
            ]
        )
    plan.append(("model.norm.weight", "model.norm.weight"))
    if not metadata.tie_word_embeddings:
        plan.append(("model.output.weight", "lm_head.weight"))
    return plan


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.view(torch.uint8).numpy().tobytes()


def write_tensor(out: BinaryIO, name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype not in DTYPE_ENUM:
        raise ValueError(f"Unsupported tensor dtype for {name}: {tensor.dtype}")
    name_bytes = name.encode("utf-8")
    out.write(struct.pack("<I", len(name_bytes)))
    out.write(name_bytes)
    out.write(struct.pack("<II", DTYPE_ENUM[tensor.dtype], tensor.ndim))
    for dim in tensor.shape:
        out.write(struct.pack("<Q", int(dim)))
    raw = tensor_bytes(tensor.contiguous())
    out.write(struct.pack("<Q", len(raw)))
    out.write(raw)


def convert(input_dir: Path, output_path: Path, dtype: str) -> None:
    config_path = input_dir / "config.json"
    tokenizer_path = input_dir / "tokenizer.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing {tokenizer_path}")

    config = json.loads(config_path.read_text())
    metadata = Metadata.from_config(config, dtype)
    store = SafeTensorStore(model_files(input_dir))
    plan = tensor_plan(metadata)
    tokenizer_json = load_file_bytes(tokenizer_path)
    token_data, token_offsets = load_token_bytes(tokenizer_path, metadata.vocab_size)

    tensor_count = len(plan) + 3
    metadata_bytes = metadata.to_json_bytes()

    with output_path.open("wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<IQ", VERSION, len(metadata_bytes)))
        out.write(metadata_bytes)
        out.write(struct.pack("<Q", tensor_count))

        for idx, (dst, src) in enumerate(plan, 1):
            tensor = convert_weight(store.get(src), dtype)
            print(f"[{idx}/{tensor_count}] {dst} {tuple(tensor.shape)} {tensor.dtype}")
            write_tensor(out, dst, tensor)

        print(f"[{len(plan) + 1}/{tensor_count}] tokenizer.json {tuple(tokenizer_json.shape)} uint8")
        write_tensor(out, "tokenizer.json", tokenizer_json)
        print(f"[{len(plan) + 2}/{tensor_count}] tokenizer.tokens {tuple(token_data.shape)} uint8")
        write_tensor(out, "tokenizer.tokens", token_data)
        print(f"[{len(plan) + 3}/{tensor_count}] tokenizer.offsets {tuple(token_offsets.shape)} int32")
        write_tensor(out, "tokenizer.offsets", token_offsets)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face Qwen3 model directory into a .qwen3 binary."
    )
    parser.add_argument("output", type=Path, help="Output .qwen3 file")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("Qwen3-0.6B"),
        help="Input Hugging Face model directory, default: Qwen3-0.6B",
    )
    parser.add_argument("--dtype", choices=sorted(SUPPORTED_DTYPES), default="fp32")
    args = parser.parse_args()

    convert(args.input, args.output, args.dtype)


if __name__ == "__main__":
    main()
