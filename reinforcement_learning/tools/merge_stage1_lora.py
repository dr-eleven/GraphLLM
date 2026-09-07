#!/usr/bin/env python3
"""Merge a verl HF export containing PEFT-wrapped weights into a plain model."""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--lora-export", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    return p.parse_args()


def main():
    args = parse_args()
    export = Path(args.lora_export)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    index = export / "model.safetensors"
    if not index.is_file():
        raise FileNotFoundError(f"missing checkpoint: {index}")

    print(f"Loading base model: {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    state = model.state_dict()
    base_updates = {}
    lora_a = {}
    lora_b = {}

    with safe_open(str(index), framework="pt", device="cpu") as reader:
        for key in reader.keys():
            tensor = reader.get_tensor(key)
            if not key.startswith("base_model.model."):
                continue
            key = key[len("base_model.model.") :]
            if ".lora_A.default.weight" in key:
                lora_a[key.replace(".lora_A.default.weight", ".weight")] = tensor
            elif ".lora_B.default.weight" in key:
                lora_b[key.replace(".lora_B.default.weight", ".weight")] = tensor
            elif ".base_layer." in key:
                base_updates[key.replace(".base_layer.", ".")] = tensor
            else:
                base_updates[key] = tensor

    missing = sorted(set(base_updates) - set(state))
    if missing:
        raise RuntimeError(f"checkpoint keys do not match base model; first missing: {missing[:5]}")
    for key, tensor in base_updates.items():
        state[key] = tensor.to(dtype=state[key].dtype)

    scale = args.lora_alpha / 32.0
    merged = 0
    for key, a in lora_a.items():
        b = lora_b.get(key)
        if b is None:
            raise RuntimeError(f"missing LoRA B tensor for {key}")
        if key not in state:
            raise RuntimeError(f"LoRA target is absent from base model: {key}")
        delta = (b.float() @ a.float()).to(dtype=state[key].dtype) * scale
        state[key] = state[key] + delta
        merged += 1

    model.load_state_dict(state, strict=True)
    model.config.use_cache = True
    print(f"Merged {merged} LoRA tensors; saving: {output}", flush=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")

    # Reuse the export tokenizer/template, which is the one used by Stage 1.
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json"):
        src = export / name
        dst = output / name
        if src.is_file() and not dst.exists():
            dst.write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()
