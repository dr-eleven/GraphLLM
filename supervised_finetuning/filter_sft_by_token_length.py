#!/usr/bin/env python3
"""Filter SFT examples whose complete formatted sequence exceeds cutoff_len.

LLaMA-Factory normally truncates an overlength SFT response and still trains on
its prefix.  GraphSSR responses are structured reasoning traces, so losing the
response suffix (including the EOS token) creates an invalid training target.
This script uses the same tokenizer and chat template as LLaMA-Factory and
writes only examples whose full prompt + response + EOS fit in cutoff_len.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="LLaMA-Factory training YAML.")
    parser.add_argument("--input", required=True, type=Path, help="Original GraphSSR SFT JSON file.")
    parser.add_argument("--output", required=True, type=Path, help="Filtered SFT JSON file.")
    parser.add_argument("--force", action="store_true", help="Ignore a valid cached filtered file.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a YAML mapping: {path}")

    required = ("model_name_or_path", "template", "cutoff_len")
    missing = [key for key in required if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing required setting(s) in {path}: {', '.join(missing)}")

    cutoff_len = config["cutoff_len"]
    if not isinstance(cutoff_len, int) or isinstance(cutoff_len, bool) or cutoff_len <= 0:
        raise ValueError(f"cutoff_len must be a positive integer, got {cutoff_len!r}")

    return config


def build_cache_key(config: dict[str, Any], input_path: Path) -> dict[str, Any]:
    source_stat = input_path.stat()
    return {
        "version": 1,
        "source": str(input_path.resolve()),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "model_name_or_path": str(config["model_name_or_path"]),
        "template": str(config["template"]),
        "cutoff_len": config["cutoff_len"],
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
        "use_fast_tokenizer": bool(config.get("use_fast_tokenizer", True)),
        "split_special_tokens": bool(config.get("split_special_tokens", False)),
        "default_system": config.get("default_system"),
        "enable_thinking": config.get("enable_thinking", True),
    }


def load_cached_metadata(metadata_path: Path) -> dict[str, Any] | None:
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    return metadata if isinstance(metadata, dict) else None


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary_name, path)
        os.chmod(path, 0o644)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def create_tokenizer_and_template(config: dict[str, Any]):
    # Keep heavyweight Transformers/LLaMA-Factory imports out of the cache-hit
    # path. This also avoids initializing PyTorch when no filtering is needed.
    from transformers import AutoTokenizer

    from llamafactory.data.template import TEMPLATES

    use_fast = bool(config.get("use_fast_tokenizer", True))
    tokenizer_kwargs = {
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
        "use_fast": use_fast,
        "padding_side": "right",
    }
    if not use_fast:
        tokenizer_kwargs["split_special_tokens"] = bool(config.get("split_special_tokens", False))

    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], **tokenizer_kwargs)
    except ValueError:
        tokenizer_kwargs["use_fast"] = not use_fast
        tokenizer_kwargs.pop("split_special_tokens", None)
        tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], **tokenizer_kwargs)

    template_name = str(config["template"])
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown LLaMA-Factory template: {template_name}")

    # Templates are registered as mutable singletons. Work on a copy so this
    # standalone preprocessing step cannot modify global template state.
    template = deepcopy(TEMPLATES[template_name])
    if config.get("default_system") is not None:
        template.default_system = str(config["default_system"])
    template.enable_thinking = config.get("enable_thinking", True)
    template.fix_special_tokens(tokenizer)
    template.fix_jinja_template(tokenizer)
    return tokenizer, template


def encoded_length(tokenizer, template, example: dict[str, Any], index: int) -> int:
    prompt = example.get("prompt")
    response = example.get("response")
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise ValueError(
            f"Example {index} must contain string fields 'prompt' and 'response'; "
            f"got prompt={type(prompt).__name__}, response={type(response).__name__}"
        )

    prompt_ids, response_ids = template.encode_oneturn(
        tokenizer,
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    )
    return len(prompt_ids) + len(response_ids)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")

    if input_path == output_path:
        raise ValueError("--input and --output must differ so the original dataset is preserved.")
    if not input_path.is_file():
        raise FileNotFoundError(f"SFT input data not found: {input_path}")

    config = load_config(config_path)
    cache_key = build_cache_key(config, input_path)
    cached_metadata = load_cached_metadata(metadata_path)
    if not args.force and output_path.is_file() and cached_metadata is not None:
        if cached_metadata.get("cache_key") == cache_key:
            print(
                "SFT length filter cache hit: "
                f"kept={cached_metadata['kept']}, dropped={cached_metadata['dropped']}, "
                f"cutoff_len={cache_key['cutoff_len']}, output={output_path}"
            )
            return

    with input_path.open("r", encoding="utf-8") as handle:
        examples = json.load(handle)
    if not isinstance(examples, list):
        raise ValueError(f"Expected a JSON array in {input_path}, got {type(examples).__name__}")

    tokenizer, template = create_tokenizer_and_template(config)
    cutoff_len = config["cutoff_len"]
    kept_examples: list[dict[str, Any]] = []
    dropped = 0
    max_kept_length = 0
    min_dropped_length: int | None = None
    max_length = 0

    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"Example {index} must be a JSON object, got {type(example).__name__}")

        sequence_length = encoded_length(tokenizer, template, example, index)
        max_length = max(max_length, sequence_length)
        if sequence_length <= cutoff_len:
            kept_examples.append(example)
            max_kept_length = max(max_kept_length, sequence_length)
        else:
            dropped += 1
            if min_dropped_length is None:
                min_dropped_length = sequence_length
            else:
                min_dropped_length = min(min_dropped_length, sequence_length)

    atomic_write_json(output_path, kept_examples)
    metadata = {
        "cache_key": cache_key,
        "total": len(examples),
        "kept": len(kept_examples),
        "dropped": dropped,
        "max_kept_length": max_kept_length,
        "min_dropped_length": min_dropped_length,
        "max_original_length": max_length,
    }
    atomic_write_json(metadata_path, metadata)

    print(
        "SFT length filter complete: "
        f"total={len(examples)}, kept={len(kept_examples)}, dropped={dropped}, "
        f"cutoff_len={cutoff_len}, max_kept_length={max_kept_length}, "
        f"min_dropped_length={min_dropped_length}, output={output_path}"
    )


if __name__ == "__main__":
    main()
