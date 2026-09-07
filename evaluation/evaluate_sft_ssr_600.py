#!/usr/bin/env python3
"""Generate and strictly audit GraphSSR SFT outputs on the 600-row RL validation set."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


SUBGRAPH_RE = re.compile(r"<Subgraph_(\d+)>(.*?)</Subgraph_\1>", re.DOTALL)
CHOSEN_RE = re.compile(
    r"<Chosen_subgraph_reasoning>(.*?)</Chosen_subgraph_reasoning>", re.DOTALL
)
NODE_RE = re.compile(r"node\d+")
EDGE_RE = re.compile(r"<\s*(node\d+)\s*,\s*(node\d+)\s*>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def messages_from_cell(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [dict(message) for message in value]


def read_completed(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                row = json.loads(raw_line)
                completed[int(row["index"])] = row
    return completed


def generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"shard-{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl"
    completed = read_completed(shard_path)

    frame = pd.read_parquet(args.data).reset_index(drop=True)
    targets = [
        index
        for index in range(len(frame))
        if index % args.num_shards == args.shard_index and index not in completed
    ]
    print(
        f"shard={args.shard_index}/{args.num_shards} total={len(frame)} "
        f"completed={len(completed)} remaining={len(targets)}",
        flush=True,
    )
    if not targets:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=False,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=32,
        max_num_batched_tokens=32768,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        repetition_penalty=1.0,
        seed=42,
    )

    with shard_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(targets), args.batch_size):
            batch_indices = targets[start : start + args.batch_size]
            prompts: list[str] = []
            for index in batch_indices:
                messages = messages_from_cell(frame.iloc[index]["prompt"])
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

            outputs = llm.generate(prompts, sampling, use_tqdm=True)
            for index, result in zip(batch_indices, outputs, strict=True):
                candidate = result.outputs[0]
                source = frame.iloc[index]
                record = {
                    "index": index,
                    "dataset": str(source["dataset"]),
                    "task": str(source["task"]),
                    "ground_truth": str(source["ground_truth"]),
                    "prompt_tokens": len(result.prompt_token_ids),
                    "response_tokens": len(candidate.token_ids),
                    "finish_reason": candidate.finish_reason,
                    "stop_reason": candidate.stop_reason,
                    "response": candidate.text,
                }
                handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
                handle.flush()
            print(
                f"shard={args.shard_index} generated={min(start + len(batch_indices), len(targets))}"
                f"/{len(targets)}",
                flush=True,
            )


def post_think(response: str) -> str:
    return response.split("</think>", 1)[-1]


def tag_text(block: str, tag: str) -> str | None:
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", block, re.DOTALL)
    return matches[-1].strip() if matches else None


def node_set(text: str | None) -> set[str]:
    return set(NODE_RE.findall(text or ""))


def edge_set(text: str | None) -> set[tuple[str, str]]:
    return {tuple(sorted(edge)) for edge in EDGE_RE.findall(text or "")}


def prompt_graph(prompt: str) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    graph_section = prompt.split("Step description:", 1)[0]
    central_match = re.search(r"- Central node ID\(s\):\s*(.*?)\n", graph_section)
    central = node_set(central_match.group(1) if central_match else "")
    nodes = set(re.findall(r"^\s*(node\d+):", graph_section, re.MULTILINE))
    edges = edge_set(graph_section)
    return central, nodes, edges


def parsed_subgraph(block: str) -> dict[str, Any]:
    central_text = tag_text(block, "Central_node_ID")
    neighbor_text = tag_text(block, "Neighboring_node_ID")
    connection_text = tag_text(block, "Connection_relationship")
    reason = tag_text(block, "Sampling_reason")
    return {
        "central": node_set(central_text),
        "neighbors": node_set(neighbor_text),
        "edges": edge_set(connection_text),
        "reason": reason or "",
        "required_tags": all(
            value is not None
            for value in (central_text, neighbor_text, connection_text, reason)
        ),
    }


def subgraph_is_real(
    subgraph: dict[str, Any],
    prompt_central: set[str],
    prompt_nodes: set[str],
    prompt_edges: set[tuple[str, str]],
) -> bool:
    selected_nodes = subgraph["central"] | subgraph["neighbors"]
    return all(
        (
            subgraph["required_tags"],
            bool(subgraph["reason"]),
            subgraph["central"] == prompt_central,
            not (subgraph["neighbors"] & prompt_central),
            subgraph["neighbors"] <= (prompt_nodes - prompt_central),
            subgraph["edges"] <= prompt_edges,
            all(set(edge) <= selected_nodes for edge in subgraph["edges"]),
            all(left != right for left, right in subgraph["edges"]),
        )
    )


def chosen_details(block: str) -> dict[str, Any]:
    chosen_text = tag_text(block, "Chosen_subgraph")
    try:
        chosen_index = int(chosen_text) if chosen_text is not None else None
    except ValueError:
        chosen_index = None
    return {
        "index": chosen_index,
        "central": node_set(tag_text(block, "Central_node_ID")),
        "neighbors": node_set(tag_text(block, "Neighboring_node_ID")),
        "edges": edge_set(tag_text(block, "Connection_relationship")),
        "answer": (tag_text(block, "Answer") or "").strip().lower(),
        "reason": tag_text(block, "Chosen_subgraph_reason") or "",
        "brief_reasoning": tag_text(block, "Brief_reasoning") or "",
    }


def repetition_stats(response: str) -> dict[str, Any]:
    text = post_think(response)
    lines = [
        re.sub(r"\s+", " ", line.strip().lower())
        for line in text.splitlines()
        if len(line.strip()) >= 20 and not re.fullmatch(r"</?[^>]+>", line.strip())
    ]
    line_counts = Counter(lines)
    max_line_repeat = max(line_counts.values(), default=0)
    duplicate_line_fraction = (
        sum(count - 1 for count in line_counts.values()) / len(lines) if lines else 0.0
    )

    tokens = re.findall(r"\S+", text.lower())
    ngram_counts = Counter(tuple(tokens[i : i + 32]) for i in range(max(0, len(tokens) - 31)))
    max_32gram_repeat = max(ngram_counts.values(), default=0)
    high_repetition = max_line_repeat >= 8 or max_32gram_repeat >= 5
    return {
        "max_line_repeat": max_line_repeat,
        "duplicate_line_fraction": duplicate_line_fraction,
        "max_32gram_repeat": max_32gram_repeat,
        "high_repetition": high_repetition,
    }


def audit_one(record: dict[str, Any], prompt: str) -> dict[str, Any]:
    response = str(record["response"])
    structured = post_think(response)
    prompt_central, prompt_nodes, prompt_edges = prompt_graph(prompt)

    matches = SUBGRAPH_RE.findall(structured)
    open_ids = re.findall(r"<Subgraph_(\d+)>", structured)
    close_ids = re.findall(r"</Subgraph_(\d+)>", structured)
    chosen_blocks = CHOSEN_RE.findall(structured)
    chosen_open_count = structured.count("<Chosen_subgraph_reasoning>")
    chosen_close_count = structured.count("</Chosen_subgraph_reasoning>")

    exact_ids = open_ids == ["0", "1", "2", "3", "4"] and close_ids == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
    exact_five = exact_ids and len(matches) == 5
    xml_complete = exact_five and chosen_open_count == 1 and chosen_close_count == 1

    parsed = [parsed_subgraph(block) for _, block in matches]
    real_flags = [
        subgraph_is_real(item, prompt_central, prompt_nodes, prompt_edges) for item in parsed
    ]
    structures = {
        (
            tuple(sorted(item["central"])),
            tuple(sorted(item["neighbors"])),
            tuple(sorted(item["edges"])),
        )
        for item in parsed
    }
    has_central_only = any(not item["neighbors"] and not item["edges"] for item in parsed)
    distinct_five = len(structures) == 5
    strict_real_subgraphs = (
        exact_five
        and len(real_flags) == 5
        and all(real_flags)
        and distinct_five
        and has_central_only
    )

    chosen = chosen_details(chosen_blocks[-1]) if chosen_blocks else chosen_details("")
    chosen_index = chosen["index"]
    chosen_consistent = False
    if isinstance(chosen_index, int) and 0 <= chosen_index < 5 and len(parsed) == 5:
        selected = parsed[chosen_index]
        chosen_consistent = all(
            (
                chosen["central"] == selected["central"],
                chosen["neighbors"] == selected["neighbors"],
                chosen["edges"] == selected["edges"],
                bool(chosen["reason"]),
                bool(chosen["brief_reasoning"]),
            )
        )

    truth = str(record["ground_truth"]).strip().lower()
    answer_correct = bool(chosen["answer"]) and chosen["answer"] == truth
    length_truncated = record.get("finish_reason") == "length"
    stopped = record.get("finish_reason") == "stop"
    repetition = repetition_stats(response)
    real_pred = (
        xml_complete
        and strict_real_subgraphs
        and chosen_consistent
        and answer_correct
        and stopped
        and not repetition["high_repetition"]
    )

    return {
        "index": int(record["index"]),
        "dataset": record["dataset"],
        "task": record["task"],
        "answer": chosen["answer"],
        "ground_truth": record["ground_truth"],
        "answer_correct": answer_correct,
        "xml_complete": xml_complete,
        "exact_five_subgraphs": exact_five,
        "strict_real_subgraphs": strict_real_subgraphs,
        "distinct_five_subgraphs": distinct_five,
        "has_central_only_subgraph": has_central_only,
        "chosen_consistent": chosen_consistent,
        "stopped_on_eos": stopped,
        "length_truncated": length_truncated,
        "real_pred": real_pred,
        **repetition,
        "prompt_tokens": int(record["prompt_tokens"]),
        "response_tokens": int(record["response_tokens"]),
        "finish_reason": record.get("finish_reason"),
        "stop_reason": record.get("stop_reason"),
    }


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(
    rows: list[dict[str, Any]], max_tokens: int = 8192, max_model_len: int = 12288
) -> dict[str, Any]:
    boolean_fields = [
        "answer_correct",
        "xml_complete",
        "exact_five_subgraphs",
        "strict_real_subgraphs",
        "distinct_five_subgraphs",
        "has_central_only_subgraph",
        "chosen_consistent",
        "stopped_on_eos",
        "length_truncated",
        "high_repetition",
        "real_pred",
        "legacy_pred",
        "legacy_real_pred",
    ]

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        response_lengths = [int(row["response_tokens"]) for row in group]
        metrics: dict[str, Any] = {"count": len(group)}
        for field in boolean_fields:
            count = sum(bool(row.get(field, False)) for row in group)
            metrics[field] = {"count": count, "rate": count / len(group) if group else 0.0}
        if group and all("legacy_score" in row for row in group):
            legacy_scores = [float(row["legacy_score"]) for row in group]
            metrics["legacy_reward"] = {
                "mean": mean(legacy_scores),
                "distribution": {
                    str(value): legacy_scores.count(value) for value in sorted(set(legacy_scores))
                },
            }
        metrics["finish_reason"] = dict(Counter(str(row["finish_reason"]) for row in group))
        metrics["response_tokens"] = {
            "mean": mean(response_lengths) if response_lengths else 0.0,
            "p50": percentile(response_lengths, 0.50),
            "p95": percentile(response_lengths, 0.95),
            "max": max(response_lengths, default=0),
        }
        return metrics

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(f'{row["dataset"]}+{row["task"]}', []).append(row)
    return {
        "definitions": {
            "generation": (
                f"greedy, max_response_tokens={max_tokens}, "
                f"max_model_len={max_model_len}"
            ),
            "high_repetition": "a normalized long line repeats >=8 times or a 32-token n-gram repeats >=5 times",
            "real_pred": "complete XML + five distinct real subgraphs + consistent choice + correct answer + EOS + no high repetition",
            "legacy_reward": "the current reinforcement_learning/verl/utils/reward_score/subgraph_size.py result, retained only for direct comparison",
        },
        "global": summarize_group(rows),
        "by_task": {name: summarize_group(group) for name, group in sorted(groups.items())},
    }


def analyze(args: argparse.Namespace) -> None:
    from reinforcement_learning.verl.utils.reward_score.subgraph_size import compute_score

    output_dir = Path(args.output_dir)
    frame = pd.read_parquet(args.data).reset_index(drop=True)
    generated: dict[int, dict[str, Any]] = {}
    for path in sorted(output_dir.glob("shard-*-of-*.jsonl")):
        generated.update(read_completed(path))

    if len(generated) != len(frame):
        missing = sorted(set(range(len(frame))) - set(generated))
        raise RuntimeError(
            f"expected {len(frame)} generated rows, found {len(generated)}; missing={missing[:20]}"
        )

    audited: list[dict[str, Any]] = []
    for index in range(len(frame)):
        prompt = messages_from_cell(frame.iloc[index]["prompt"])[-1]["content"]
        audit = audit_one(generated[index], prompt)
        legacy = compute_score(
            generated[index]["response"],
            generated[index]["ground_truth"],
            "graph_rl_data",
            {"prompt_str": prompt, "in_second_stage": False},
        )
        audit.update(
            {
                "legacy_score": float(legacy["score"]),
                "legacy_pred": bool(legacy["pred"]),
                "legacy_real_pred": bool(legacy["real_pred"]),
            }
        )
        audited.append(audit)

    audit_path = output_dir / "audit_rows.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for row in audited:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(audited, args.max_tokens, args.max_model_len)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if not args.analyze_only:
        generate(args)
    if args.analyze_only or args.num_shards == 1:
        analyze(args)


if __name__ == "__main__":
    main()
