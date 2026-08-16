"""Benchmark end-to-end LLM generation latency for this backend.

This script uses the same LLMService class as the API server to measure
realistic request latency in milliseconds and estimated output tokens/sec.
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch

from llm_service import LLMService


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "Respond clearly with no emojis."
)

DEFAULT_MESSAGE = "Explain in three short points how emotion-aware AI can help customer support."


def percentile(values, pct):
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def sync_cuda_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_once(service, user_id, system_prompt, message):
    service.clear_conversation(user_id)
    sync_cuda_if_needed()
    t0 = time.perf_counter()
    reply, _ = service.chat(user_id=user_id, message=message, system_prompt=system_prompt)
    sync_cuda_if_needed()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    out_tokens = 0
    if service.tokenizer is not None:
        out_tokens = len(service.tokenizer.encode(reply, add_special_tokens=False))

    return dt_ms, out_tokens, reply


def main():
    parser = argparse.ArgumentParser(description="Evaluate backend LLM latency.")
    parser.add_argument("--runs", type=int, default=10, help="Measured runs")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="User message")
    parser.add_argument("--model-path", default=None, help="Override MODEL_PATH")
    parser.add_argument("--adapter-path", default=None, help="Override LORA_ADAPTER_PATH")
    parser.add_argument("--output", default="llm_latency_report.json", help="Output JSON path")
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    service = LLMService(model_path=args.model_path, adapter_path=args.adapter_path)
    if not service.is_loaded():
        raise RuntimeError("Model failed to load. Check model path/dependencies before benchmarking.")

    benchmark_user = "latency-benchmark"

    print(f"Warmup runs: {args.warmup}")
    for i in range(args.warmup):
        ms, toks, _ = run_once(service, benchmark_user, args.system_prompt, args.message)
        print(f"  warmup {i + 1}/{args.warmup}: {ms:.2f} ms, out_tokens={toks}")

    latencies_ms = []
    output_tokens = []
    sample_reply = ""

    print(f"Measured runs: {args.runs}")
    for i in range(args.runs):
        ms, toks, reply = run_once(service, benchmark_user, args.system_prompt, args.message)
        latencies_ms.append(ms)
        output_tokens.append(toks)
        if not sample_reply:
            sample_reply = reply
        print(f"  run {i + 1}/{args.runs}: {ms:.2f} ms, out_tokens={toks}")

    total_ms = sum(latencies_ms)
    total_tokens = sum(output_tokens)
    avg_ms = total_ms / len(latencies_ms)
    p50_ms = percentile(latencies_ms, 50)
    p95_ms = percentile(latencies_ms, 95)
    p99_ms = percentile(latencies_ms, 99)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    tokens_per_sec = (total_tokens / total_ms) * 1000.0 if total_ms > 0 else 0.0

    report = {
        "timestamp": datetime.now().isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_path": service.model_path,
        "adapter_path": service.adapter_path,
        "runs": args.runs,
        "warmup": args.warmup,
        "message": args.message,
        "max_new_tokens": service.max_new_tokens,
        "temperature": service.temperature,
        "top_p": service.top_p,
        "latency_ms": {
            "min": min_ms,
            "p50": p50_ms,
            "p95": p95_ms,
            "p99": p99_ms,
            "max": max_ms,
            "mean": avg_ms,
        },
        "output_tokens": {
            "total": total_tokens,
            "mean_per_run": (total_tokens / len(output_tokens)) if output_tokens else 0.0,
        },
        "throughput_tokens_per_sec": tokens_per_sec,
        "sample_reply": sample_reply,
    }

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.getcwd(), out_path)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== LLM Latency Summary ===")
    print(f"Device: {report['device']}")
    print(f"Mean latency: {avg_ms:.2f} ms")
    print(f"P50 latency: {p50_ms:.2f} ms")
    print(f"P95 latency: {p95_ms:.2f} ms")
    print(f"P99 latency: {p99_ms:.2f} ms")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
