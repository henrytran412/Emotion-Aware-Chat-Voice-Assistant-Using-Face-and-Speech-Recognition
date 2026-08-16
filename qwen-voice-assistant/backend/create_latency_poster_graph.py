"""Create a poster-ready latency figure from llm_latency_report.json."""

import argparse
import json
import os

import matplotlib.pyplot as plt


def ms_to_s(value_ms):
    return float(value_ms) / 1000.0


def fmt_seconds(value_ms):
    return f"{ms_to_s(value_ms):.2f}s"


def build_poster(report, output_path):
    latency = report["latency_ms"]
    labels = ["Min", "P50", "Mean", "P95", "P99", "Max"]
    keys = ["min", "p50", "mean", "p95", "p99", "max"]
    values_ms = [float(latency[k]) for k in keys]
    values_s = [ms_to_s(v) for v in values_ms]

    fig = plt.figure(figsize=(14, 8), dpi=220)
    fig.patch.set_facecolor("#f7f7f5")

    ax = fig.add_axes([0.08, 0.18, 0.64, 0.70])
    ax.set_facecolor("#f7f7f5")

    colors = ["#8ecae6", "#219ebc", "#2a9d8f", "#f4a261", "#e76f51", "#c1121f"]
    bars = ax.bar(labels, values_s, color=colors, edgecolor="#1f2933", linewidth=1.1)

    for idx, bar in enumerate(bars):
        v = values_ms[idx]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            fmt_seconds(v),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#1b1b1b",
        )

    ax.set_ylabel("Latency (seconds)", fontsize=12, color="#1f2933")
    ax.set_title("LLM Inference Latency Summary", fontsize=20, fontweight="bold", color="#102a43", pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)

    throughput = float(report.get("throughput_tokens_per_sec", 0.0))
    runs = int(report.get("runs", 0))
    device = report.get("device", "unknown").upper()
    mean_tokens = float(report.get("output_tokens", {}).get("mean_per_run", 0.0))
    model = report.get("model_path", "unknown")

    side_text = (
        f"Device: {device}\n"
        f"Runs: {runs}\n"
        f"Avg output tokens/run: {mean_tokens:.1f}\n"
        f"Throughput: {throughput:.2f} tokens/sec\n"
        f"Model: {model}"
    )

    fig.text(
        0.76,
        0.67,
        "Benchmark Details",
        fontsize=13,
        fontweight="bold",
        color="#102a43",
    )
    fig.text(
        0.76,
        0.44,
        side_text,
        fontsize=10.5,
        color="#1f2933",
        linespacing=1.5,
    )

    fig.text(
        0.08,
        0.07,
        "End-to-end generation latency from backend LLMService (warm cache runs).",
        fontsize=10,
        color="#334e68",
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create poster-ready LLM latency graph")
    parser.add_argument("--input", default="llm_latency_report.json", help="Latency JSON report path")
    parser.add_argument("--output", default="llm_latency_poster.png", help="Output poster PNG path")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.getcwd(), input_path)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input report not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    build_poster(report, output_path)
    print(f"Saved latency poster graph: {output_path}")


if __name__ == "__main__":
    main()
