from pathlib import Path
import csv

import numpy as np


WARMUP_RUNS = 5
EXPECTED_RUNS = 100

PROFILES = {
    "Local QAIRT FP": Path(
        "data/qnn_profiles/resnet18_fp32/profile.csv"
    ),
    "AI Hub FP": Path(
        "data/qnn_profiles/resnet18_aihub_fp/profile.csv"
    ),
    "W8A16": Path(
        "data/qnn_profiles/resnet18_w8a16/profile.csv"
    ),
    "W8A8": Path(
        "data/qnn_profiles/resnet18_w8a8/profile.csv"
    ),
}

# Use the AI Hub floating-point DLC as the reference for quantization
# comparisons. W8A16 and W8A8 were also generated through AI Hub, making
# this a more apples-to-apples comparison than the locally generated
# QAIRT model library.
QUANTIZATION_BASELINE = "AI Hub FP"


def load_profile(path: Path):
    """Extract NetRun and accelerator execution times from a QNN profile CSV."""
    netrun = []
    accelerator = []

    with path.open(newline="") as f:
        for line in f:
            if line.startswith("Msg Timestamp,"):
                header = line
                break
        else:
            raise RuntimeError(
                f"Could not find profiling CSV header in {path}"
            )

        reader = csv.DictReader(
            f,
            fieldnames=next(csv.reader([header])),
        )

        for row in reader:
            message = row[" Message"].strip()
            source = row[" Timing Source"].strip()
            event = row[" Event Identifier"].strip()
            time_us = float(row[" Time"].strip())

            if message != "EXECUTE":
                continue

            if source == "NETRUN" and event.startswith("Graph "):
                netrun.append(time_us / 1000.0)

            if (
                source == "BACKEND"
                and event
                == "Accelerator (execute excluding wait) time"
            ):
                accelerator.append(time_us / 1000.0)

    return (
        np.asarray(netrun, dtype=np.float64),
        np.asarray(accelerator, dtype=np.float64),
    )


def calculate_stats(values):
    """Return latency statistics for a set of measurements."""
    return {
        "runs": len(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "p90": np.percentile(values, 90),
        "p95": np.percentile(values, 95),
        "p99": np.percentile(values, 99),
        "min": np.min(values),
        "max": np.max(values),
        "std": np.std(values),
    }


def print_stats(name, stats):
    """Print latency statistics in milliseconds."""
    print(f"\n{name}")
    print("-" * 60)
    print(f"Runs:    {stats['runs']}")
    print(f"Mean:    {stats['mean']:.4f} ms")
    print(f"Median:  {stats['median']:.4f} ms")
    print(f"P90:     {stats['p90']:.4f} ms")
    print(f"P95:     {stats['p95']:.4f} ms")
    print(f"P99:     {stats['p99']:.4f} ms")
    print(f"Min:     {stats['min']:.4f} ms")
    print(f"Max:     {stats['max']:.4f} ms")
    print(f"Std dev: {stats['std']:.4f} ms")


def improvement(reference, candidate):
    """Return percentage latency reduction relative to reference."""
    return (reference - candidate) / reference * 100.0


def speedup(reference, candidate):
    """Return multiplicative speedup relative to reference."""
    return reference / candidate


def main():
    results = {}

    print("QNN RESNET-18 PHYSICAL DEVICE BENCHMARK")
    print("=" * 60)
    print(f"Warm-up runs excluded: {WARMUP_RUNS}")
    print(f"Expected runs/profile: {EXPECTED_RUNS}")

    for model_name, path in PROFILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing profile for {model_name}: {path}"
            )

        netrun, accelerator = load_profile(path)

        if len(netrun) != EXPECTED_RUNS:
            raise RuntimeError(
                f"{model_name}: expected {EXPECTED_RUNS} NetRun "
                f"executions, found {len(netrun)}"
            )

        if len(accelerator) != EXPECTED_RUNS:
            raise RuntimeError(
                f"{model_name}: expected {EXPECTED_RUNS} accelerator "
                f"executions, found {len(accelerator)}"
            )

        netrun_steady = netrun[WARMUP_RUNS:]
        accelerator_steady = accelerator[WARMUP_RUNS:]

        results[model_name] = {
            "netrun": calculate_stats(netrun_steady),
            "accelerator": calculate_stats(accelerator_steady),
        }

        print_stats(
            f"{model_name} — NetRun steady state",
            results[model_name]["netrun"],
        )

        print_stats(
            f"{model_name} — Accelerator steady state",
            results[model_name]["accelerator"],
        )

    # ------------------------------------------------------------------
    # Complete physical-device results
    # ------------------------------------------------------------------

    print("\n\nPHYSICAL GALAXY S24 — RESNET-18")
    print("=" * 88)

    print(
        f"{'Model':<20}"
        f"{'NetRun Mean':>14}"
        f"{'NetRun P95':>14}"
        f"{'Accel Mean':>14}"
        f"{'Accel P95':>14}"
    )

    print("-" * 88)

    for model_name in PROFILES:
        net = results[model_name]["netrun"]
        acc = results[model_name]["accelerator"]

        print(
            f"{model_name:<20}"
            f"{net['mean']:>12.3f} ms"
            f"{net['p95']:>12.3f} ms"
            f"{acc['mean']:>12.3f} ms"
            f"{acc['p95']:>12.3f} ms"
        )

    # ------------------------------------------------------------------
    # Apples-to-apples AI Hub quantization comparison
    # ------------------------------------------------------------------

    baseline_net = results[QUANTIZATION_BASELINE]["netrun"]["mean"]
    baseline_acc = results[QUANTIZATION_BASELINE]["accelerator"]["mean"]

    print(
        f"\n\nQUANTIZATION PERFORMANCE VS {QUANTIZATION_BASELINE.upper()}"
    )
    print("=" * 76)

    for model_name in ("W8A16", "W8A8"):
        net = results[model_name]["netrun"]["mean"]
        acc = results[model_name]["accelerator"]["mean"]

        print(f"\n{model_name}")

        print(
            f"  NetRun:      "
            f"{improvement(baseline_net, net):.2f}% lower latency "
            f"({speedup(baseline_net, net):.2f}x speedup)"
        )

        print(
            f"  Accelerator: "
            f"{improvement(baseline_acc, acc):.2f}% lower latency "
            f"({speedup(baseline_acc, acc):.2f}x speedup)"
        )

    # ------------------------------------------------------------------
    # W8A8 vs W8A16
    # ------------------------------------------------------------------

    w8a16_net = results["W8A16"]["netrun"]["mean"]
    w8a16_acc = results["W8A16"]["accelerator"]["mean"]

    w8a8_net = results["W8A8"]["netrun"]["mean"]
    w8a8_acc = results["W8A8"]["accelerator"]["mean"]

    print("\n\nW8A8 VS W8A16")
    print("=" * 76)

    print(
        f"NetRun:      "
        f"{improvement(w8a16_net, w8a8_net):.2f}% lower latency "
        f"({speedup(w8a16_net, w8a8_net):.2f}x speedup)"
    )

    print(
        f"Accelerator: "
        f"{improvement(w8a16_acc, w8a8_acc):.2f}% lower latency "
        f"({speedup(w8a16_acc, w8a8_acc):.2f}x speedup)"
    )

    # ------------------------------------------------------------------
    # Local QAIRT vs AI Hub floating deployment
    # ------------------------------------------------------------------

    local_net = results["Local QAIRT FP"]["netrun"]["mean"]
    local_acc = results["Local QAIRT FP"]["accelerator"]["mean"]

    aihub_net = results["AI Hub FP"]["netrun"]["mean"]
    aihub_acc = results["AI Hub FP"]["accelerator"]["mean"]

    print("\n\nFLOATING DEPLOYMENT PATH COMPARISON")
    print("=" * 76)

    print(
        f"Local QAIRT FP NetRun:      {local_net:.3f} ms\n"
        f"AI Hub FP NetRun:           {aihub_net:.3f} ms\n"
        f"Difference:                 "
        f"{improvement(aihub_net, local_net):.2f}% lower latency "
        f"for Local QAIRT FP"
    )

    print()

    print(
        f"Local QAIRT FP Accelerator: {local_acc:.3f} ms\n"
        f"AI Hub FP Accelerator:      {aihub_acc:.3f} ms\n"
        f"Difference:                 "
        f"{improvement(aihub_acc, local_acc):.2f}% lower latency "
        f"for Local QAIRT FP"
    )

    print(
        "\nNote: Local QAIRT FP and AI Hub FP were produced through "
        "different compilation/deployment paths. This comparison does "
        "not isolate packaging format as the cause of the performance "
        "difference."
    )


if __name__ == "__main__":
    main()