#!/usr/bin/env python3
"""Benchmark analysis: compare Ray vs Monarch RLVR pipeline stage timing.

Usage:
    python benchmark_ray_vs_monarch.py /tmp/rlvr_ray_50.log /tmp/rlvr_monarch_50_final.log
"""
import json
import sys
from collections import defaultdict


STAGE_KEYS = [
    # Top-level wall-clock stages
    ("time/step_total", "step_total"),
    ("time/step_generate", "  generate"),
    ("time/step_model_update", "  model_update"),
    ("time/ref_log_probs_values", "  ref_log_probs"),
    ("time/old_log_probs", "  old_log_probs"),
    ("time/step_train", "  train_step"),
    # Worker-level actual GPU compute
    ("time/actor_train/train_step/total", "  [gpu] train compute"),
    ("time/actor_train/model_update/total", "  [gpu] model_update compute"),
    ("time/actor_train/compute_log_probs/total", "  [gpu] old_logprobs compute"),
    ("time/reference/compute_log_probs/total", "  [gpu] ref_logprobs compute"),
]

THROUGHPUT_KEYS = [
    ("system/actor_infer/tps", "infer tps"),
    ("system/actor_train/tps", "train tps"),
]


def extract_steps(logfile):
    """Parse JSON metrics from each step in the log file."""
    steps = []
    with open(logfile) as f:
        for line in f:
            if "time/step_total" not in line:
                continue
            try:
                idx = line.index("{")
                data = json.loads(line[idx:])
                steps.append(data)
            except (ValueError, json.JSONDecodeError):
                pass
    return steps


def avg(values):
    return sum(values) / len(values) if values else 0.0


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ray_log> <monarch_log>")
        sys.exit(1)

    ray_log, monarch_log = sys.argv[1], sys.argv[2]
    ray_steps = extract_steps(ray_log)
    mon_steps = extract_steps(monarch_log)

    # Skip step 0 (cold start)
    ray_steps = [s for s in ray_steps if s.get("time/step_total", 0) > 0]
    mon_steps = [s for s in mon_steps if s.get("time/step_total", 0) > 0]

    print("=" * 90)
    print("  RLVR Pipeline Stage Timing: Ray vs Monarch")
    print(f"  Ray steps: {len(ray_steps)}, Monarch steps: {len(mon_steps)}")
    print("=" * 90)
    print(f"  {'Stage':<32s} {'Ray (s)':>9s} {'Monarch (s)':>12s} {'Ratio':>8s} {'Ray %':>7s} {'Mon %':>7s}")
    print("-" * 90)

    ray_total = avg([s.get("time/step_total", 0) for s in ray_steps])
    mon_total = avg([s.get("time/step_total", 0) for s in mon_steps])

    ray_accounted = 0.0
    mon_accounted = 0.0

    for key, label in STAGE_KEYS:
        r = avg([s.get(key, 0) for s in ray_steps])
        m = avg([s.get(key, 0) for s in mon_steps])
        ratio = m / r if r > 0 else float("inf")
        r_pct = (r / ray_total * 100) if ray_total > 0 else 0
        m_pct = (m / mon_total * 100) if mon_total > 0 else 0

        if key != "time/step_total" and not key.startswith("time/actor_") and not key.startswith("time/reference"):
            ray_accounted += r
            mon_accounted += m

        print(f"  {label:<32s} {r:>9.2f} {m:>12.2f} {ratio:>7.2f}x {r_pct:>6.1f}% {m_pct:>6.1f}%")

    # Unaccounted overhead
    ray_overhead = ray_total - ray_accounted
    mon_overhead = mon_total - mon_accounted
    oh_ratio = mon_overhead / ray_overhead if ray_overhead > 0 else float("inf")
    print(f"  {'  overhead (unaccounted)':<32s} {ray_overhead:>9.2f} {mon_overhead:>12.2f} {oh_ratio:>7.2f}x "
          f"{ray_overhead/ray_total*100:>6.1f}% {mon_overhead/mon_total*100:>6.1f}%")

    print("-" * 90)

    # RPC overhead = wall time - GPU compute time
    print()
    print("  RPC / Serialization Overhead (wall_time - gpu_compute_time):")
    print(f"  {'Stage':<32s} {'Ray (s)':>9s} {'Monarch (s)':>12s} {'Ratio':>8s}")
    print("-" * 90)

    pairs = [
        ("time/step_train", "time/actor_train/train_step/total", "train"),
        ("time/step_model_update", "time/actor_train/model_update/total", "model_update"),
        ("time/old_log_probs", "time/actor_train/compute_log_probs/total", "old_log_probs"),
        ("time/ref_log_probs_values", "time/reference/compute_log_probs/total", "ref_log_probs"),
    ]
    for wall_key, compute_key, label in pairs:
        r_wall = avg([s.get(wall_key, 0) for s in ray_steps])
        r_compute = avg([s.get(compute_key, 0) for s in ray_steps])
        m_wall = avg([s.get(wall_key, 0) for s in mon_steps])
        m_compute = avg([s.get(compute_key, 0) for s in mon_steps])
        r_oh = r_wall - r_compute
        m_oh = m_wall - m_compute
        ratio = m_oh / r_oh if r_oh > 0.01 else float("inf")
        print(f"  {label:<32s} {r_oh:>9.2f} {m_oh:>12.2f} {ratio:>7.2f}x")

    # Throughput
    print()
    print("-" * 90)
    print(f"  {'Throughput':<32s} {'Ray':>9s} {'Monarch':>12s} {'Ratio':>8s}")
    print("-" * 90)
    for key, label in THROUGHPUT_KEYS:
        r = avg([s.get(key, 0) for s in ray_steps])
        m = avg([s.get(key, 0) for s in mon_steps])
        ratio = r / m if m > 0 else float("inf")
        print(f"  {label:<32s} {r:>8.1f} {m:>11.1f} {ratio:>7.2f}x faster")

    print("=" * 90)


if __name__ == "__main__":
    main()
