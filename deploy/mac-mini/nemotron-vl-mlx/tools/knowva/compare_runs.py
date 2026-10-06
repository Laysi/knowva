"""Compares bench_sets.py runs with a reference run: logit differences, the
top-6 sets Knowva would keep, their order, and rerank time per question.

    uv run tools/knowva/compare_runs.py REF.json RUN.json [RUN.json ...]
"""

import json
import statistics
import sys

import numpy as np

TOP_N = 6
ref_path, run_paths = sys.argv[1], sys.argv[2:]
ref = json.load(open(ref_path))
ref_by_id = {r["id"]: r for r in ref["results"]}


def describe(run):
    return f"{run['backend']} {run['device']} {run['dtype']} {run.get('batching', 'padded')}"


def timing(run):
    seconds = [r["seconds"] for r in run["results"]]
    return statistics.mean(seconds), statistics.median(seconds), max(seconds)


mean, median, slowest = timing(ref)
print(f"reference {describe(ref)}: mean {mean:.2f}s median {median:.2f}s max {slowest:.2f}s, "
      f"peak device {ref['peak_device_bytes'] / 2**30:.2f} GiB, rss {ref['max_rss_bytes'] / 2**30:.2f} GiB")
for path in run_paths:
    run = json.load(open(path))
    worst = 0.0
    set_mismatch = []
    order_diffs = []
    for r in run["results"]:
        a = np.array(ref_by_id[r["id"]]["logits"])
        b = np.array(r["logits"])
        worst = max(worst, float(np.abs(a - b).max()))
        top_a = list(np.argsort(-a, kind="stable")[:TOP_N])
        top_b = list(np.argsort(-b, kind="stable")[:TOP_N])
        if set(top_a) != set(top_b):
            gap = float(np.sort(a)[::-1][TOP_N - 1] - np.sort(a)[::-1][TOP_N])
            set_mismatch.append(f"{r['id']} ref={[int(i) for i in top_a]} run={[int(i) for i in top_b]} "
                                f"(ref gap at #6/#7 {gap:.4f})")
        elif top_a != top_b:
            order_diffs.append(f"{r['id']} ref={[int(i) for i in top_a]} run={[int(i) for i in top_b]}")
    mean_run, median_run, slowest_run = timing(run)
    print(f"\n{describe(run)} ({path})")
    print(f"  max|logit diff| {worst:.5f}; top-{TOP_N} set differs in {len(set_mismatch)}/{len(run['results'])}, "
          f"order differs in {len(order_diffs)}")
    for line in set_mismatch:
        print(f"    set: {line}")
    for line in order_diffs:
        print(f"    order: {line}")
    print(f"  time mean {mean_run:.2f}s median {median_run:.2f}s max {slowest_run:.2f}s; "
          f"speed-up vs reference mean {mean / mean_run:.2f}x median {median / median_run:.2f}x")
    print(f"  peak device {run['peak_device_bytes'] / 2**30:.2f} GiB, rss {run['max_rss_bytes'] / 2**30:.2f} GiB")
