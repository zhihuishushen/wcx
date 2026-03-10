import os
import json
import argparse
from collections import Counter

def analyze_samples(samples_path: str):
    """
    Print simple statistics:
      - per-type positive counts (if present)
      - number of samples with upper_graph candidate
      - distribution of upper_graph_confidence
    """
    samples = []
    if os.path.isdir(samples_path):
        for fn in os.listdir(samples_path):
            if fn.endswith('.json'):
                with open(os.path.join(samples_path, fn), 'r', encoding='utf-8') as f:
                    samples.append(json.load(f))
    elif os.path.isfile(samples_path):
        with open(samples_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            samples = data if isinstance(data, list) else [data]

    total = len(samples)
    has_upper = sum(1 for s in samples if s.get('has_upper_graph', False))
    confidences = [s.get('upper_graph_confidence', 0.0) for s in samples if s.get('has_upper_graph', False)]

    print(f"Total samples: {total}")
    print(f"Samples with upper_graph candidate: {has_upper} ({(has_upper/total*100) if total else 0:.2f}%)")
    if confidences:
        confidences_sorted = sorted(confidences)
        print(f"Upper confidences: min={confidences_sorted[0]:.3f}, median={confidences_sorted[len(confidences_sorted)//2]:.3f}, max={confidences_sorted[-1]:.3f}")
        # histogram
        bins = [0]*10
        for c in confidences:
            idx = min(int(c*10), 9)
            bins[idx] += 1
        print("Confidence histogram (0-0.1,...,0.9-1.0):")
        print(bins)
    else:
        print("No upper_graph confidences present.")

    # per-type counts if present in 'target_values' as dict
    type_counter = Counter()
    for s in samples:
        tv = s.get('target_values')
        if isinstance(tv, dict):
            for k, v in tv.items():
                if v:
                    type_counter[k] += 1
    if type_counter:
        print("Per-type positive counts:")
        for k, v in type_counter.most_common():
            print(f"  {k}: {v}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('samples', help='Path to samples dir or json file')
    args = parser.parse_args()
    analyze_samples(args.samples)














