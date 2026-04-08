"""Collect all unique prediction labels across all models and parse types.

Scans all result JSON files, flattens them, and strips array indices
to produce a canonical set of field labels that any model has ever predicted.
"""

import json
import re
from pathlib import Path

from kajima.extract_llm import FILES_DIR


def _strip_indices(key: str) -> str:
    """Remove array indices from a flattened key.

    e.g. "コア情報.岩石土区分[0].岩石土区分_下端深度"
      -> "コア情報.岩石土区分.岩石土区分_下端深度"
    """
    return re.sub(r"\[\d+\]\.?", ".", key).rstrip(".")


def _flatten_keys(data: object, prefix: str = "") -> set[str]:
    """Flatten a nested dict/list and return leaf keys."""
    keys: set[str] = set()

    if isinstance(data, dict):
        for k, v in data.items():
            key = f"{prefix}.{k}" if prefix else k
            keys |= _flatten_keys(v, key)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            keys |= _flatten_keys(item, f"{prefix}[{i}]")
    else:
        if prefix:
            keys.add(prefix)

    return keys


def collect_all_labels(files_dir: Path = FILES_DIR) -> set[str]:
    """Collect all unique labels from all prediction results."""
    all_labels: set[str] = set()

    for d in sorted(files_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("results_"):
            continue
        results_base = d

        for parse_dir in sorted(results_base.iterdir()):
            if not parse_dir.is_dir():
                continue

            for result_file in parse_dir.glob("*.json"):
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                raw_keys = _flatten_keys(data)
                for key in raw_keys:
                    all_labels.add(_strip_indices(key))

    return all_labels


def main() -> None:
    labels = sorted(collect_all_labels())
    output_path = FILES_DIR / "all_prediction_labels.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"Collected {len(labels)} unique labels -> {output_path}")
    for label in labels:
        print(f"  {label}")


if __name__ == "__main__":
    main()
