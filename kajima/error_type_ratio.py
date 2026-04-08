"""LLMごとのerror_type割合を集計してTSVで出力するスクリプト。"""

import json
from collections import defaultdict
from pathlib import Path

from kajima.extract_llm import FILES_DIR, LLM_CHOICES

PARSE_TYPES = ["pdf", "jpg", "pymupdf", "pymupdf4llm", "position", "position_spatial"]
OUTPUT_PATH = FILES_DIR / "error_type_ratio.tsv"


def _make_row(label_cols: list[str], et_counts: dict[str, int], all_error_types: list[str]) -> list[str]:
    total = sum(et_counts.values())
    row = label_cols + [str(total)]
    for et in all_error_types:
        row.append(str(et_counts.get(et, 0)))
    for et in all_error_types:
        ratio = et_counts.get(et, 0) / total if total > 0 else 0.0
        row.append(f"{ratio:.4f}")
    return row


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    print(f"Output: {path}")


def main() -> None:
    # llm -> parse_type -> error_type -> count
    counts: dict[str, dict[str, dict[str, int]]] = {}

    for llm in LLM_CHOICES:
        counts[llm] = {}
        for parse_type in PARSE_TYPES:
            eval_path = FILES_DIR / f"evaluations_{llm}" / f"{parse_type}.json"
            if not eval_path.exists():
                continue
            with open(eval_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            et = defaultdict(int)
            for ex in data["incorrect_examples"]:
                et[ex["error_type"]] += 1
            counts[llm][parse_type] = dict(et)

    # Collect all error types
    all_error_types = sorted(
        {et for llm_d in counts.values() for pt_d in llm_d.values() for et in pt_d}
    )

    et_header_suffix = [f"{et}_count" for et in all_error_types] + [f"{et}_ratio" for et in all_error_types]

    header = ["llm", "parse_type", "total_incorrect"] + et_header_suffix
    rows: list[list[str]] = []

    for llm in LLM_CHOICES:
        # Per parse_type rows
        for parse_type in PARSE_TYPES:
            if parse_type not in counts[llm]:
                continue
            rows.append(_make_row([llm, parse_type], counts[llm][parse_type], all_error_types))

        # All parse_types combined
        total_et: dict[str, int] = defaultdict(int)
        for pt_d in counts[llm].values():
            for et, c in pt_d.items():
                total_et[et] += c
        if sum(total_et.values()) > 0:
            rows.append(_make_row([llm, "all"], dict(total_et), all_error_types))

    _write_tsv(OUTPUT_PATH, header, rows)

    # --- Print summary ---
    print(f"\n=== Summary (overall) ===")
    for llm in LLM_CHOICES:
        total_all = defaultdict(int)
        grand_total = 0
        for pt_d in counts[llm].values():
            for et, c in pt_d.items():
                total_all[et] += c
                grand_total += c
        if grand_total == 0:
            continue
        print(f"\n{llm} (total incorrect: {grand_total}):")
        for et in all_error_types:
            c = total_all[et]
            print(f"  {et}: {c} ({c/grand_total:.1%})")


if __name__ == "__main__":
    main()
