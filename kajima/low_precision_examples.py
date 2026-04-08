"""各LLMでprecisionが低いsubsectionのincorrect_examplesを抽出するスクリプト。

overall precisionよりも0.2以上低いsubsectionを対象に、
incorrect_examplesを最大10件取得し、TSVとして出力する。
"""

import json
from pathlib import Path

from kajima.extract_llm import FILES_DIR, LLM_CHOICES

PARSE_TYPES = ["pdf", "jpg", "pymupdf", "pymupdf4llm", "position", "position_spatial"]
OUTPUT_PATH = FILES_DIR / "low_precision_incorrect_examples.tsv"


def main() -> None:
    rows: list[list[str]] = []
    header = ["llm", "parse_type", "subsection", "subsection_precision", "overall_precision", "diff", "file", "field", "error_type", "xml", "llm_value"]

    for llm in LLM_CHOICES:
        for parse_type in PARSE_TYPES:
            eval_path = FILES_DIR / f"evaluations_{llm}" / f"{parse_type}.json"
            if not eval_path.exists():
                continue

            with open(eval_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            overall_precision = data["overall_precision"]
            sub_sections = data["section_analysis"]["sub"]
            incorrect_examples = data["incorrect_examples"]

            # Build index: subsection -> incorrect examples
            from kajima.evaluate import _section_key
            sub_examples: dict[str, list[dict]] = {}
            for ex in incorrect_examples:
                sec = _section_key(ex["field"])
                sub_examples.setdefault(sec, []).append(ex)

            # Find subsections with precision < overall - 0.2
            threshold = overall_precision - 0.2
            for section_name, stats in sorted(sub_sections.items()):
                if stats["evaluated"] == 0:
                    continue
                sec_precision = stats["precision"]
                if sec_precision >= threshold:
                    continue

                examples = sub_examples.get(section_name, [])[:10]
                if not examples:
                    continue

                diff = sec_precision - overall_precision
                for ex in examples:
                    rows.append([
                        llm,
                        parse_type,
                        section_name,
                        f"{sec_precision:.4f}",
                        f"{overall_precision:.4f}",
                        f"{diff:.4f}",
                        ex["file"],
                        ex["field"],
                        ex["error_type"],
                        ex.get("xml", ""),
                        ex.get("llm", ""),
                    ])

    # Write TSV
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            # Escape tabs/newlines in values
            escaped = [v.replace("\t", " ").replace("\n", " ") for v in row]
            f.write("\t".join(escaped) + "\n")

    print(f"Output: {OUTPUT_PATH}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
