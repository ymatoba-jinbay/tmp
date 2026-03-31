"""Evaluate LLM extraction results against XML ground truth."""

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from kajima.extract_llm import FILES_DIR, PARSE_TYPES, XML_DIR
from kajima.parse_xml import parse_xml


def _normalize(value: str) -> str:
    """Normalize a value for comparison."""
    value = unicodedata.normalize("NFKC", value).strip()
    # 各種ハイフン・ダッシュ・マイナス記号を統一
    value = re.sub(r"[\u2010-\u2015\u2212\uFF0D\uFF70\u30FC]", "-", value)
    # 空白・括弧類（半角/全角）を除去して比較精度を上げる
    value = re.sub(r"[\s()\[\]{}（）［］｛｝【】「」『』〈〉《》〔〕]", "", value)
    # 区切り文字（読点、カンマ、改行）をソートして比較できるよう統一
    value = re.sub(r"[、,，\n]+", ",", value)
    return value


def _flatten(
    data: object,
    prefix: str = "",
    out: dict[str, str] | None = None,
) -> dict[str, str]:
    """Flatten a nested dict/list into dot-notation key-value pairs."""
    if out is None:
        out = {}

    if isinstance(data, dict):
        for k, v in data.items():
            key = f"{prefix}.{k}" if prefix else k
            _flatten(v, key, out)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            _flatten(item, f"{prefix}[{i}]", out)
    else:
        out[prefix] = str(data) if data is not None else ""

    return out


def _section_key(field: str) -> str:
    """Extract section key (top 2 levels) from a dot-notation field.

    e.g. "コア情報.標準貫入試験[0].開始深度" -> "コア情報.標準貫入試験"
    """
    parts = re.split(r"[.\[]", field)
    depth = min(len(parts), 2)
    return ".".join(parts[:depth])


def _top_section_key(field: str) -> str:
    """Extract top-level section key from a dot-notation field.

    e.g. "コア情報.標準貫入試験[0].開始深度" -> "コア情報"
    """
    parts = re.split(r"[.\[]", field)
    return parts[0]


def _classify_error(xml_norm: str, llm_norm: str) -> str:
    """Classify the type of mismatch between xml and llm values."""
    try:
        xml_f = float(xml_norm)
        llm_f = float(llm_norm)
        if xml_f != 0 and abs(xml_f - llm_f) / abs(xml_f) < 0.01:
            return "numeric_close"
    except ValueError:
        pass

    if xml_norm in llm_norm or llm_norm in xml_norm:
        return "partial_match"

    return "wrong_value"


def _match_arrays(
    xml_items: list[dict],
    llm_items: list[dict],
) -> list[tuple[int, int | None]]:
    """Match XML array elements to LLM array elements by best field overlap.

    Returns list of (xml_index, llm_index or None) pairs.
    Each LLM element is used at most once (greedy best-match).
    """
    if not xml_items:
        return []

    used_llm: set[int] = set()
    matches: list[tuple[int, int | None]] = []

    for xi, x_item in enumerate(xml_items):
        x_flat = _flatten(x_item)
        best_j: int | None = None
        best_score = 0

        for lj, l_item in enumerate(llm_items):
            if lj in used_llm:
                continue
            l_flat = _flatten(l_item)
            score = sum(
                1
                for k, v in x_flat.items()
                if _normalize(v)
                and k in l_flat
                and _normalize(v) == _normalize(l_flat[k])
            )
            if score > best_score:
                best_score = score
                best_j = lj

        matches.append((xi, best_j))
        if best_j is not None:
            used_llm.add(best_j)

    return matches


def _compare_values(
    key: str,
    xml_value: str,
    llm_value: str,
    details: list[dict],
    counters: dict,
) -> None:
    """Compare a single xml vs llm value and update counters/details."""
    xml_norm = _normalize(xml_value)
    if not xml_norm:
        return

    llm_norm = _normalize(llm_value)

    if not llm_norm:
        counters["not_extracted"] += 1
        details.append({
            "field": key,
            "top_section": _top_section_key(key),
            "section": _section_key(key),
            "status": "not_extracted",
            "xml": xml_value,
            "llm": "",
        })
        return

    if xml_norm == llm_norm:
        counters["correct"] += 1
        details.append({
            "field": key,
            "top_section": _top_section_key(key),
            "section": _section_key(key),
            "status": "correct",
            "xml": xml_value,
            "llm": llm_value,
        })
    else:
        counters["incorrect"] += 1
        error_type = _classify_error(xml_norm, llm_norm)
        details.append({
            "field": key,
            "top_section": _top_section_key(key),
            "section": _section_key(key),
            "status": "incorrect",
            "error_type": error_type,
            "xml": xml_value,
            "llm": llm_value,
        })


def _evaluate_recursive(
    xml_data: object,
    llm_data: object,
    prefix: str,
    details: list[dict],
    counters: dict,
) -> None:
    """Recursively evaluate xml vs llm data with smart array matching."""
    if isinstance(xml_data, list):
        llm_list = llm_data if isinstance(llm_data, list) else []
        # XML/LLM両方がdictのリストならベストマッチ
        xml_dicts = [
            x for x in xml_data if isinstance(x, dict)
        ]
        llm_dicts = [
            x for x in llm_list if isinstance(x, dict)
        ]
        if xml_dicts and len(xml_dicts) == len(xml_data):
            matches = _match_arrays(xml_dicts, llm_dicts)
            for xi, lj in matches:
                matched_llm = (
                    llm_dicts[lj] if lj is not None else {}
                )
                _evaluate_recursive(
                    xml_dicts[xi],
                    matched_llm,
                    f"{prefix}[{xi}]",
                    details,
                    counters,
                )
        else:
            # scalar list: compare by index
            for i, x_item in enumerate(xml_data):
                l_item = (
                    llm_list[i]
                    if i < len(llm_list)
                    else None
                )
                _evaluate_recursive(
                    x_item,
                    l_item,
                    f"{prefix}[{i}]",
                    details,
                    counters,
                )
    elif isinstance(xml_data, dict):
        llm_dict = llm_data if isinstance(llm_data, dict) else {}
        for k, v in xml_data.items():
            key = f"{prefix}.{k}" if prefix else k
            _evaluate_recursive(
                v, llm_dict.get(k), key, details, counters
            )
    else:
        xml_str = str(xml_data) if xml_data is not None else ""
        llm_str = str(llm_data) if llm_data is not None else ""
        _compare_values(prefix, xml_str, llm_str, details, counters)


def evaluate_single(
    xml_data: dict,
    llm_data: dict,
) -> dict:
    """Evaluate a single file.

    Evaluation focuses on precision: only fields where LLM returned
    a non-empty value are scored as correct/incorrect.
    Fields where LLM returned empty are tracked separately as
    not_extracted (may or may not be present in the PDF).

    Array elements are matched by best field overlap, not by index.
    """
    details: list[dict] = []
    counters = {"correct": 0, "incorrect": 0, "not_extracted": 0}

    _evaluate_recursive(xml_data, llm_data, "", details, counters)

    evaluated = counters["correct"] + counters["incorrect"]
    precision = (
        counters["correct"] / evaluated if evaluated > 0 else 0.0
    )

    return {
        "correct": counters["correct"],
        "incorrect": counters["incorrect"],
        "not_extracted": counters["not_extracted"],
        "evaluated": evaluated,
        "precision": precision,
        "details": details,
    }


def _build_stats(counts: dict) -> dict:
    """Build stats dict from raw counts."""
    evaluated = counts["correct"] + counts["incorrect"]
    return {
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "not_extracted": counts["not_extracted"],
        "evaluated": evaluated,
        "precision": (
            counts["correct"] / evaluated
            if evaluated > 0
            else 0.0
        ),
        "error_types": dict(counts["error_types"]),
    }


def _new_counter() -> dict:
    return {
        "correct": 0,
        "incorrect": 0,
        "not_extracted": 0,
        "error_types": defaultdict(int),
    }


def _aggregate_sections(
    all_details: list[dict],
) -> dict[str, dict]:
    """Aggregate evaluation results by section.

    Returns dict with two keys:
      - "top": top-level section stats
      - "sub": sub-section (top 2 levels) stats
    """
    top_sections: dict[str, dict] = defaultdict(_new_counter)
    sub_sections: dict[str, dict] = defaultdict(_new_counter)

    for d in all_details:
        status = d["status"]
        for key, target in [
            (d["top_section"], top_sections),
            (d["section"], sub_sections),
        ]:
            target[key][status] += 1
            if status == "incorrect":
                target[key]["error_types"][
                    d["error_type"]
                ] += 1

    return {
        "top": {
            k: _build_stats(v)
            for k, v in sorted(top_sections.items())
        },
        "sub": {
            k: _build_stats(v)
            for k, v in sorted(sub_sections.items())
        },
    }


def evaluate_batch(
    result_dir: Path,
    xml_dir: Path,
    output_path: Path,
) -> dict:
    """Run batch evaluation."""
    all_file_results: list[dict] = []
    all_details: list[dict] = []

    for result_file in sorted(result_dir.glob("*.json")):
        stem = result_file.stem
        xml_file = xml_dir / f"{stem}.xml"

        try:
            xml_data = parse_xml(xml_file)
        except FileNotFoundError:
            print(
                f"XML not found for {result_file.name}, skipping"
            )
            continue

        with open(result_file, "r", encoding="utf-8") as f:
            llm_data = json.load(f)

        result = evaluate_single(xml_data, llm_data)
        result["file"] = stem
        all_file_results.append(result)
        all_details.extend(result["details"])

    total_correct = sum(r["correct"] for r in all_file_results)
    total_incorrect = sum(r["incorrect"] for r in all_file_results)
    total_evaluated = total_correct + total_incorrect
    total_not_extracted = sum(
        r["not_extracted"] for r in all_file_results
    )

    section_analysis = _aggregate_sections(all_details)

    error_type_totals: dict[str, int] = defaultdict(int)
    for d in all_details:
        if d["status"] == "incorrect":
            error_type_totals[d["error_type"]] += 1

    incorrect_examples: list[dict] = [
        {
            "file": r["file"],
            "field": d["field"],
            "error_type": d["error_type"],
            "xml": d["xml"],
            "llm": d["llm"],
        }
        for r in all_file_results
        for d in r["details"]
        if d["status"] == "incorrect"
    ]

    summary = {
        "total_files": len(all_file_results),
        "total_correct": total_correct,
        "total_incorrect": total_incorrect,
        "total_evaluated": total_evaluated,
        "total_not_extracted": total_not_extracted,
        "overall_precision": (
            total_correct / total_evaluated
            if total_evaluated > 0
            else 0.0
        ),
        "error_type_totals": dict(error_type_totals),
        "section_analysis": section_analysis,
        "incorrect_examples": incorrect_examples,
        "per_file": [
            {k: v for k, v in r.items() if k != "details"}
            for r in all_file_results
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Evaluation saved: {output_path}")

    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as txt_f:
        _print_summary(summary, file=txt_f)
    print(f"Summary saved: {txt_path}")

    return summary


def _print_summary(summary: dict, file=None) -> None:
    """Print evaluation summary to stdout and optionally to a file object."""
    def _p(msg: str = "") -> None:
        print(msg)
        if file is not None:
            print(msg, file=file)

    _p(f"\nFiles: {summary['total_files']}")
    _p(
        f"Evaluated: {summary['total_evaluated']} "
        f"(not extracted: {summary['total_not_extracted']})"
    )
    _p(
        f"Correct: {summary['total_correct']}  "
        f"Incorrect: {summary['total_incorrect']}"
    )
    _p(
        f"Precision: {summary['overall_precision']:.2%}"
    )

    if summary["error_type_totals"]:
        _p("\nError types:")
        for etype, count in sorted(
            summary["error_type_totals"].items(),
            key=lambda x: -x[1],
        ):
            _p(f"  {etype}: {count}")

    section_analysis = summary["section_analysis"]

    for label, key in [
        ("top-level", "top"),
        ("sub-section", "sub"),
    ]:
        _p(f"\nSection analysis ({label}):")
        for section, stats in section_analysis[key].items():
            if stats["evaluated"] == 0:
                continue
            _p(
                f"  {section}: "
                f"{stats['precision']:.0%} "
                f"({stats['correct']}/{stats['evaluated']})"
                + (
                    f"  errors: {dict(stats['error_types'])}"
                    if stats["error_types"]
                    else ""
                )
            )

    worst = sorted(
        summary["per_file"],
        key=lambda r: r["precision"],
    )[:5]
    if worst and worst[0]["precision"] < 1.0:
        _p("\nWorst 5 files:")
        for r in worst:
            _p(
                f"  {r['file']}: {r['precision']:.0%} "
                f"({r['correct']}/{r['evaluated']}, "
                f"{r['incorrect']} incorrect)"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LLM extraction accuracy"
    )
    parser.add_argument(
        "--parse-type",
        choices=PARSE_TYPES + ["all"],
        default="pdf",
        help="Input parse type (default: pdf, "
        "'all' to run all types found in results dir)",
    )
    parser.add_argument(
        "--llm",
        choices=["gemini", "claude"],
        default="gemini",
    )
    parser.add_argument(
        "--xml-dir",
        default=None,
        help="XML directory (default: kajima/files/xml)",
    )
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir) if args.xml_dir else XML_DIR

    if args.parse_type == "all":
        results_base = FILES_DIR / f"results_{args.llm}"
        if not results_base.exists():
            print(f"Results base directory not found: {results_base}")
            raise SystemExit(1)
        parse_types = sorted(
            d.name
            for d in results_base.iterdir()
            if d.is_dir()
        )
        if not parse_types:
            print(f"No parse type directories found in: {results_base}")
            raise SystemExit(1)
    else:
        parse_types = [args.parse_type]

    for parse_type in parse_types:
        result_dir = (
            FILES_DIR / f"results_{args.llm}" / parse_type
        )
        output_path = (
            FILES_DIR
            / f"evaluations_{args.llm}"
            / f"{parse_type}.json"
        )

        if not result_dir.exists():
            print(f"Result directory not found: {result_dir}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"LLM: {args.llm}")
        print(f"Parse type: {parse_type}")
        print(f"Results: {result_dir}")
        print(f"Output: {output_path}")

        evaluate_batch(result_dir, xml_dir, output_path)
