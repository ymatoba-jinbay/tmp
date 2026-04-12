"""End-to-end evaluation pipeline for kajima.

Chains the following steps into a single command:

  1. collect_labels       - expected_labels across all predictions
  2. evaluate_batch       - JSON evaluation per (llm, parse_type)
  3. summarize            - overall / subsection / pivot TSVs
  4. sheet_builder        - formatted Google Sheet (Summary / Detail /
                            Precision / Failed Patterns / Error Types)

Usage:

    # 全LLM × 全parse_type を一気に実行 → スプレッドシート作成
    uv run python -m kajima.pipeline --google-account you@example.com

    # 環境変数で固定しておくと毎回指定不要
    KAJIMA_GOOGLE_ACCOUNT=you@example.com uv run python -m kajima.pipeline

    # 評価だけ走らせてシート作成はスキップ
    uv run python -m kajima.pipeline --no-sheet

    # 特定のLLMだけ (カンマ区切り、省略時は results_{llm}/ の存在で自動検出)
    uv run python -m kajima.pipeline --llm gemini,claude

    # 評価は既に走っているので summarize + sheet だけ
    uv run python -m kajima.pipeline --skip-evaluate
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from kajima.collect_labels import collect_all_labels
from kajima.evaluate import evaluate_batch, section_key
from kajima.extract_llm import FILES_DIR, LLM_CHOICES, XML_DIR
from kajima.sheet_builder import SheetSpec, build_spreadsheet
from kajima.summarize import load_evaluations, write_summary_tsvs

# -----------------------------------------------------------------------------
# Step 1-3: evaluate + summarize
# -----------------------------------------------------------------------------


def _detect_llms(files_dir: Path) -> list[str]:
    """Return LLMs that have a ``results_{llm}/`` directory."""
    found: list[str] = []
    for d in sorted(files_dir.iterdir()):
        if d.is_dir() and d.name.startswith("results_"):
            llm = d.name.removeprefix("results_")
            if llm in LLM_CHOICES:
                found.append(llm)
    return found


def _detect_parse_types(results_dir: Path) -> list[str]:
    if not results_dir.exists():
        return []
    return sorted(d.name for d in results_dir.iterdir() if d.is_dir())


def run_evaluations(
    llms: list[str],
    files_dir: Path,
    xml_dir: Path,
) -> None:
    """Run evaluate_batch for every (llm, parse_type) combination."""
    print("Collecting expected labels from all predictions...")
    expected_labels = collect_all_labels(files_dir)
    print(f"Collected {len(expected_labels)} expected labels")

    for llm in llms:
        results_base = files_dir / f"results_{llm}"
        parse_types = _detect_parse_types(results_base)
        if not parse_types:
            print(f"[{llm}] no parse type dirs found in {results_base}, skipping")
            continue

        for parse_type in parse_types:
            result_dir = results_base / parse_type
            output_path = files_dir / f"evaluations_{llm}" / f"{parse_type}.json"
            print(f"\n=== [{llm}] parse_type={parse_type} ===")
            evaluate_batch(result_dir, xml_dir, output_path, expected_labels)


# -----------------------------------------------------------------------------
# Step 4: sheet data builders
# -----------------------------------------------------------------------------


def build_summary_spec(evals: dict[tuple[str, str], dict]) -> SheetSpec:
    headers = [
        "llm",
        "parse_type",
        "precision",
        "recall",
        "f1",
        "files",
        "correct",
        "incorrect",
        "not_extracted",
        "evaluated",
        "total_retries",
    ]
    rows: list[list[Any]] = []
    for (llm, pt), s in sorted(
        evals.items(), key=lambda x: x[1]["overall_f1"], reverse=True
    ):
        rows.append(
            [
                llm,
                pt,
                s["overall_precision"],
                s["overall_recall"],
                s["overall_f1"],
                s["total_files"],
                s["total_correct"],
                s["total_incorrect"],
                s["total_not_extracted"],
                s["total_evaluated"],
                s.get("total_retries", 0),
            ]
        )
    return SheetSpec(
        title="Summary",
        headers=headers,
        rows=rows,
        frozen_rows=1,
        frozen_cols=2,
        gradient_cols=[(2, 5)],  # precision, recall, f1
        percent_cols=[2, 3, 4],
    )


def build_detail_spec(evals: dict[tuple[str, str], dict]) -> SheetSpec:
    headers = [
        "llm",
        "parse_type",
        "section",
        "precision",
        "recall",
        "f1",
        "correct",
        "incorrect",
        "not_extracted",
        "evaluated",
    ]
    rows: list[list[Any]] = []
    for (llm, pt), s in sorted(evals.items()):
        sub = s.get("section_analysis", {}).get("sub", {})
        for section, stats in sorted(sub.items()):
            rows.append(
                [
                    llm,
                    pt,
                    section,
                    stats["precision"],
                    stats["recall"],
                    stats["f1"],
                    stats["correct"],
                    stats["incorrect"],
                    stats["not_extracted"],
                    stats["evaluated"],
                ]
            )
    return SheetSpec(
        title="Detail",
        headers=headers,
        rows=rows,
        frozen_rows=1,
        frozen_cols=0,
        gradient_cols=[(3, 6)],
        percent_cols=[3, 4, 5],
        sort_col=3,
    )


def build_precision_pivot_spec(evals: dict[tuple[str, str], dict]) -> SheetSpec:
    """sub_section × (llm_parse_type) matrix of precision values."""
    col_keys: list[str] = []
    all_sections: set[str] = set()
    pivot: dict[tuple[str, str], float] = {}
    for (llm, pt), s in sorted(evals.items()):
        ck = f"{llm}_{pt}"
        col_keys.append(ck)
        for section, stats in s.get("section_analysis", {}).get("sub", {}).items():
            all_sections.add(section)
            pivot[(section, ck)] = stats["precision"]

    headers = ["sub_section"] + col_keys
    rows: list[list[Any]] = []
    for section in sorted(all_sections):
        row: list[Any] = [section]
        for ck in col_keys:
            row.append(pivot.get((section, ck), ""))
        rows.append(row)

    n_data_cols = len(col_keys)
    return SheetSpec(
        title="Precision",
        headers=headers,
        rows=rows,
        frozen_rows=1,
        frozen_cols=1,
        gradient_cols=[(1, 1 + n_data_cols)],
        percent_cols=list(range(1, 1 + n_data_cols)),
    )


def build_failed_patterns_spec(
    evals: dict[tuple[str, str], dict],
    precision_drop: float = 0.2,
    per_section_limit: int = 10,
) -> SheetSpec:
    """Subsections where precision is >=drop below overall.

    For each such subsection collect up to ``per_section_limit`` incorrect
    examples. Columns match the reference sheet.
    """
    headers = [
        "llm",
        "parse_type",
        "subsection",
        "file",
        "field",
        "error_type",
        "xml",
        "llm_value",
    ]
    rows: list[list[Any]] = []
    for (llm, pt), s in sorted(evals.items()):
        overall_precision = s["overall_precision"]
        threshold = overall_precision - precision_drop
        sub_stats = s.get("section_analysis", {}).get("sub", {})

        # index incorrect_examples by subsection
        by_section: dict[str, list[dict]] = defaultdict(list)
        for ex in s.get("incorrect_examples", []):
            by_section[section_key(ex["field"])].append(ex)

        for section, stats in sorted(sub_stats.items()):
            if stats["evaluated"] == 0:
                continue
            if stats["precision"] >= threshold:
                continue
            examples = by_section.get(section, [])[:per_section_limit]
            for ex in examples:
                rows.append(
                    [
                        llm,
                        pt,
                        section,
                        ex.get("file", ""),
                        ex.get("field", ""),
                        ex.get("error_type", ""),
                        _clean_cell(ex.get("xml", "")),
                        _clean_cell(ex.get("llm", "")),
                    ]
                )
    return SheetSpec(
        title="Failed Patterns",
        headers=headers,
        rows=rows,
        frozen_rows=1,
        frozen_cols=0,
    )


def _clean_cell(value) -> str:
    """Strip tabs/newlines so they don't break the sheet layout."""
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\n", " ")


def build_error_types_spec(
    evals: dict[tuple[str, str], dict],
) -> SheetSpec:
    """Per-llm×parse_type error-type breakdown + '全体' summary rows.

    Columns (matches reference sheet):
      llm, parse_type, total_incorrect,
      false_positive_count, partial+wrong count,
      partial_match_count, wrong_value_count,
      false_positive_ratio, partial+wrong ratio,
      partial_match_ratio, wrong_value_ratio
    """
    headers = [
        "llm",
        "parse_type",
        "total_incorrect",
        "false_positive_count",
        "partial+wrong count",
        "partial_match_count",
        "wrong_value_count",
        "false_positive_ratio",
        "partial+wrong ratio",
        "partial_match_ratio",
        "wrong_value_ratio",
    ]

    # llm -> parse_type -> error_type -> count
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for (llm, pt), s in evals.items():
        et_counts: dict[str, int] = defaultdict(int)
        for ex in s.get("incorrect_examples", []):
            et_counts[ex["error_type"]] += 1
        counts[llm][pt] = dict(et_counts)

    def row_for(llm: str, pt: str, c: dict[str, int]) -> list[Any]:
        fp = c.get("false_positive", 0)
        pm = c.get("partial_match", 0)
        wv = c.get("wrong_value", 0)
        pw = pm + wv
        total = fp + pw
        if total == 0:
            return [llm, pt, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0]
        return [
            llm,
            pt,
            total,
            fp,
            pw,
            pm,
            wv,
            fp / total,
            pw / total,
            pm / total,
            wv / total,
        ]

    # Stable ordering: follow LLM_CHOICES then parse_type alpha.
    ordered_llms = [llm for llm in LLM_CHOICES if llm in counts]

    rows: list[list[Any]] = []
    for llm in ordered_llms:
        for pt in sorted(counts[llm]):
            rows.append(row_for(llm, pt, counts[llm][pt]))

    # '全体' summary rows (one per llm, across all parse_types)
    summary_rows: list[list[Any]] = []
    for llm in ordered_llms:
        agg: dict[str, int] = defaultdict(int)
        for pt_counts in counts[llm].values():
            for et, c in pt_counts.items():
                agg[et] += c
        summary_rows.append(row_for(llm, "全体", dict(agg)))
    rows.extend(summary_rows)

    return SheetSpec(
        title="Error Types",
        headers=headers,
        rows=rows,
        frozen_rows=1,
        frozen_cols=0,
        # Counts of false_positive / partial+wrong columns get a gradient.
        gradient_cols=[(3, 5)],
        summary_gradient=(3, 5, len(summary_rows)),
        percent_cols=[7, 8, 9, 10],
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def run_pipeline(
    llms: list[str] | None,
    files_dir: Path,
    xml_dir: Path,
    *,
    skip_evaluate: bool = False,
    make_sheet: bool = True,
    sheet_title: str | None = None,
    share_with: str | None = None,
    google_account: str | None = None,
) -> dict[str, str] | None:
    """Run collect_labels → evaluate → summarize → sheet in one go."""
    target_llms = llms or _detect_llms(files_dir)
    if not target_llms:
        raise SystemExit(f"No results_{{llm}}/ directories in {files_dir}")

    if not skip_evaluate:
        run_evaluations(target_llms, files_dir, xml_dir)
    else:
        print("--skip-evaluate: reusing existing evaluations_*/*.json")

    evals = load_evaluations(files_dir, llm_filter=target_llms)
    if not evals:
        raise SystemExit("No evaluations_*/*.json found; nothing to summarize")
    print(f"\nLoaded {len(evals)} evaluation results")
    write_summary_tsvs(evals, files_dir)

    if not make_sheet:
        return None

    title = sheet_title or (
        "kajima 精度検証 " + datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    print(f"\nBuilding Google Sheet: {title}")
    specs = [
        build_summary_spec(evals),
        build_detail_spec(evals),
        build_precision_pivot_spec(evals),
        build_failed_patterns_spec(evals),
        build_error_types_spec(evals),
    ]
    info = build_spreadsheet(
        title=title,
        specs=specs,
        google_account=google_account,
        share_with=share_with,
    )
    print(f"Spreadsheet created: {info['url']}")
    if share_with:
        print(f"Shared with: {share_with}")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full kajima evaluation pipeline."
    )
    parser.add_argument(
        "--llm",
        default=None,
        help="Comma-separated LLM names (default: auto-detect from results_*/)",
    )
    parser.add_argument(
        "--xml-dir",
        default=None,
        help=f"XML directory (default: {XML_DIR})",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip the evaluate step; reuse existing evaluations_*/*.json",
    )
    parser.add_argument(
        "--no-sheet",
        action="store_true",
        help="Skip Google Sheet creation (evaluate + TSV summarize only)",
    )
    parser.add_argument(
        "--sheet-title",
        default=None,
        help="Title for the created spreadsheet",
    )
    parser.add_argument(
        "--google-account",
        default=os.environ.get("KAJIMA_GOOGLE_ACCOUNT"),
        help=(
            "Googleアカウントemail(OAuth, デフォルト認証方式)。"
            "gogでauth済みのアカウントを指定する。"
            "(default: $KAJIMA_GOOGLE_ACCOUNT)"
        ),
    )
    parser.add_argument(
        "--share-with",
        default=os.environ.get("KAJIMA_SHEET_SHARE_WITH"),
        help=(
            "作成後にeditor権限でshareする追加アカウント "
            "(default: $KAJIMA_SHEET_SHARE_WITH)"
        ),
    )
    args = parser.parse_args()

    llms = [x.strip() for x in args.llm.split(",")] if args.llm else None
    xml_dir = Path(args.xml_dir) if args.xml_dir else XML_DIR

    run_pipeline(
        llms=llms,
        files_dir=FILES_DIR,
        xml_dir=xml_dir,
        skip_evaluate=args.skip_evaluate,
        make_sheet=not args.no_sheet,
        sheet_title=args.sheet_title,
        share_with=args.share_with,
        google_account=args.google_account,
    )


if __name__ == "__main__":
    main()
