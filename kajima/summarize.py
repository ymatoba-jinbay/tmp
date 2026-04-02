"""Summarize evaluation results across all models and parse types.

Reads evaluation JSON files from all evaluations_*/ directories
and produces overall and subsection TSV summaries.
"""

import json
from pathlib import Path

from kajima.extract_llm import FILES_DIR


def _find_eval_dirs(files_dir: Path) -> dict[str, Path]:
    """Find all evaluations_*/ directories and return {llm_name: path}."""
    result = {}
    for d in sorted(files_dir.iterdir()):
        if d.is_dir() and d.name.startswith("evaluations_"):
            llm = d.name.removeprefix("evaluations_")
            result[llm] = d
    return result


def _load_summaries(
    eval_dirs: dict[str, Path],
) -> dict[tuple[str, str], dict]:
    """Load all evaluation JSONs. Returns {(llm, parse_type): summary}."""
    summaries: dict[tuple[str, str], dict] = {}
    for llm, eval_dir in eval_dirs.items():
        for json_file in sorted(eval_dir.glob("*.json")):
            parse_type = json_file.stem
            with open(json_file, "r", encoding="utf-8") as f:
                summaries[(llm, parse_type)] = json.load(f)
    return summaries


def write_summary_tsvs(
    summaries: dict[tuple[str, str], dict],
    output_dir: Path,
) -> None:
    """Write overall and subsection summary TSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Overall TSV
    overall_path = output_dir / "evaluation_summary_overall.tsv"
    with open(overall_path, "w", encoding="utf-8") as f:
        f.write(
            "llm\tparse_type\tfiles\tcorrect\tincorrect\t"
            "not_extracted\tevaluated\t"
            "precision\trecall\tf1\n"
        )
        for (llm, pt), s in sorted(
            summaries.items(), key=lambda x: x[1]["overall_f1"], reverse=True
        ):
            f.write(
                f"{llm}\t{pt}\t{s['total_files']}\t"
                f"{s['total_correct']}\t{s['total_incorrect']}\t"
                f"{s['total_not_extracted']}\t{s['total_evaluated']}\t"
                f"{s['overall_precision']:.4f}\t"
                f"{s['overall_recall']:.4f}\t"
                f"{s['overall_f1']:.4f}\n"
            )
    print(f"Overall TSV saved: {overall_path}")

    # Subsection TSV
    sub_path = output_dir / "evaluation_summary_subsection.tsv"
    with open(sub_path, "w", encoding="utf-8") as f:
        f.write(
            "llm\tparse_type\tsection\tcorrect\tincorrect\t"
            "not_extracted\tevaluated\t"
            "precision\trecall\tf1\n"
        )
        for (llm, pt), s in sorted(
            summaries.items(), key=lambda x: x[1]["overall_f1"], reverse=True
        ):
            for section, stats in sorted(s.get("section_analysis", {}).get("sub", {}).items()):
                f.write(
                    f"{llm}\t{pt}\t{section}\t"
                    f"{stats['correct']}\t{stats['incorrect']}\t"
                    f"{stats['not_extracted']}\t{stats['evaluated']}\t"
                    f"{stats['precision']:.4f}\t"
                    f"{stats['recall']:.4f}\t"
                    f"{stats['f1']:.4f}\n"
                )
    print(f"Subsection TSV saved: {sub_path}")


def main() -> None:
    eval_dirs = _find_eval_dirs(FILES_DIR)
    if not eval_dirs:
        print(f"No evaluations_*/ directories found in {FILES_DIR}")
        raise SystemExit(1)

    print(f"Found evaluation directories: {list(eval_dirs.keys())}")
    summaries = _load_summaries(eval_dirs)
    print(f"Loaded {len(summaries)} evaluation results")

    write_summary_tsvs(summaries, FILES_DIR)


if __name__ == "__main__":
    main()
