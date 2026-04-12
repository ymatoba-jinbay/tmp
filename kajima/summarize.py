"""Summarize evaluation results across all models and parse types.

Reads evaluation JSON files from all evaluations_*/ directories
and produces overall and subsection TSV summaries.
"""

import json
from pathlib import Path

from kajima.extract_llm import FILES_DIR


def load_evaluations(
    files_dir: Path,
    llm_filter: list[str] | None = None,
) -> dict[tuple[str, str], dict]:
    """Load all evaluation JSONs.

    Returns {(llm, parse_type): summary_dict}.
    If ``llm_filter`` is given, only those LLMs are loaded.
    """
    out: dict[tuple[str, str], dict] = {}
    for d in sorted(files_dir.iterdir()):
        if not (d.is_dir() and d.name.startswith("evaluations_")):
            continue
        llm = d.name.removeprefix("evaluations_")
        if llm_filter is not None and llm not in llm_filter:
            continue
        for jf in sorted(d.glob("*.json")):
            with open(jf, "r", encoding="utf-8") as f:
                out[(llm, jf.stem)] = json.load(f)
    return out


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
            "precision\trecall\tf1\t"
            "total_retries\tretried_files\n"
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
                f"{s['overall_f1']:.4f}\t"
                f"{s.get('total_retries', 0)}\t"
                f"{s.get('retried_files', 0)}\n"
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

    # Pivot TSV: rows=subsection, columns={llm}_{parse_type}, values=precision
    pivot_path = output_dir / "evaluation_summary_subsection_pivot.tsv"

    # Collect all column keys and all subsections
    col_keys: list[str] = []
    all_sections: set[str] = set()
    for (llm, pt), s in sorted(summaries.items()):
        col_keys.append(f"{llm}_{pt}")
        for section in s.get("section_analysis", {}).get("sub", {}):
            all_sections.add(section)

    # Build lookup: (section, col_key) -> precision
    pivot: dict[tuple[str, str], float] = {}
    for (llm, pt), s in summaries.items():
        col_key = f"{llm}_{pt}"
        for section, stats in s.get("section_analysis", {}).get("sub", {}).items():
            pivot[(section, col_key)] = stats["precision"]

    with open(pivot_path, "w", encoding="utf-8") as f:
        f.write("sub_section\t" + "\t".join(col_keys) + "\n")
        for section in sorted(all_sections):
            vals = [
                f"{pivot[(section, ck)]:.4f}" if (section, ck) in pivot else ""
                for ck in col_keys
            ]
            f.write(f"{section}\t" + "\t".join(vals) + "\n")
    print(f"Subsection pivot TSV saved: {pivot_path}")


def main() -> None:
    summaries = load_evaluations(FILES_DIR)
    if not summaries:
        print(f"No evaluations_*/ directories found in {FILES_DIR}")
        raise SystemExit(1)

    print(f"Loaded {len(summaries)} evaluation results")
    write_summary_tsvs(summaries, FILES_DIR)


if __name__ == "__main__":
    main()
