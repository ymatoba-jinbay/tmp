"""Evaluate LLM extraction results against XML ground truth."""

import json
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from kajima.parse_xml import parse_xml
from kajima.schema import BoringInfo


def _normalize(value: str) -> str:
    """Normalize a value for comparison."""
    return unicodedata.normalize("NFKC", value).strip()


def _flatten_model(model: BoringInfo) -> dict[str, str]:
    """Flatten a model into key-value pairs for comparison."""
    flat: dict[str, str] = {}
    header = model.header_info

    for section_name, section in [
        ("survey_basic_info", header.survey_basic_info),
        ("coordinate_info", header.coordinate_info),
        ("ordering_organization", header.ordering_organization),
        ("survey_period", header.survey_period),
        ("survey_company", header.survey_company),
        ("boring_basic_info", header.boring_basic_info),
    ]:
        for field_name, value in section.model_dump().items():
            key = f"header_info.{section_name}.{field_name}"
            flat[key] = str(value)

    core = model.core_info
    for field_name in type(core).model_fields:
        items = getattr(core, field_name)
        if not isinstance(items, list):
            continue
        for i, item in enumerate(items):
            if not isinstance(item, BaseModel):
                continue
            for k, v in item.model_dump().items():
                flat[f"core_info.{field_name}[{i}].{k}"] = str(v)

    return flat


def evaluate_single(
    xml_data: BoringInfo,
    llm_data: BoringInfo,
) -> dict:
    """Evaluate a single file.

    Only evaluates fields where XML has non-empty values,
    since not all XML content may be present in the PDF.
    """
    xml_flat = _flatten_model(xml_data)
    llm_flat = _flatten_model(llm_data)

    details: list[dict] = []
    matched = 0
    mismatched = 0
    missing_in_llm = 0
    non_empty_xml_fields = 0

    for key, xml_value in xml_flat.items():
        xml_norm = _normalize(xml_value)
        if not xml_norm:
            continue

        non_empty_xml_fields += 1
        llm_value = llm_flat.get(key, "")
        llm_norm = _normalize(llm_value)

        if not llm_norm:
            missing_in_llm += 1
            status = "missing"
        elif xml_norm == llm_norm:
            matched += 1
            status = "match"
        else:
            mismatched += 1
            status = "mismatch"

        details.append({
            "field": key,
            "status": status,
            "xml": xml_value,
            "llm": llm_value,
        })

    return {
        "total_xml_fields": len(xml_flat),
        "non_empty_xml_fields": non_empty_xml_fields,
        "matched": matched,
        "mismatched": mismatched,
        "missing_in_llm": missing_in_llm,
        "accuracy": (
            matched / non_empty_xml_fields
            if non_empty_xml_fields > 0
            else 0.0
        ),
        "details": details,
    }


def evaluate_batch(
    xml_dir: str | Path,
    result_dir: str | Path,
    llm: Literal["gemini", "claude"] = "gemini",
    output_path: str | Path | None = None,
) -> dict:
    """Run batch evaluation.

    Args:
        xml_dir: Directory containing XML files.
        result_dir: Directory containing LLM extraction results.
        llm: LLM name.
        output_path: Path to save evaluation results.

    Returns:
        Summary results.
    """
    xml_dir = Path(xml_dir)
    result_dir = Path(result_dir)

    all_results: list[dict] = []

    for result_file in sorted(result_dir.glob(f"*_{llm}.json")):
        stem = result_file.stem.replace(f"_{llm}", "")
        xml_file = xml_dir / f"{stem}.xml"

        try:
            xml_data = parse_xml(xml_file)
        except FileNotFoundError:
            print(
                f"XML not found for {result_file.name}, skipping"
            )
            continue

        with open(result_file, "r", encoding="utf-8") as f:
            llm_data = BoringInfo.model_validate(json.load(f))

        result = evaluate_single(xml_data, llm_data)
        result["file"] = stem
        all_results.append(result)

    total_non_empty = sum(
        r["non_empty_xml_fields"] for r in all_results
    )
    total_matched = sum(r["matched"] for r in all_results)

    summary = {
        "llm": llm,
        "total_files": len(all_results),
        "total_non_empty_fields": total_non_empty,
        "total_matched": total_matched,
        "total_mismatched": sum(
            r["mismatched"] for r in all_results
        ),
        "total_missing": sum(
            r["missing_in_llm"] for r in all_results
        ),
        "overall_accuracy": (
            total_matched / total_non_empty
            if total_non_empty > 0
            else 0.0
        ),
        "per_file": all_results,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Evaluation saved: {output_path}")

    print(f"\n=== Evaluation Summary ({llm}) ===")
    print(f"Files evaluated: {summary['total_files']}")
    print(
        f"Non-empty XML fields: {summary['total_non_empty_fields']}"
    )
    print(f"Matched: {summary['total_matched']}")
    print(f"Mismatched: {summary['total_mismatched']}")
    print(f"Missing in LLM: {summary['total_missing']}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.2%}")

    print("\nPer-file accuracy:")
    for r in all_results:
        print(
            f"  {r['file']}: {r['accuracy']:.2%} "
            f"({r['matched']}/{r['non_empty_xml_fields']})"
        )

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LLM extraction accuracy"
    )
    parser.add_argument(
        "--xml-dir",
        default="kajima/files/xml",
        help="XML directory",
    )
    parser.add_argument(
        "--result-dir",
        default="kajima_results",
        help="LLM result directory",
    )
    parser.add_argument(
        "--llm", choices=["gemini", "claude"], default="gemini"
    )
    parser.add_argument(
        "--output", default=None, help="Evaluation output path"
    )
    args = parser.parse_args()

    output = args.output or f"kajima_eval/{args.llm}_evaluation.json"
    evaluate_batch(
        args.xml_dir, args.result_dir, llm=args.llm, output_path=output
    )
