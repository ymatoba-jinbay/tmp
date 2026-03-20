"""Extract text from PDF and save to file."""

from enum import Enum
from pathlib import Path

import pdfplumber


class ExtractionType(str, Enum):
    """PDF text extraction method."""

    TEXT = "text"
    POSITION = "position"
    MARKER_MARKDOWN = "marker_markdown"
    MARKER_HTML = "marker_html"
    PYMUPDF_MARKDOWN = "pymupdf_markdown"
    PYMUPDF_HTML = "pymupdf_html"


def _extract_with_text(pdf_path: Path) -> str:
    """Extract plain text from PDF using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        texts = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        return "\n".join(texts)


def _group_vertical_words(
    words: list[dict],
    x_tolerance: float = 3.0,
    y_gap_max: float = 5.0,
) -> list[dict]:
    """Group single-char words into vertical text strings."""
    from collections import defaultdict

    single_chars = []
    multi_chars = []
    for w in words:
        text = w["text"].strip()
        if len(text) == 1:
            single_chars.append(w)
        else:
            multi_chars.append(w)

    x_groups: dict[int, list[dict]] = defaultdict(list)
    for w in single_chars:
        x_key = round(w["x0"] / x_tolerance)
        x_groups[x_key].append(w)

    vertical_words = []
    remaining_singles = []
    for group in x_groups.values():
        group.sort(key=lambda w: w["top"])
        current = [group[0]]
        for w in group[1:]:
            prev = current[-1]
            if w["top"] - prev["bottom"] < y_gap_max:
                current.append(w)
            else:
                if len(current) > 1:
                    vertical_words.append({
                        "text": "".join(
                            c["text"] for c in current
                        ),
                        "x0": min(c["x0"] for c in current),
                        "top": current[0]["top"],
                        "x1": max(c["x1"] for c in current),
                        "bottom": current[-1]["bottom"],
                        "direction": "vertical",
                    })
                else:
                    remaining_singles.append(current[0])
                current = [w]
        if len(current) > 1:
            vertical_words.append({
                "text": "".join(c["text"] for c in current),
                "x0": min(c["x0"] for c in current),
                "top": current[0]["top"],
                "x1": max(c["x1"] for c in current),
                "bottom": current[-1]["bottom"],
                "direction": "vertical",
            })
        else:
            remaining_singles.append(current[0])

    return multi_chars + remaining_singles + vertical_words


def _extract_with_position(pdf_path: Path) -> str:
    """Extract text with word-level coordinates using pdfplumber.

    Groups characters into words (horizontal and vertical).
    """
    with pdfplumber.open(pdf_path) as pdf:
        lines = []
        for page_idx, page in enumerate(pdf.pages):
            lines.append(f"=== Page {page_idx + 1} ===")
            words = page.extract_words(
                keep_blank_chars=False,
                x_tolerance=2,
                y_tolerance=2,
            )
            grouped = _group_vertical_words(words)
            grouped.sort(key=lambda w: (w["top"], w["x0"]))
            for w in grouped:
                d = "V" if w.get("direction") == "vertical" else "H"
                lines.append(
                    f"[{d} "
                    f"x={w['x0']:.0f},"
                    f"y={w['top']:.0f}] "
                    f"{w['text']}"
                )
        return "\n".join(lines)


def _create_marker_converter(output_format: str):  # type: ignore[no-untyped-def]
    """Create a marker PdfConverter (expensive, should be reused)."""
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    config: dict = {
        "output_format": output_format,
        "disable_ocr": True,
    }
    config_parser = ConfigParser(config)

    return PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )


def _extract_with_marker(pdf_path: Path, converter) -> str:  # type: ignore[no-untyped-def]
    """Extract text using a pre-built marker converter."""
    from marker.output import text_from_rendered

    rendered = converter(str(pdf_path))
    text, _, _ = text_from_rendered(rendered)
    return text


def _extract_with_pymupdf_markdown(pdf_path: Path) -> str:
    """Extract markdown with table structure using pymupdf4llm."""
    import pymupdf4llm

    result = pymupdf4llm.to_markdown(str(pdf_path))
    assert isinstance(result, str)
    return result


def _extract_with_pymupdf_html(pdf_path: Path) -> str:
    """Extract HTML with position/font info using pymupdf."""
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text("html"))
    doc.close()
    return "\n".join(pages)


def extract_text_from_pdf(
    pdf_path: str | Path,
    extraction_type: ExtractionType = ExtractionType.MARKER_MARKDOWN,
    converter=None,  # type: ignore[no-untyped-def]
) -> str:
    """Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        extraction_type: Extraction method to use.
        converter: Pre-built marker converter (for batch reuse).

    Returns:
        Extracted text.
    """
    pdf_path = Path(pdf_path)

    match extraction_type:
        case ExtractionType.TEXT:
            return _extract_with_text(pdf_path)
        case ExtractionType.POSITION:
            return _extract_with_position(pdf_path)
        case ExtractionType.MARKER_MARKDOWN | ExtractionType.MARKER_HTML:
            if converter is None:
                fmt = (
                    "markdown"
                    if extraction_type == ExtractionType.MARKER_MARKDOWN
                    else "html"
                )
                converter = _create_marker_converter(fmt)
            return _extract_with_marker(pdf_path, converter)
        case ExtractionType.PYMUPDF_MARKDOWN:
            return _extract_with_pymupdf_markdown(pdf_path)
        case ExtractionType.PYMUPDF_HTML:
            return _extract_with_pymupdf_html(pdf_path)


_SUFFIX_MAP: dict[ExtractionType, str] = {
    ExtractionType.TEXT: ".txt",
    ExtractionType.POSITION: ".txt",
    ExtractionType.MARKER_MARKDOWN: ".md",
    ExtractionType.MARKER_HTML: ".html",
    ExtractionType.PYMUPDF_MARKDOWN: ".md",
    ExtractionType.PYMUPDF_HTML: ".html",
}


def parse_and_save(
    pdf_path: str | Path,
    extraction_type: ExtractionType = ExtractionType.MARKER_MARKDOWN,
    output_dir: str | Path | None = None,
    converter=None,  # type: ignore[no-untyped-def]
) -> Path:
    """Extract text from PDF and save to file.

    Args:
        pdf_path: Path to the PDF file.
        extraction_type: Extraction method to use.
        output_dir: Directory to save results.
            Defaults to kajima/files/parsed/<extraction_type>.
        converter: Pre-built marker converter (for batch reuse).

    Returns:
        Path to the saved file.
    """
    pdf_path = Path(pdf_path)
    if output_dir is None:
        output_dir = (
            Path("kajima/files/parsed") / extraction_type.value
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = extract_text_from_pdf(
        pdf_path, extraction_type, converter=converter
    )

    suffix = _SUFFIX_MAP[extraction_type]
    output_path = output_dir / f"{pdf_path.stem}{suffix}"
    output_path.write_text(text, encoding="utf-8")
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract text from boring PDF"
    )
    parser.add_argument(
        "pdf_path", help="Path to PDF file or directory"
    )
    parser.add_argument(
        "--extraction-type",
        choices=[e.value for e in ExtractionType],
        default=ExtractionType.MARKER_MARKDOWN.value,
        help="PDF text extraction method",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: kajima/files/parsed/<type>)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max PDFs to process (0=all)",
    )
    args = parser.parse_args()

    extraction_type = ExtractionType(args.extraction_type)
    pdf_path = Path(args.pdf_path)

    # Build converter once for marker-based extraction
    marker_converter = None
    if extraction_type in (
        ExtractionType.MARKER_MARKDOWN,
        ExtractionType.MARKER_HTML,
    ):
        fmt = (
            "markdown"
            if extraction_type == ExtractionType.MARKER_MARKDOWN
            else "html"
        )
        marker_converter = _create_marker_converter(fmt)

    if pdf_path.is_dir():
        pdf_files = sorted(pdf_path.glob("*.pdf"))
        if args.limit > 0:
            pdf_files = pdf_files[:args.limit]
        for i, pf in enumerate(pdf_files):
            print(
                f"[{i + 1}/{len(pdf_files)}] Processing: {pf.name}"
            )
            try:
                parse_and_save(
                    pf,
                    extraction_type=extraction_type,
                    output_dir=args.output_dir,
                    converter=marker_converter,
                )
            except Exception as e:
                print(f"  Error: {e}")
    else:
        parse_and_save(
            pdf_path,
            extraction_type=extraction_type,
            output_dir=args.output_dir,
            converter=marker_converter,
        )
