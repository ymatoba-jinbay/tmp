"""Extract text from PDF and save to file."""

import re
import unicodedata
from enum import Enum
from pathlib import Path
from typing import Any, cast

import pdfplumber

FILES_DIR = Path(__file__).resolve().parent / "files"


class ExtractionType(str, Enum):
    """PDF text extraction method."""

    POSITION = "position"
    PYMUPDF4LLM = "pymupdf4llm"
    HTML = "html"
    PYMUPDF = "pymupdf"


def _flush_vertical_group(
    current: list[dict],
    vertical_words: list[dict],
    remaining_singles: list[dict],
) -> None:
    """Flush a group of single-char words into vertical or remaining."""
    if len(current) > 1:
        vertical_words.append(
            {
                "text": "".join(c["text"] for c in current),
                "x0": min(c["x0"] for c in current),
                "top": current[0]["top"],
                "x1": max(c["x1"] for c in current),
                "bottom": current[-1]["bottom"],
                "direction": "vertical",
            }
        )
    else:
        remaining_singles.append(current[0])


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
                _flush_vertical_group(current, vertical_words, remaining_singles)
                current = [w]
        _flush_vertical_group(current, vertical_words, remaining_singles)

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
                lines.append(f"[x={w['x0']:.0f},y={w['top']:.0f}] {w['text']}")
        return "\n".join(lines)


def _build_spatial_text(position_text: str, page_width: float = 595.0) -> str:
    """座標情報付きテキストから、座標に基づいて空間配置したテキストを生成する。

    x座標を文字カラム位置に変換し、y座標でソートして行ごとにテキストを配置する。
    """

    # x座標→カラム位置の変換倍率（PDF座標をテキスト幅に変換）
    col_scale = 120.0 / page_width  # 120文字幅に収める
    y_line_height = 8.0  # この間隔以内のy座標は同じ行とみなす

    pages_output = []
    current_entries: list[tuple[float, float, str]] = []
    current_page_header = ""

    for line in position_text.split("\n"):
        if line.startswith("=== Page"):
            if current_entries:
                pages_output.append(
                    _render_spatial_page(
                        current_page_header,
                        current_entries,
                        col_scale,
                        y_line_height,
                    )
                )
                current_entries = []
            current_page_header = line
            continue

        m = re.match(r"\[x=(\d+),y=(\d+)\]\s(.*)", line)
        if m:
            x = float(m.group(1))
            y = float(m.group(2))
            text = m.group(3)
            current_entries.append((x, y, text))

    if current_entries:
        pages_output.append(
            _render_spatial_page(
                current_page_header,
                current_entries,
                col_scale,
                y_line_height,
            )
        )

    return "\n".join(pages_output)


def _render_spatial_page(
    header: str,
    entries: list[tuple[float, float, str]],
    col_scale: float,
    y_line_height: float,
) -> str:
    """1ページ分のエントリを空間配置してテキストに変換する。"""
    entries.sort(key=lambda e: (e[1], e[0]))

    lines = [header]
    row_entries: list[tuple[float, str]] = []
    current_y = entries[0][1] if entries else 0

    for x, y, text in entries:
        if y - current_y > y_line_height and row_entries:
            lines.append(_render_row(row_entries, col_scale))
            row_entries = []
            current_y = y
        row_entries.append((x, text))

    if row_entries:
        lines.append(_render_row(row_entries, col_scale))

    return "\n".join(lines)


def _render_row(
    entries: list[tuple[float, str]],
    col_scale: float,
) -> str:
    """同じ行のエントリをx座標に基づいてスペースで配置する。"""
    entries.sort(key=lambda e: e[0])
    buf = []
    current_col = 0
    for x, text in entries:
        target_col = int(x * col_scale)
        if target_col > current_col:
            buf.append(" " * (target_col - current_col))
        buf.append(text)
        text_width = sum(
            2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in text
        )
        current_col = max(target_col + text_width, current_col + text_width)
    return "".join(buf)


def _extract_with_markdown(pdf_path: Path) -> str:
    """Extract markdown with table structure using pymupdf4llm."""
    import pymupdf4llm

    result = pymupdf4llm.to_markdown(str(pdf_path))
    assert isinstance(result, str)
    return result


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Convert a table (list of rows) to markdown table string."""
    if not table:
        return ""
    rows = []
    for row in table:
        cells = [str(c).replace("\n", " ") if c else "" for c in row]
        rows.append("| " + " | ".join(cells) + " |")
    if len(rows) >= 1:
        sep = "| " + " | ".join("---" for _ in table[0]) + " |"
        rows.insert(1, sep)
    return "\n".join(rows)


def _extract_with_pymupdf(pdf_path: Path) -> str:
    """Extract text and tables from PDF using pymupdf (no OCR)."""
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    pages = []
    for page in doc:
        tabs = page.find_tables()
        tables = [] if tabs is None else list(tabs.tables)
        table_rects = [t.bbox for t in tables]

        # テーブル領域外のテキストを抽出
        non_table_blocks = []
        text_dict = cast(dict[str, Any], page.get_text("dict"))
        blocks = cast(list[dict[str, Any]], text_dict.get("blocks", []))
        for block in blocks:
            if block["type"] != 0:
                continue
            bx0, by0, bx1, by1 = block["bbox"]
            in_table = any(
                bx0 >= tx0 - 1 and by0 >= ty0 - 1 and bx1 <= tx1 + 1 and by1 <= ty1 + 1
                for tx0, ty0, tx1, ty1 in table_rects
            )
            if not in_table:
                lines_text = []
                for line in cast(list[dict[str, Any]], block.get("lines", [])):
                    spans_text = "".join(
                        span["text"]
                        for span in cast(
                            list[dict[str, Any]],
                            line.get("spans", []),
                        )
                    )
                    if spans_text.strip():
                        lines_text.append(spans_text)
                if lines_text:
                    non_table_blocks.append(
                        {
                            "y": by0,
                            "content": "\n".join(lines_text),
                        }
                    )

        # テーブルをmarkdown化
        table_blocks = []
        for t, rect in zip(tables, table_rects):
            md = _table_to_markdown(t.extract())
            if md.strip():
                table_blocks.append(
                    {
                        "y": rect[1],
                        "content": "\n" + md + "\n",
                    }
                )

        # y座標順にソートして結合
        all_blocks = non_table_blocks + table_blocks
        all_blocks.sort(key=lambda b: b["y"])
        page_text = "\n".join(b["content"] for b in all_blocks)
        if page_text.strip():
            pages.append(page_text)

    doc.close()
    return "\n\n".join(pages)


def _convert_markdown_to_html(md_path: Path) -> str:
    """Convert a parsed markdown file to HTML."""
    import markdown

    md_text = md_path.read_text(encoding="utf-8")
    return markdown.markdown(md_text, extensions=["tables"])


def extract_text_from_pdf(
    pdf_path: str | Path,
    extraction_type: ExtractionType = ExtractionType.PYMUPDF4LLM,
) -> str:
    """Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        extraction_type: Extraction method to use.

    Returns:
        Extracted text.
    """
    pdf_path = Path(pdf_path)

    match extraction_type:
        case ExtractionType.POSITION:
            return _extract_with_position(pdf_path)
        case ExtractionType.PYMUPDF4LLM:
            return _extract_with_markdown(pdf_path)
        case ExtractionType.HTML:
            raise ValueError(
                "HTML type converts from parsed markdown. Use parse_and_save() instead."
            )
        case ExtractionType.PYMUPDF:
            return _extract_with_pymupdf(pdf_path)


_SUFFIX_MAP: dict[ExtractionType, str] = {
    ExtractionType.POSITION: ".txt",
    ExtractionType.PYMUPDF4LLM: ".md",
    ExtractionType.HTML: ".html",
    ExtractionType.PYMUPDF: ".md",
}


def parse_and_save(
    pdf_path: str | Path,
    extraction_type: ExtractionType = ExtractionType.PYMUPDF4LLM,
    output_dir: str | Path | None = None,
) -> Path:
    """Extract text from PDF and save to file.

    Args:
        pdf_path: Path to the PDF file.
        extraction_type: Extraction method to use.
        output_dir: Directory to save results.
            Defaults to kajima/files/parsed/<extraction_type>.

    Returns:
        Path to the saved file.
    """
    pdf_path = Path(pdf_path)
    if output_dir is None:
        output_dir = FILES_DIR / "parsed" / extraction_type.value
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if extraction_type == ExtractionType.HTML:
        base_parsed = FILES_DIR / "parsed"
        md_sources = {
            "pymupdf": base_parsed / "pymupdf",
            "pymupdf4llm": base_parsed / "pymupdf4llm",
        }
        saved_paths = []
        for source_name, source_dir in md_sources.items():
            md_path = source_dir / f"{pdf_path.stem}.md"
            if not md_path.exists():
                continue
            out_dir = base_parsed / f"{source_name}_html"
            out_dir.mkdir(parents=True, exist_ok=True)
            html = _convert_markdown_to_html(md_path)
            out = out_dir / f"{pdf_path.stem}.html"
            out.write_text(html, encoding="utf-8")
            print(f"Saved: {out}")
            saved_paths.append(out)
        if not saved_paths:
            raise FileNotFoundError(
                f"No markdown files found for {pdf_path.stem}. "
                "Run pymupdf/pymupdf4llm extraction first."
            )
        return saved_paths[0]
    else:
        text = extract_text_from_pdf(pdf_path, extraction_type)

    suffix = _SUFFIX_MAP[extraction_type]
    output_path = output_dir / f"{pdf_path.stem}{suffix}"
    output_path.write_text(text, encoding="utf-8")
    print(f"Saved: {output_path}")

    if extraction_type == ExtractionType.POSITION:
        spatial_dir = output_dir.parent / "position_spatial"
        spatial_dir.mkdir(parents=True, exist_ok=True)
        spatial_text = _build_spatial_text(text)
        spatial_path = spatial_dir / f"{pdf_path.stem}.txt"
        spatial_path.write_text(spatial_text, encoding="utf-8")
        print(f"Saved: {spatial_path}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from boring PDF")
    parser.add_argument("pdf_path", help="Path to PDF file or directory")
    parser.add_argument(
        "--extraction-type",
        choices=[e.value for e in ExtractionType],
        default=ExtractionType.PYMUPDF4LLM.value,
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

    if pdf_path.is_dir():
        pdf_files = sorted(pdf_path.glob("*.pdf"))
        if args.limit > 0:
            pdf_files = pdf_files[: args.limit]
        for i, pf in enumerate(pdf_files):
            print(f"[{i + 1}/{len(pdf_files)}] Processing: {pf.name}")
            try:
                parse_and_save(
                    pf,
                    extraction_type=extraction_type,
                    output_dir=args.output_dir,
                )
            except Exception as e:
                print(f"  Error: {e}")
    else:
        parse_and_save(
            pdf_path,
            extraction_type=extraction_type,
            output_dir=args.output_dir,
        )
