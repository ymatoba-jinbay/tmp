"""PDFから文字の重なりがないファイルを選定するスクリプト"""

import sys
from pathlib import Path

import fitz


def has_text_overlap(pdf_path: str, tolerance: float = 2.0) -> bool:
    """PDFに文字の重なりがあるかチェックする。

    各ページで文字単位のbboxを取得し、異なる文字同士でbboxが重なっている
    ケースを検出する。同じspan内の隣接文字は自然に隣り合うため除外する。
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return True

    with doc:
        for page in doc:
            try:
                flags = fitz.TEXT_PRESERVE_WHITESPACE
                text_dict = page.get_text("rawdict", flags=flags)
            except Exception:
                return True

            # 各spanのbboxを集める（行・ブロック情報付き）
            spans = []
            for block in text_dict.get("blocks", []):  # type: ignore[union-attr]
                if block.get("type") != 0:
                    continue
                block_no = block.get("number", 0)
                for li, line in enumerate(block.get("lines", [])):
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if not text.strip():
                            continue
                        bbox = span.get("bbox")
                        w = bbox[2] - bbox[0] if bbox else 0
                        h = bbox[3] - bbox[1] if bbox else 0
                        if bbox and w > 0.1 and h > 0.1:
                            spans.append({
                                "bbox": bbox,
                                "block": block_no,
                                "line": li,
                            })

            # y座標でソート
            spans.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))

            overlap_count = 0
            for i in range(len(spans)):
                bi = spans[i]["bbox"]
                for j in range(i + 1, len(spans)):
                    bj = spans[j]["bbox"]

                    # y方向が離れすぎたらbreak
                    if bj[1] > bi[3] + tolerance:
                        break

                    # 同じブロック・同じ行のspan同士はスキップ
                    same_block = spans[i]["block"] == spans[j]["block"]
                    same_line = spans[i]["line"] == spans[j]["line"]
                    if same_block and same_line:
                        continue

                    # y方向の重なり
                    y_overlap = min(bi[3], bj[3]) - max(bi[1], bj[1])
                    if y_overlap <= tolerance:
                        continue

                    # x方向の重なり
                    x_overlap = min(bi[2], bj[2]) - max(bi[0], bj[0])
                    if x_overlap > tolerance:
                        overlap_count += 1
                        if overlap_count >= 3:  # 3箇所以上の重なりで判定
                            return True

    return False


def main():
    pdf_dir = Path(__file__).parent / "files" / "pdf"
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    print(f"Total PDF files: {len(pdf_files)}", file=sys.stderr)

    no_overlap = []
    for i, pdf_path in enumerate(pdf_files):
        if i % 50 == 0:
            msg = f"Processing {i}/{len(pdf_files)}... found {len(no_overlap)} so far"
            print(msg, file=sys.stderr)

        if not has_text_overlap(str(pdf_path)):
            no_overlap.append(pdf_path.stem)

        if len(no_overlap) >= 100:
            break

    print(f"Found {len(no_overlap)} files without overlap", file=sys.stderr)

    for name in no_overlap:
        print(name)


if __name__ == "__main__":
    main()
