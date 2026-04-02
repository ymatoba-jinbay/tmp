"""PDFから文字の重なりがないファイルを選定するスクリプト"""

import sys
from pathlib import Path

import fitz


def has_text_overlap(pdf_path: str) -> bool:
    """PDFに文字の重なりがあるかチェックする（文字単位）。

    異なる行の文字同士が実際に重なっているケースを検出する。
    完全な二重化（同じ文字が同じ位置に重複）は問題ないのでスキップする。
    表の隣接行で自然に近接している（重なりはない）ケースは許容する。
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

            # 文字単位でbboxを集める（ブロック・行情報付き）
            chars = []
            for block in text_dict.get("blocks", []):  # type: ignore[union-attr]
                if block.get("type") != 0:
                    continue
                block_no = block.get("number", 0)
                for li, line in enumerate(block.get("lines", [])):
                    for span in line.get("spans", []):
                        for ch in span.get("chars", []):
                            c = ch.get("c", "")
                            bbox = ch.get("bbox")
                            if not c.strip() or not bbox:
                                continue
                            w = bbox[2] - bbox[0]
                            h = bbox[3] - bbox[1]
                            if w > 0.1 and h > 0.1:
                                chars.append((c, bbox, block_no, li))

            # y座標でソート
            chars.sort(key=lambda s: (s[1][1], s[1][0]))

            for i in range(len(chars)):
                ci, bi, bni, lni = chars[i]
                for j in range(i + 1, len(chars)):
                    cj, bj, bnj, lnj = chars[j]

                    # y方向が離れたらbreak
                    if bj[1] > bi[3] + 0.5:
                        break

                    # 同じブロック・同じ行の文字同士はスキップ
                    if bni == bnj and lni == lnj:
                        continue

                    # x方向の重なりチェック
                    x_overlap = min(bi[2], bj[2]) - max(bi[0], bj[0])
                    if x_overlap <= 0:
                        continue

                    # y方向の重なりチェック
                    y_overlap = min(bi[3], bj[3]) - max(bi[1], bj[1])
                    if y_overlap <= 0:
                        continue

                    # 完全な二重化（同じ文字・同じ位置）はスキップ
                    if ci == cj:
                        dx = abs(bi[0] - bj[0]) + abs(bi[2] - bj[2])
                        dy = abs(bi[1] - bj[1]) + abs(bi[3] - bj[3])
                        if dx < 0.5 and dy < 0.5:
                            continue

                    # ここに到達 = 異なる行の文字が x, y 両方で重なっている
                    return True

    return False


def main() -> None:
    pdf_dir = Path(__file__).parent / "files" / "pdf"
    out_path = Path(__file__).parent / "files" / "test_filenames.txt"
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

    out_path.write_text("\n".join(no_overlap) + "\n")
    print(f"Written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
