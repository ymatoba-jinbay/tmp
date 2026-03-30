"""Extract structured boring information from parsed text using LLM."""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from kajima.parse_xml import build_json_schema, parse_xml

load_dotenv()

_COMMON_INSTRUCTIONS = """\
# 役割
あなたは建設会社で働く地盤のプロフェッショナルで、ボーリング調査の結果の理解に優れています。
以下のcontextに従ってtaskを実行してください。
回答は日本語で出力してください。

# context
添付のデータは、ボーリング調査結果の柱状図です。
内容を確認し、指示に従って、柱状図の内容を解析し、出力形式はformatに記載の内容に限りjson形式で出力してください。
jsonの項目名は固定です。取得した値の間で矛盾がないか、論理的に検討してください。

# format
{schema}

# task
1.  柱状図の内容をすべてを確認する。
2.  添付されるデータは複数のフォーマットを持つ可能性があるため、以下のルールは柔軟に適用すること。
3.  特に間違えやすい内容と、フォーマット差を吸収するためのルールを以下に記載します。
4.  **最終出力前の必須確認事項**: JSON出力前に、「北緯」「東経」の値が以下の条件を満たしているか必ず確認してください：
    *   値の形式: `"XX° XX' XX.XXXX"` （度・分・秒の形式）
    *   余計な文字なし: 値の末尾に引用符（`"`）、アポストロフィ（`'`）、その他の記号が余計に付加されていないこと
    *   カンマの変換: 小数点にカンマ（`,`）が使用されている場合は、ピリオド（`.`）に変換済みであること

    *   **全般:**
        *   情報が記載されていない項目は、値に空文字列 `""` を設定してください。
        *   数値はそのまま文字列として返してください。

    *   **表題情報:**
        *   ヘッダー情報から、各項目名と一致または類似するラベルを探して値を取得します。
        *   **「事業・工事名」**: 「事業・工事名」と明確にラベル付けされた欄からのみテキストを取得してください。この欄が存在しない、または空欄である場合は、必ず空文字列 `""` を出力してください。隣接する「調査名」欄の値を流用してはいけません。
        *   **「ボーリング名」**: 「ボーリング名」または「ボーリングNo.」というラベルの欄から取得します。
        *   **「調査位置」**: 「調査位置」というラベルの欄から取得します。このラベルが見つからない場合は、空文字列 `""` を出力します。
        *   **「北緯」「東経」の重要な注意事項**:
            * 「北緯」「東経」欄から値を読み取る際は、**数値と記号のみ**を抽出し、余計な文字（引用符、スペース、その他の記号）は削除してください。
            * **正しい出力例**: `"XX° XX' XX.XXXX"` （引用符やその他の余計な文字なし）
            * **誤った出力例**: `"XX° XX' XX.XXXX\\""'''` （末尾に余計な引用符や文字が含まれている）
            * カンマ（,）が小数点として使用されている場合は、ピリオド（.）に変換してください。
            * 値の末尾に余計な引用符や文字列が付加されていないか、出力前に必ず確認してください。
        *   **「調査業者名」**: 「調査業者名」というラベルの欄から取得します。このラベルが見つからない場合は、ヘッダーやフッターに記載されている会社名を取得します。
        *   **その他項目（調査期間、孔口標高など）**も同様に、対応するラベルの欄から値を取得してください。

    *   **地層情報:**
        *   **「層開始深度」「層終了深度」**: 柱状図本体の左側にある深度を示す列（例：「深度(m)」）から、各土質区分の境界の値を正確に読み取ってください。
        *   **「土質区分」**: 主に「土質区分」または「土質名」という見出しの列から取得します。
        *   **「土質区分_eng」**: 取得した日本語の「土質区分」または「土質名」を地盤工学分野において適切な英語に翻訳して出力します。
               英訳の適切な単語としては例えば、「Gravel, Sands, Silts, Clays, Fills, Rock」などがあります。
        *   **「土質シンボル」**: 「土質区分」列内の括弧書き（例: `(VH2-S)`）から取得します。「土質区分」列が見つからない、または括弧書きがない場合は、空文字列 `""` を出力します。
        *   **「色調」**: 各土質区分の行と水平に並んでいる「色調」列から取得します。複数の色が記載されている場合（例：「黄灰~褐灰」）は、記載通りに結合します。
        *   **「記事」**: 各記述がどの地層に属するかを以下のロジックで厳密に判断し、対応する地層の「記事」に含めてください。
            1.  **明示的な深度情報に基づく判定:** 記事内に角括弧や`m付近`といった形式で深度が明示されている場合（例:`[3.50]`, `4m付近`）、その深度が含まれる地層（層開始深度 <= 記述内深度 <= 層終了深度）に割り当てます。境界値に該当する場合は、より浅い層に含めます。
            2.  **視覚的所属に基づく判定:** 上記で特定できない場合、その記事が水平方向にどの土質区分の行ブロックと視覚的に対応しているかを判断し、その地層に割り当てます。
            3.  **同一地層内の記述の結合:** 同一地層に割り当てられた複数の記述は、出現順に改行文字 `\\n` で結合し、単一の「記事」データとします。

    *   **標準貫入試験データ:**
        *   「標準貫入試験」という明確な表区画から抽出してください。
        *   **「試験開始深度」**: 試験セクション内の深度を示す列（例：「深度(m)」、「貫入深度(m)」）から取得します。
        *   **「N値」**: 対応する行の「N値」列から数値を取得します。`>`記号が付随している場合は、数値部分のみを取得します（例: `>50` -> `50`）。
        *   **「試験終了深度」**: `試験終了深度 = 試験開始深度 + (貫入量 / 100)` の式で算出します。`貫入量(cm)`は、以下の優先順位で決定してください。
            1.  「打撃回数/貫入量(cm)」という形式の列があれば、その分母の値を取得します。
            2.  上記がない場合で、「N値」列が分数形式（例: `45/30`）で記載されている場合は、その分母の値を取得します。

    *   **孔内水位測定記録:**
        *   **キーワード探索**: データ全体から**「孔内水位」という文字列**（完全に一致しなくても、この語句を含んでいれば可）を探します。
        *   **値の特定**: 上記で見つかった「孔内水位」という文字列の近くで、まずは「測定日」（例：`YYYY-MM-DD`または`MM/DD`形式）を特定し、測定日の表記直下、かつ同じ列に記載されている数値（「X.XX」形式（小数点以下2桁））を「孔内水位」として取得してください。
        *   **最終決定**:
            *   候補が複数見つかった場合は、専用の欄の見出しの直下にある値です。
            *   候補が1つしか見つからない場合は、その値を孔内水位として採用します。
        *   **除外ルール**: 「備考」欄などに記載されている**「泥水水位」**というキーワードに関連する数値は、孔内水位ではないため、常に除外します。

JSONのみを出力してください。説明は不要です。"""

EXTRACTION_PROMPT = """\
以下はボーリング柱状図PDFから抽出されたテキストです。

テキスト:
{text}

""" + _COMMON_INSTRUCTIONS

PDF_EXTRACTION_PROMPT = _COMMON_INSTRUCTIONS

RETRY_PROMPT = """\
前回の出力に以下のエラーがありました。修正して再度JSONのみを出力してください。

エラー:
{errors}

出力するJSONスキーマ:
{schema}

JSONのみを出力してください。"""

FILES_DIR = Path(__file__).resolve().parent / "files"
XML_DIR = FILES_DIR / "xml"

PARSE_TYPES = [
    "pdf", "jpg", "position", "position_spatial",
    "pymupdf4llm", "html", "pymupdf",
]
MAX_RETRIES = 2

_gemini_client = None
_claude_client = None


def get_gemini_client():  # type: ignore[no-untyped-def]
    """Get or create a cached Gemini client."""
    global _gemini_client  # noqa: PLW0603
    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client(
            vertexai=True,
            project=os.environ["PROJECT_ID"],
            location=os.environ.get("LOCATION", "us-central1"),
        )
    return _gemini_client


def get_claude_client(
    region: str = "ap-northeast-1",
):  # type: ignore[no-untyped-def]
    """Get or create a cached Claude client."""
    global _claude_client  # noqa: PLW0603
    if _claude_client is None:
        import anthropic

        _claude_client = anthropic.AnthropicBedrock(
            aws_region=region
        )
    return _claude_client


@dataclass
class TokenUsage:
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ExtractionResult:
    """Result of LLM extraction."""

    data: dict = field(default_factory=dict)
    usage: TokenUsage = field(default_factory=TokenUsage)
    elapsed_seconds: float = 0.0


def _resolve_schema(stem: str, xml_dir: Path) -> str:
    """Build a JSON schema string from the corresponding XML file."""
    xml_path = xml_dir / f"{stem}.xml"
    if not xml_path.exists():
        msg = f"Corresponding XML not found: {xml_path}"
        raise FileNotFoundError(msg)
    data = parse_xml(xml_path)
    schema = build_json_schema(data)
    return json.dumps(schema, ensure_ascii=False, indent=2)


def _validate_schema(data: dict, schema: dict) -> None:
    """Validate data against a JSON Schema.

    Raises ValueError with all validation errors on failure.
    """
    import jsonschema

    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(
        validator.iter_errors(data),
        key=lambda e: list(e.absolute_path),
    )
    if errors:
        messages = []
        for e in errors:
            path = ".".join(str(p) for p in e.absolute_path)
            loc = path or "root"
            messages.append(f"  - {loc}: {e.message}")
        msg = "Schema validation failed:\n" + "\n".join(messages)
        raise ValueError(msg)


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    json_lines = []
    in_block = False
    for line in lines:
        if line.startswith("```") and not in_block:
            in_block = True
            continue
        elif line.startswith("```") and in_block:
            break
        elif in_block:
            json_lines.append(line)
    return "\n".join(json_lines)


def _parse_and_validate(
    result_text: str, schema: dict
) -> dict:
    """Parse JSON text and validate against schema.

    Raises json.JSONDecodeError or ValueError on failure.
    """
    data = json.loads(result_text)
    _validate_schema(data, schema)
    return data


def _pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[bytes]:
    """Convert a PDF file to a list of JPEG image bytes."""
    import io

    import fitz
    from PIL import Image

    doc = fitz.open(pdf_path)
    images: list[bytes] = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes(
            "RGB", (pix.width, pix.height), pix.samples
        )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        images.append(buf.getvalue())
    doc.close()
    return images


def _resolve_input_dir(parse_type: str) -> Path:
    """Resolve input directory from parse_type."""
    if parse_type in ("pdf", "jpg"):
        return FILES_DIR / "pdf"
    return FILES_DIR / "parsed" / parse_type


def _resolve_output_dir(parse_type: str, llm: str) -> Path:
    """Resolve output directory."""
    return FILES_DIR / f"results_{llm}" / parse_type


def _load_test_filenames() -> list[str]:
    """Load ordered filenames from test_filenames.txt."""
    txt_path = FILES_DIR / "test_filenames.txt"
    if not txt_path.exists():
        return []
    return [
        line.strip()
        for line in txt_path.read_text().splitlines()
        if line.strip()
    ]


def _list_input_files(
    input_dir: Path, parse_type: str
) -> list[Path]:
    """List input files for the given parse_type.

    Files are filtered and ordered by test_filenames.txt.
    """
    test_names = _load_test_filenames()

    if parse_type in ("pdf", "jpg"):
        all_files = {f.stem: f for f in input_dir.glob("*.pdf")}
    else:
        all_files = {
            f.stem: f
            for f in (
                list(input_dir.glob("*.md"))
                + list(input_dir.glob("*.html"))
                + list(input_dir.glob("*.txt"))
            )
        }

    if test_names:
        return [
            all_files[name]
            for name in test_names
            if name in all_files
        ]
    return sorted(all_files.values(), key=lambda f: f.name)


def extract_with_gemini(
    stem: str,
    text: str | None = None,
    pdf_path: Path | None = None,
    images: list[bytes] | None = None,
    model_name: str = "gemini-2.5-pro",
    xml_dir: Path = XML_DIR,
) -> ExtractionResult:
    """Extract information using Gemini via VertexAI."""
    from google.genai import types

    client = get_gemini_client()

    schema_json = _resolve_schema(stem, xml_dir)
    schema = json.loads(schema_json)

    if images is not None:
        prompt = PDF_EXTRACTION_PROMPT.format(schema=schema_json)
        parts = [
            types.Part.from_bytes(
                data=img, mime_type="image/jpeg"
            )
            for img in images
        ] + [types.Part(text=prompt)]
    elif pdf_path is not None:
        prompt = PDF_EXTRACTION_PROMPT.format(schema=schema_json)
        parts = [
            types.Part.from_bytes(
                data=pdf_path.read_bytes(),
                mime_type="application/pdf",
            ),
            types.Part(text=prompt),
        ]
    else:
        prompt = EXTRACTION_PROMPT.format(
            schema=schema_json, text=text
        )
        parts = [types.Part(text=prompt)]

    total_usage = TokenUsage()
    last_error = ""

    for attempt in range(1 + MAX_RETRIES):
        if attempt > 0:
            retry_prompt = RETRY_PROMPT.format(
                errors=last_error, schema=schema_json
            )
            parts = [types.Part(text=retry_prompt)]
            print(f"    Retry {attempt}/{MAX_RETRIES}")

        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(role="user", parts=parts)
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        if response.usage_metadata:
            total_usage.input_tokens += (
                response.usage_metadata.prompt_token_count or 0
            )
            total_usage.output_tokens += (
                response.usage_metadata.candidates_token_count or 0
            )

        result_text = (response.text or "").strip()
        try:
            data = _parse_and_validate(result_text, schema)
            return ExtractionResult(
                data=data, usage=total_usage
            )
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            if attempt == MAX_RETRIES:
                raise

    raise RuntimeError("Unreachable")


def extract_with_claude(
    stem: str,
    text: str | None = None,
    pdf_path: Path | None = None,
    images: list[bytes] | None = None,
    model_name: str = "anthropic.claude-opus-4-0-20250514-v1:0",
    region: str = "ap-northeast-1",
    xml_dir: Path = XML_DIR,
) -> ExtractionResult:
    """Extract information using Claude via Bedrock."""
    import base64

    import anthropic

    client = get_claude_client(region)

    schema_json = _resolve_schema(stem, xml_dir)
    schema = json.loads(schema_json)

    if images is not None:
        prompt = PDF_EXTRACTION_PROMPT.format(schema=schema_json)
        initial_content: list = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(
                        img
                    ).decode("ascii"),
                },
            }
            for img in images
        ] + [{"type": "text", "text": prompt}]
    elif pdf_path is not None:
        prompt = PDF_EXTRACTION_PROMPT.format(schema=schema_json)
        pdf_b64 = base64.standard_b64encode(
            pdf_path.read_bytes()
        ).decode("ascii")
        initial_content = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_b64,
                },
            },
            {"type": "text", "text": prompt},
        ]
    else:
        initial_content = [
            {
                "type": "text",
                "text": EXTRACTION_PROMPT.format(
                    schema=schema_json, text=text
                ),
            }
        ]

    total_usage = TokenUsage()
    last_error = ""
    result_text = ""
    messages: list = [
        {"role": "user", "content": initial_content}
    ]

    for attempt in range(1 + MAX_RETRIES):
        if attempt > 0:
            retry_prompt = RETRY_PROMPT.format(
                errors=last_error, schema=schema_json
            )
            messages = [
                *messages,
                {"role": "assistant", "content": result_text},
                {"role": "user", "content": retry_prompt},
            ]
            print(f"    Retry {attempt}/{MAX_RETRIES}")

        response = client.messages.create(
            model=model_name,
            max_tokens=8192,
            messages=messages,  # type: ignore[arg-type]
        )

        total_usage.input_tokens += response.usage.input_tokens
        total_usage.output_tokens += response.usage.output_tokens

        first_block = response.content[0]
        if not isinstance(first_block, anthropic.types.TextBlock):
            msg = (
                f"Expected TextBlock, "
                f"got {type(first_block).__name__}"
            )
            raise TypeError(msg)
        result_text = _strip_markdown_fences(
            first_block.text.strip()
        )

        try:
            data = _parse_and_validate(result_text, schema)
            return ExtractionResult(
                data=data, usage=total_usage
            )
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            if attempt == MAX_RETRIES:
                raise

    raise RuntimeError("Unreachable")


def process_file(
    file_path: Path,
    llm: Literal["gemini", "claude"],
    output_dir: Path,
    xml_dir: Path = XML_DIR,
    parse_type: str = "pdf",
) -> ExtractionResult:
    """Extract boring info from a single file.

    Args:
        file_path: Path to the input file (PDF or parsed text).
        llm: LLM to use.
        output_dir: Directory to save results.
        xml_dir: Directory containing XML files for schema generation.
        parse_type: Input parse type.

    Returns:
        Extraction result with data and token usage.
    """
    stem = file_path.stem
    is_pdf = file_path.suffix.lower() == ".pdf"

    extract_fn = (
        extract_with_gemini if llm == "gemini"
        else extract_with_claude
    )
    kwargs: dict = {"xml_dir": xml_dir}

    start_time = time.monotonic()

    if is_pdf and parse_type == "jpg":
        result = extract_fn(
            stem, images=_pdf_to_images(file_path), **kwargs
        )
    elif is_pdf:
        result = extract_fn(
            stem, pdf_path=file_path, **kwargs
        )
    else:
        text = file_path.read_text(encoding="utf-8")
        result = extract_fn(stem, text=text, **kwargs)

    result.elapsed_seconds = time.monotonic() - start_time

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {output_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract boring info using LLM"
    )
    parser.add_argument(
        "--parse-type",
        choices=PARSE_TYPES,
        default="pdf",
        help="Input type (default: pdf)",
    )
    parser.add_argument(
        "--llm",
        choices=["gemini", "claude"],
        default="gemini",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max files to process (0=all)",
    )
    parser.add_argument(
        "--xml-dir",
        default=None,
        help="XML directory (default: kajima/files/xml)",
    )
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir) if args.xml_dir else XML_DIR

    input_dir = _resolve_input_dir(args.parse_type)
    output_dir = _resolve_output_dir(args.parse_type, args.llm)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        raise SystemExit(1)

    input_files = _list_input_files(input_dir, args.parse_type)
    if args.limit > 0:
        input_files = input_files[: args.limit]

    print(f"LLM: {args.llm}")
    print(f"Parse type: {args.parse_type}")
    print(f"Input: {input_dir} ({len(input_files)} files)")
    print(f"Output: {output_dir}")
    print()

    total_input = 0
    total_output = 0
    total_elapsed = 0.0

    for i, f in enumerate(input_files):
        print(f"[{i + 1}/{len(input_files)}] {f.name}")
        try:
            result = process_file(
                f,
                llm=args.llm,
                output_dir=output_dir,
                xml_dir=xml_dir,
                parse_type=args.parse_type,
            )
            total_input += result.usage.input_tokens
            total_output += result.usage.output_tokens
            total_elapsed += result.elapsed_seconds
            print(f"  Time: {result.elapsed_seconds:.1f}s")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n=== Token Usage ===")
    print(f"Input tokens:  {total_input:,}")
    print(f"Output tokens: {total_output:,}")
    print(f"Total tokens:  {total_input + total_output:,}")
    print("\n=== Elapsed Time ===")
    print(f"Total: {total_elapsed:.1f}s")

    # Explicitly delete cached clients to avoid ImportError at shutdown
    del _gemini_client, _claude_client
