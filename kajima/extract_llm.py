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

EXTRACTION_PROMPT = """\
以下はボーリング柱状図PDFから抽出されたテキストです。
このテキストから以下のJSON形式で情報を抽出してください。
値が読み取れない場合は空文字""にしてください。
数値はそのまま文字列として返してください。

出力するJSONスキーマ:
{schema}

テキスト:
{text}

JSONのみを出力してください。説明は不要です。"""

PDF_EXTRACTION_PROMPT = """\
添付のボーリング柱状図PDFから以下のJSON形式で情報を抽出してください。
値が読み取れない場合は空文字""にしてください。
数値はそのまま文字列として返してください。

出力するJSONスキーマ:
{schema}

JSONのみを出力してください。説明は不要です。"""

RETRY_PROMPT = """\
前回の出力に以下のエラーがありました。修正して再度JSONのみを出力してください。

エラー:
{errors}

出力するJSONスキーマ:
{schema}

JSONのみを出力してください。"""

FILES_DIR = Path(__file__).resolve().parent / "files"
XML_DIR = FILES_DIR / "xml"

PARSE_TYPES = ["pdf", "position", "pymupdf4llm", "html", "pymupdf"]
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
    region: str = "us-east-1",
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


def _resolve_input_dir(parse_type: str) -> Path:
    """Resolve input directory from parse_type."""
    if parse_type == "pdf":
        return FILES_DIR / "pdf"
    return FILES_DIR / "parsed" / parse_type


def _resolve_output_dir(parse_type: str, llm: str) -> Path:
    """Resolve output directory."""
    return FILES_DIR / f"results_{llm}" / parse_type


def _list_input_files(
    input_dir: Path, parse_type: str
) -> list[Path]:
    """List input files for the given parse_type."""
    if parse_type == "pdf":
        return sorted(input_dir.glob("*.pdf"))
    return sorted(
        list(input_dir.glob("*.md"))
        + list(input_dir.glob("*.html"))
        + list(input_dir.glob("*.txt"))
    )


def extract_with_gemini(
    stem: str,
    text: str | None = None,
    pdf_path: Path | None = None,
    model_name: str = "gemini-2.5-pro",
    xml_dir: Path = XML_DIR,
) -> ExtractionResult:
    """Extract information using Gemini via VertexAI."""
    from google.genai import types

    client = get_gemini_client()

    schema_json = _resolve_schema(stem, xml_dir)
    schema = json.loads(schema_json)

    if pdf_path is not None:
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
    model_name: str = "anthropic.claude-sonnet-4-20250514-v1:0",
    region: str = "us-east-1",
    xml_dir: Path = XML_DIR,
) -> ExtractionResult:
    """Extract information using Claude via Bedrock."""
    import base64

    import anthropic

    client = get_claude_client(region)

    schema_json = _resolve_schema(stem, xml_dir)
    schema = json.loads(schema_json)

    if pdf_path is not None:
        prompt = PDF_EXTRACTION_PROMPT.format(schema=schema_json)
        pdf_b64 = base64.standard_b64encode(
            pdf_path.read_bytes()
        ).decode("ascii")
        initial_content: list = [
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
) -> ExtractionResult:
    """Extract boring info from a single file.

    Args:
        file_path: Path to the input file (PDF or parsed text).
        llm: LLM to use.
        output_dir: Directory to save results.
        xml_dir: Directory containing XML files for schema generation.

    Returns:
        Extraction result with data and token usage.
    """
    stem = file_path.stem
    is_pdf = file_path.suffix.lower() == ".pdf"

    kwargs: dict = {"xml_dir": xml_dir}

    start_time = time.monotonic()

    if is_pdf:
        if llm == "gemini":
            result = extract_with_gemini(
                stem, pdf_path=file_path, **kwargs
            )
        else:
            result = extract_with_claude(
                stem, pdf_path=file_path, **kwargs
            )
    else:
        text = file_path.read_text(encoding="utf-8")
        if llm == "gemini":
            result = extract_with_gemini(
                stem, text=text, **kwargs
            )
        else:
            result = extract_with_claude(
                stem, text=text, **kwargs
            )

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
    print(f"\n=== Elapsed Time ===")
    print(f"Total: {total_elapsed:.1f}s")

    # Explicitly delete cached clients to avoid ImportError at shutdown
    del _gemini_client, _claude_client
