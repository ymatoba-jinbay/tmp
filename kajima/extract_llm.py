"""Extract structured boring information from parsed text using LLM."""

import json
import os
from pathlib import Path
from typing import Literal

from kajima.schema import BoringInfo

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

_SCHEMA_JSON = json.dumps(
    BoringInfo.model_json_schema(),
    ensure_ascii=False,
    indent=2,
)


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


def _gemini_client(client=None):  # type: ignore[no-untyped-def]
    """Get or create a Gemini client."""
    if client is not None:
        return client
    from google import genai

    return genai.Client(
        vertexai=True,
        project=os.environ["PROJECT_ID"],
        location=os.environ.get("LOCATION", "us-central1"),
    )


def extract_with_gemini(
    text: str | None = None,
    pdf_path: Path | None = None,
    model_name: str = "gemini-2.5-flash",
    client=None,  # type: ignore[no-untyped-def]
) -> BoringInfo:
    """Extract information using Gemini via VertexAI."""
    from google.genai import types

    client = _gemini_client(client)

    if pdf_path is not None:
        prompt = PDF_EXTRACTION_PROMPT.format(schema=_SCHEMA_JSON)
        parts = [
            types.Part.from_bytes(
                data=pdf_path.read_bytes(),
                mime_type="application/pdf",
            ),
            types.Part(text=prompt),
        ]
    else:
        prompt = EXTRACTION_PROMPT.format(
            schema=_SCHEMA_JSON, text=text
        )
        parts = [types.Part(text=prompt)]

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

    result_text = (response.text or "").strip()
    return BoringInfo.model_validate_json(result_text)


def _claude_client(client=None, region: str = "us-east-1"):  # type: ignore[no-untyped-def]
    """Get or create a Claude client."""
    if client is not None:
        return client
    import anthropic

    return anthropic.AnthropicBedrock(aws_region=region)


def _parse_claude_response(response) -> BoringInfo:  # type: ignore[no-untyped-def]
    """Parse Claude response into BoringInfo."""
    import anthropic

    first_block = response.content[0]
    if not isinstance(first_block, anthropic.types.TextBlock):
        msg = (
            f"Expected TextBlock, got {type(first_block).__name__}"
        )
        raise TypeError(msg)
    result_text = _strip_markdown_fences(first_block.text.strip())
    return BoringInfo.model_validate_json(result_text)


def extract_with_claude(
    text: str | None = None,
    pdf_path: Path | None = None,
    model_name: str = "anthropic.claude-sonnet-4-20250514-v1:0",
    region: str = "us-east-1",
    client=None,  # type: ignore[no-untyped-def]
) -> BoringInfo:
    """Extract information using Claude via Bedrock."""
    import base64

    client = _claude_client(client, region)

    if pdf_path is not None:
        prompt = PDF_EXTRACTION_PROMPT.format(schema=_SCHEMA_JSON)
        pdf_b64 = base64.standard_b64encode(
            pdf_path.read_bytes()
        ).decode("ascii")
        content: str | list = [
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
        content = EXTRACTION_PROMPT.format(
            schema=_SCHEMA_JSON, text=text
        )

    response = client.messages.create(
        model=model_name,
        max_tokens=8192,
        messages=[{"role": "user", "content": content}],
    )
    return _parse_claude_response(response)


def process_text(
    text_path: str | Path,
    llm: Literal["gemini", "claude"] = "gemini",
    output_dir: str | Path = "kajima_results",
    client=None,  # type: ignore[no-untyped-def]
) -> BoringInfo:
    """Read parsed text (or PDF) and extract boring info with LLM.

    Args:
        text_path: Path to the parsed text file or PDF.
        llm: LLM to use.
        output_dir: Directory to save results.
        client: Pre-built LLM client (for batch reuse).

    Returns:
        Extracted boring information.
    """
    text_path = Path(text_path)
    is_pdf = text_path.suffix.lower() == ".pdf"

    if is_pdf:
        if llm == "gemini":
            result = extract_with_gemini(pdf_path=text_path, client=client)
        else:
            result = extract_with_claude(pdf_path=text_path, client=client)
    else:
        text = text_path.read_text(encoding="utf-8")
        if llm == "gemini":
            result = extract_with_gemini(text=text, client=client)
        else:
            result = extract_with_claude(text=text, client=client)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = text_path.stem
    output_path = output_dir / f"{stem}_{llm}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            result.model_dump(), f, ensure_ascii=False, indent=2
        )
    print(f"Saved: {output_path}")

    return result


def _create_client(llm: Literal["gemini", "claude"]):  # type: ignore[no-untyped-def]
    """Create an LLM client."""
    if llm == "gemini":
        return _gemini_client()
    else:
        return _claude_client()


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Extract boring info from parsed text using LLM"
    )
    parser.add_argument(
        "text_path",
        help="Path to parsed text file or directory",
    )
    parser.add_argument(
        "--llm", choices=["gemini", "claude"], default="gemini"
    )
    parser.add_argument(
        "--output-dir",
        default="kajima_results",
        help="Output directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max files to process (0=all)",
    )
    args = parser.parse_args()

    llm_client = _create_client(args.llm)
    text_path = Path(args.text_path)
    if text_path.is_dir():
        text_files = sorted(
            list(text_path.glob("*.md"))
            + list(text_path.glob("*.html"))
            + list(text_path.glob("*.txt"))
            + list(text_path.glob("*.pdf"))
        )
        if args.limit > 0:
            text_files = text_files[:args.limit]
        for i, tf in enumerate(text_files):
            print(
                f"[{i + 1}/{len(text_files)}] Processing: {tf.name}"
            )
            try:
                process_text(
                    tf,
                    llm=args.llm,
                    output_dir=args.output_dir,
                    client=llm_client,
                )
            except Exception as e:
                print(f"  Error: {e}")
    else:
        process_text(
            text_path,
            llm=args.llm,
            output_dir=args.output_dir,
            client=llm_client,
        )
