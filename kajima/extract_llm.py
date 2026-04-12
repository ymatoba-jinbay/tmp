"""Extract structured boring information from parsed text using LLM."""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from kajima.parse_xml import build_json_schema, parse_xml

load_dotenv()

FILES_DIR = Path(__file__).resolve().parent / "files"
XML_DIR = FILES_DIR / "xml"
PROMPTS_DIR = FILES_DIR / "prompts"

# --- プロンプト切り替え ---
# プロンプトファイル名（拡張子なし）。コマンドラインの --prompt で上書き可能。
PROMPT_NAME: str = "simple_schemaless"
# PROMPT_NAME: str = "common_instructions_schemaless"
# PROMPT_NAME: str = "balanced_schemaless"


def _load_prompt(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").rstrip("\n")


def _load_prompts() -> tuple[str, str, str]:
    """Load prompt templates based on current settings."""
    prompt_file = f"{PROMPT_NAME}.txt"
    common = _load_prompt(prompt_file)
    text_prefix = _load_prompt("text_prefix.txt")
    retry = _load_prompt("retry.txt")
    return common, text_prefix, retry


def _get_prompts() -> tuple[str, str, str]:
    """Get (EXTRACTION_PROMPT, PDF_EXTRACTION_PROMPT, RETRY_PROMPT)."""
    common, text_prefix, retry = _load_prompts()
    return text_prefix + "\n" + common, common, retry


MAX_RETRIES = 2

PARSE_TYPES = [
    "pdf",
    "jpg",
    "position",
    "position_spatial",
    "pymupdf4llm",
    "pymupdf",
    "odl_fast",
    "odl_hybrid",
]

LLM_CHOICES = [
    "gemini",
    "gemini-3-flash",
    "gemini-3.1-pro",
    "claude",
    "gpt5.4",
]

# LLM name -> (extract function name, model name override or None for default)
LLM_CONFIG: dict[str, tuple[str, str | None]] = {
    "gemini": ("gemini", None),  # default: gemini-2.5-pro
    "gemini-3-flash": ("gemini", "gemini-3-flash-preview"),
    "gemini-3.1-pro": ("gemini", "gemini-3.1-pro-preview"),
    "claude": ("claude", None),  # default: jp.anthropic.claude-sonnet-4-6
    "gpt5.4": ("openai", "gpt-5.4"),
}

_gemini_client = None
_claude_client = None
_openai_client = None


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

        _claude_client = anthropic.AnthropicBedrock(aws_region=region)
    return _claude_client


def get_openai_client():  # type: ignore[no-untyped-def]
    """Get or create a cached OpenAI client."""
    global _openai_client  # noqa: PLW0603
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


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
    retry_count: int = 0


def _resolve_schema(stem: str, xml_dir: Path) -> tuple[str, dict]:
    """Build a JSON Schema from the corresponding XML file.

    Returns:
        A tuple of (schema_text_for_prompt, json_schema_for_validation).
    """
    xml_path = xml_dir / f"{stem}.xml"
    if not xml_path.exists():
        msg = f"Corresponding XML not found: {xml_path}"
        raise FileNotFoundError(msg)
    data = parse_xml(xml_path)
    schema = build_json_schema(data)
    return json.dumps(schema, ensure_ascii=False, indent=2), schema


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


def _parse_and_validate(result_text: str, schema: dict) -> dict:
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
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
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
    """Resolve output directory.

    When PROMPT_NAME is not the default ("simple_schemaless"),
    the prompt name is appended to the parse_type folder name.
    """
    folder = parse_type
    if PROMPT_NAME != "simple_schemaless":
        folder = f"{parse_type}_{PROMPT_NAME}"
    return FILES_DIR / f"results_{llm}" / folder


def _load_test_filenames() -> list[str]:
    """Load ordered filenames from test_filenames.txt."""
    txt_path = FILES_DIR / "test_filenames.txt"
    if not txt_path.exists():
        return []
    return [line.strip() for line in txt_path.read_text().splitlines() if line.strip()]


def _list_input_files(input_dir: Path, parse_type: str) -> list[Path]:
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
        return [all_files[name] for name in test_names if name in all_files]
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
    extraction_prompt, pdf_prompt, retry_tmpl = _get_prompts()

    schema_text, schema = _resolve_schema(stem, xml_dir)

    if images is not None:
        prompt = pdf_prompt.format(schema=schema_text)
        base_parts = [
            types.Part.from_bytes(data=img, mime_type="image/jpeg") for img in images
        ] + [types.Part(text=prompt)]
    elif pdf_path is not None:
        prompt = pdf_prompt.format(schema=schema_text)
        base_parts = [
            types.Part.from_bytes(
                data=pdf_path.read_bytes(),
                mime_type="application/pdf",
            ),
            types.Part(text=prompt),
        ]
    else:
        base_parts = [
            types.Part(text=extraction_prompt.format(schema=schema_text, text=text))
        ]

    total_usage = TokenUsage()
    last_error = ""
    result_text = ""

    for attempt in range(1 + MAX_RETRIES):
        parts = list(base_parts)
        if attempt > 0:
            retry_prompt = retry_tmpl.format(errors=last_error, schema=schema_text)
            parts.extend(
                [
                    types.Part(text="Previous invalid JSON response:"),
                    types.Part(text=result_text),
                    types.Part(text=retry_prompt),
                ]
            )
            print(f"    Retry {attempt}/{MAX_RETRIES}")

        response = client.models.generate_content(
            model=model_name,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        if response.usage_metadata:
            total_usage.input_tokens += response.usage_metadata.prompt_token_count or 0
            total_usage.output_tokens += (
                response.usage_metadata.candidates_token_count or 0
            )

        result_text = (response.text or "").strip()
        try:
            data = _parse_and_validate(result_text, schema)
            return ExtractionResult(data=data, usage=total_usage, retry_count=attempt)
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
    model_name: str = "jp.anthropic.claude-sonnet-4-6",
    region: str = "ap-northeast-1",
    xml_dir: Path = XML_DIR,
) -> ExtractionResult:
    """Extract information using Claude via Bedrock."""
    import base64

    import anthropic

    client = get_claude_client(region)
    extraction_prompt, pdf_prompt, retry_tmpl = _get_prompts()

    schema_text, schema = _resolve_schema(stem, xml_dir)

    if images is not None:
        prompt = pdf_prompt.format(schema=schema_text)
        initial_content: list = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(img).decode("ascii"),
                },
            }
            for img in images
        ] + [{"type": "text", "text": prompt}]
    elif pdf_path is not None:
        prompt = pdf_prompt.format(schema=schema_text)
        pdf_b64 = base64.standard_b64encode(pdf_path.read_bytes()).decode("ascii")
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
                "text": extraction_prompt.format(schema=schema_text, text=text),
            }
        ]

    total_usage = TokenUsage()
    last_error = ""
    result_text = ""
    messages: list = [{"role": "user", "content": initial_content}]

    for attempt in range(1 + MAX_RETRIES):
        if attempt > 0:
            retry_prompt = retry_tmpl.format(errors=last_error, schema=schema_text)
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
            msg = f"Expected TextBlock, got {type(first_block).__name__}"
            raise TypeError(msg)
        result_text = _strip_markdown_fences(first_block.text.strip())

        try:
            data = _parse_and_validate(result_text, schema)
            return ExtractionResult(data=data, usage=total_usage, retry_count=attempt)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            if attempt == MAX_RETRIES:
                raise

    raise RuntimeError("Unreachable")


def extract_with_openai(
    stem: str,
    text: str | None = None,
    pdf_path: Path | None = None,
    images: list[bytes] | None = None,
    model_name: str = "gpt-5.4",
    xml_dir: Path = XML_DIR,
) -> ExtractionResult:
    """Extract information using OpenAI GPT."""
    import base64

    client = get_openai_client()
    extraction_prompt, pdf_prompt, retry_tmpl = _get_prompts()

    schema_text, schema = _resolve_schema(stem, xml_dir)

    if images is not None:
        prompt = pdf_prompt.format(schema=schema_text)
        initial_content: list = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.standard_b64encode(img).decode('ascii')}"
                },
            }
            for img in images
        ] + [{"type": "text", "text": prompt}]
    elif pdf_path is not None:
        # Convert PDF to images since not all OpenAI models accept PDFs directly.
        prompt = pdf_prompt.format(schema=schema_text)
        initial_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.standard_b64encode(img).decode('ascii')}"
                },
            }
            for img in _pdf_to_images(pdf_path)
        ] + [{"type": "text", "text": prompt}]
    else:
        initial_content = [
            {
                "type": "text",
                "text": extraction_prompt.format(schema=schema_text, text=text),
            }
        ]

    total_usage = TokenUsage()
    last_error = ""
    result_text = ""
    messages: list = [{"role": "user", "content": initial_content}]

    for attempt in range(1 + MAX_RETRIES):
        if attempt > 0:
            retry_prompt = retry_tmpl.format(errors=last_error, schema=schema_text)
            messages = [
                *messages,
                {"role": "assistant", "content": result_text},
                {"role": "user", "content": retry_prompt},
            ]
            print(f"    Retry {attempt}/{MAX_RETRIES}")

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},
            temperature=0,
        )

        if response.usage:
            total_usage.input_tokens += response.usage.prompt_tokens
            total_usage.output_tokens += response.usage.completion_tokens

        result_text = _strip_markdown_fences(
            (response.choices[0].message.content or "").strip()
        )

        try:
            data = _parse_and_validate(result_text, schema)
            return ExtractionResult(data=data, usage=total_usage, retry_count=attempt)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            if attempt == MAX_RETRIES:
                raise

    raise RuntimeError("Unreachable")


def process_file(
    file_path: Path,
    llm: str,
    output_dir: Path,
    xml_dir: Path = XML_DIR,
    parse_type: str = "pdf",
) -> ExtractionResult | None:
    """Extract boring info from a single file.

    Args:
        file_path: Path to the input file (PDF or parsed text).
        llm: LLM to use (key in LLM_CONFIG).
        output_dir: Directory to save results.
        xml_dir: Directory containing XML files for schema generation.
        parse_type: Input parse type.

    Returns:
        Extraction result with data and token usage, or None on failure.
    """
    stem = file_path.stem
    is_pdf = file_path.suffix.lower() == ".pdf"

    backend, model_override = LLM_CONFIG[llm]
    if backend == "gemini":
        extract_fn = extract_with_gemini
    elif backend == "claude":
        extract_fn = extract_with_claude
    else:
        extract_fn = extract_with_openai
    extra_kwargs: dict = {}
    if model_override:
        extra_kwargs["model_name"] = model_override
    start_time = time.monotonic()

    try:
        if is_pdf and parse_type == "jpg":
            result = extract_fn(
                stem, images=_pdf_to_images(file_path), xml_dir=xml_dir, **extra_kwargs
            )
        elif is_pdf:
            result = extract_fn(
                stem, pdf_path=file_path, xml_dir=xml_dir, **extra_kwargs
            )
        else:
            text = file_path.read_text(encoding="utf-8")
            result = extract_fn(stem, text=text, xml_dir=xml_dir, **extra_kwargs)
    except Exception as e:
        elapsed = time.monotonic() - start_time
        print(f"  Failed ({elapsed:.1f}s): {e}")
        _save_error(output_dir, stem, str(e))
        return None

    result.elapsed_seconds = time.monotonic() - start_time

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.json"
    output_data = {
        "_metadata": {"retry_count": result.retry_count},
        **result.data,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {output_path}")
    if result.retry_count > 0:
        print(f"  Retries: {result.retry_count}")

    return result


def _save_error(output_dir: Path, stem: str, error: str) -> None:
    """Append a failed extraction record to errors.jsonl in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"
    with open(errors_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"file": stem, "error": error}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract boring info using LLM")
    parser.add_argument(
        "--parse-type",
        choices=PARSE_TYPES,
        default="pdf",
        help="Input type (default: pdf)",
    )
    parser.add_argument(
        "--llm",
        choices=LLM_CHOICES,
        default="gemini",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of files to skip (default: 0)",
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
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt file name without .txt (e.g. simple_schemaless, balanced_schemaless)",
    )
    args = parser.parse_args()

    if args.prompt:
        PROMPT_NAME = args.prompt

    xml_dir = Path(args.xml_dir) if args.xml_dir else XML_DIR

    input_dir = _resolve_input_dir(args.parse_type)
    output_dir = _resolve_output_dir(args.parse_type, args.llm)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        raise SystemExit(1)

    input_files = _list_input_files(input_dir, args.parse_type)
    if args.offset > 0:
        input_files = input_files[args.offset :]
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
        result = process_file(
            f,
            llm=args.llm,
            output_dir=output_dir,
            xml_dir=xml_dir,
            parse_type=args.parse_type,
        )
        if result is not None:
            total_input += result.usage.input_tokens
            total_output += result.usage.output_tokens
            total_elapsed += result.elapsed_seconds
            print(f"  Time: {result.elapsed_seconds:.1f}s")

    print("\n=== Token Usage ===")
    print(f"Input tokens:  {total_input:,}")
    print(f"Output tokens: {total_output:,}")
    print(f"Total tokens:  {total_input + total_output:,}")
    print("\n=== Elapsed Time ===")
    print(f"Total: {total_elapsed:.1f}s")

    # Explicitly delete cached clients to avoid ImportError at shutdown
    del _gemini_client, _claude_client, _openai_client
