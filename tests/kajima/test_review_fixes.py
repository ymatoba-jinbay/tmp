import unittest
from pathlib import Path
from unittest.mock import patch

from kajima.evaluate import _match_arrays, evaluate_single
from kajima.extract_llm import extract_with_gemini
from kajima.parse_pdf import ExtractionType, parse_and_save


class EvaluateFixesTest(unittest.TestCase):
    def test_match_arrays_skips_zero_overlap(self) -> None:
        matches = _match_arrays(
            [{"a": "1"}],
            [{"a": "2"}],
        )

        self.assertEqual(matches, [(0, None)])

    def test_evaluate_single_counts_unexpected_llm_fields(self) -> None:
        result = evaluate_single(
            {"a": ""},
            {"a": "", "b": "x"},
            {"a", "b"},
        )

        self.assertEqual(result["incorrect"], 1)
        self.assertEqual(result["details"][0]["field"], "b")
        self.assertEqual(
            result["details"][0]["error_type"],
            "false_positive",
        )


class GeminiRetryTest(unittest.TestCase):
    def test_retry_keeps_original_input_parts(self) -> None:
        class FakeUsageMetadata:
            prompt_token_count = 1
            candidates_token_count = 1

        class FakeResponse:
            def __init__(self, text: str) -> None:
                self.text = text
                self.usage_metadata = FakeUsageMetadata()

        class FakeClient:
            def __init__(self) -> None:
                self.calls: list[list[object]] = []
                self.models = self

            def generate_content(self, *, contents, **kwargs):  # type: ignore[no-untyped-def]
                self.calls.append(contents[0].parts)
                if len(self.calls) == 1:
                    return FakeResponse("not json")
                return FakeResponse('{"ok": true}')

        fake_client = FakeClient()

        with (
            patch("kajima.extract_llm.get_gemini_client", return_value=fake_client),
            patch(
                "kajima.extract_llm._get_prompts",
                return_value=(
                    "schema={schema}\ntext={text}",
                    "pdf={schema}",
                    "retry {errors} {schema}",
                ),
            ),
            patch(
                "kajima.extract_llm._resolve_schema",
                return_value=("{schema}", None),
            ),
        ):
            result = extract_with_gemini("sample", text="source text")

        self.assertEqual(result.data, {"ok": True})
        self.assertEqual(len(fake_client.calls), 2)
        self.assertGreater(len(fake_client.calls[1]), len(fake_client.calls[0]))


class ParsePdfPathTest(unittest.TestCase):
    def test_parse_and_save_uses_module_files_dir_by_default(self) -> None:
        expected_output = Path("/tmp/kajima-files/parsed/pymupdf4llm/sample.md")

        with (
            patch("kajima.parse_pdf.FILES_DIR", Path("/tmp/kajima-files")),
            patch(
                "pathlib.Path.mkdir",
                autospec=True,
            ) as mkdir_mock,
            patch(
                "kajima.parse_pdf.extract_text_from_pdf",
                return_value="body",
            ),
            patch(
                "pathlib.Path.write_text",
                autospec=True,
            ) as write_text_mock,
        ):
            output = parse_and_save(
                "/input/sample.pdf",
                extraction_type=ExtractionType.PYMUPDF4LLM,
            )

        self.assertEqual(output, expected_output)
        mkdir_mock.assert_any_call(
            Path("/tmp/kajima-files/parsed/pymupdf4llm"),
            parents=True,
            exist_ok=True,
        )
        write_text_mock.assert_called_once_with(
            expected_output,
            "body",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
