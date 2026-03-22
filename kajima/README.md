# kajima - ボーリング柱状図PDF情報抽出・評価パイプライン

ボーリング柱状図PDFからLLMを使って情報を構造化抽出し、XMLの正解データと比較して精度を評価するツール群。

## ディレクトリ構成

```
kajima/
├── files/              # データ
│   ├── pdf/            # 元データ（ボーリング柱状図PDF）627件
│   ├── xml/            # 解析データ（Shift_JIS, CRLF）627件
│   └── parsed/         # PDFテキスト抽出結果（parse_pdfの出力先）
│       ├── pymupdf4llm/ # pymupdf4llmでMarkdown変換
│       ├── pymupdf/     # pymupdfでテキスト+テーブル検出（Markdown）
│       ├── pymupdf_html/    # pymupdf MarkdownからHTML変換
│       ├── pymupdf4llm_html/ # pymupdf4llm MarkdownからHTML変換
│       └── position/
├── schema.py           # Pydanticスキーマ定義（BoringInfo等）
├── parse_xml.py        # XMLファイルの解析
├── parse_pdf.py        # PDFからテキスト抽出・保存
├── extract_llm.py      # LLMによる構造化情報抽出
└── evaluate.py         # XML正解データとLLM結果の精度評価
```

PDF/XMLはファイル名でペアになっている（例: `BED01405_080103-012-004IBR.pdf` と `BED01405_080103-012-004IBR.xml`）。

## パイプライン

```
PDF → [parse_pdf] → テキストファイル → [extract_llm] → JSON → [evaluate] ← XML
PDF ──────────────────────────────────→ [extract_llm] → JSON → [evaluate] ← XML
```

### Step 1: PDFからテキスト抽出

```bash
# pymupdf4llm（テーブル構造付きMarkdown） → kajima/files/parsed/pymupdf4llm/
uv run python -m kajima.parse_pdf kajima/files/pdf/ --extraction-type pymupdf4llm

# pymupdf（テキスト+テーブル検出、Markdown） → kajima/files/parsed/pymupdf/
uv run python -m kajima.parse_pdf kajima/files/pdf/ --extraction-type pymupdf

# HTML（パース済みMarkdownから変換） → kajima/files/parsed/{pymupdf,pymupdf4llm}_html/
uv run python -m kajima.parse_pdf kajima/files/pdf/ --extraction-type html

# 座標付きテキスト（pdfplumber） → kajima/files/parsed/position/
uv run python -m kajima.parse_pdf kajima/files/pdf/ --extraction-type position

# 先頭5件のみ
uv run python -m kajima.parse_pdf kajima/files/pdf/ --limit 5
```

抽出方式:

| `--extraction-type` | 説明 | 出力先 | 出力拡張子 |
|---|---|---|---|
| `pymupdf4llm` | pymupdf4llmでMarkdown変換（テーブル構造付き、デフォルト） | `kajima/files/parsed/pymupdf4llm/` | `.md` |
| `pymupdf` | pymupdfでテキスト+テーブル検出（OCRなし） | `kajima/files/parsed/pymupdf/` | `.md` |
| `html` | パース済みMarkdown（pymupdf/pymupdf4llm）をHTMLに変換 | `kajima/files/parsed/{source}_html/` | `.html` |
| `position` | pdfplumberで文字座標付きテキスト | `kajima/files/parsed/position/` | `.txt` |

> **Note**: pymupdf系はOCRを使わず埋め込みテキストを直接使用します。HTML変換は事前にpymupdf/pymupdf4llm方式のMarkdown出力が必要です。

出力先: `--output-dir`（デフォルト: `kajima/files/parsed/<extraction_type>/`）

### Step 2: LLMで構造化情報を抽出

```bash
# パース済みテキストから（Gemini）
uv run python -m kajima.extract_llm kajima/files/parsed/pymupdf/ --llm gemini

# パース済みテキストから（Claude）
uv run python -m kajima.extract_llm kajima/files/parsed/pymupdf/ --llm claude

# PDFを直接LLMに渡す
uv run python -m kajima.extract_llm kajima/files/pdf/ --llm gemini

# 単一ファイル指定
uv run python -m kajima.extract_llm kajima/files/pdf/BED01405_080103-012-004IBR.pdf --llm gemini
```

出力先: `--output-dir`（デフォルト: `kajima_results/`）に `{ファイル名}_{llm}.json` として保存。

### Step 3: 精度評価

```bash
# Gemini結果の評価
uv run python -m kajima.evaluate --llm gemini

# Claude結果の評価
uv run python -m kajima.evaluate --llm claude

# ディレクトリ指定
uv run python -m kajima.evaluate --xml-dir kajima/files/xml --result-dir kajima_results --llm gemini
```

出力先: `--output`（デフォルト: `kajima_eval/{llm}_evaluation.json`）

### XMLの単独解析

```bash
uv run python -m kajima.parse_xml kajima/files/xml/BED01405_080103-012-004IBR.xml
```

## 前提条件

OCR機能を使用するには Tesseract のインストールが必要です。

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-jpn
```

## 環境変数

`.env` ファイルに設定（`extract_llm.py`のCLI実行時のみ読み込み）。

| 変数 | 用途 |
|---|---|
| `PROJECT_ID` | GCP プロジェクトID（Gemini用） |
| `LOCATION` | GCP リージョン（デフォルト: `us-central1`） |

Claude（Bedrock経由）は AWS クレデンシャル（`AWS_PROFILE` 等）で認証。

## 評価ロジック

- XMLの値が空でないフィールドのみ評価対象（XMLの全内容がPDFに含まれるとは限らないため）
- NFKC Unicode正規化後に完全一致で判定
- ファイルごとの精度 + 全体の精度を算出
