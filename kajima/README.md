# kajima - ボーリング柱状図PDF情報抽出・評価パイプライン

ボーリング柱状図PDFからLLMを使って情報を構造化抽出し、XMLの正解データと比較して精度を評価するツール群。

## ディレクトリ構成

```
kajima/
├── files/              # データ
│   ├── pdf/            # 元データ（ボーリング柱状図PDF）627件
│   ├── xml/            # 解析データ（Shift_JIS, CRLF）627件
│   └── parsed/         # PDFテキスト抽出結果（parse_pdfの出力先）
│       ├── markdown/
│       ├── html/
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
```

### Step 1: PDFからテキスト抽出

```bash
# Markdown（テーブル構造付き、推奨） → kajima/files/parsed/markdown/
uv run python -m kajima.parse_pdf kajima/files/pdf/ --extraction-type markdown

# HTML（位置・フォント情報付き） → kajima/files/parsed/html/
uv run python -m kajima.parse_pdf kajima/files/pdf/ --extraction-type html

# プレーンテキスト（pdfplumber） → kajima/files/parsed/text/
uv run python -m kajima.parse_pdf kajima/files/pdf/ --extraction-type text

# 座標付きテキスト（pdfplumber） → kajima/files/parsed/position/
uv run python -m kajima.parse_pdf kajima/files/pdf/ --extraction-type position

# 先頭5件のみ
uv run python -m kajima.parse_pdf kajima/files/pdf/ --limit 5
```

抽出方式:

| `--extraction-type` | 説明 | 出力先 | 出力拡張子 |
|---|---|---|---|
| `markdown` | pymupdf4llmでMarkdown変換（テーブル構造付き、デフォルト） | `kajima/files/parsed/markdown/` | `.md` |
| `html` | pymupdfでHTML変換（位置・フォント情報付き） | `kajima/files/parsed/html/` | `.html` |
| `text` | pdfplumberでプレーンテキスト抽出 | `kajima/files/parsed/text/` | `.txt` |
| `position` | pdfplumberで文字座標付きテキスト | `kajima/files/parsed/position/` | `.txt` |

> **Note**: pymupdf系はOCRを使わず埋め込みテキストを直接使用します。

出力先: `--output-dir`（デフォルト: `kajima/files/parsed/<extraction_type>/`）

### Step 2: LLMで構造化情報を抽出

```bash
# Gemini（VertexAI経由）
uv run python -m kajima.extract_llm kajima/files/parsed/markdown/ --llm gemini

# Claude（Bedrock経由）
uv run python -m kajima.extract_llm kajima/files/parsed/markdown/ --llm claude

# 単一ファイル指定
uv run python -m kajima.extract_llm kajima/files/parsed/markdown/BED01405_080103-012-004IBR.md --llm gemini
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
