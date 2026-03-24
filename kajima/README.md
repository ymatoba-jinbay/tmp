# kajima - ボーリング柱状図PDF情報抽出・評価パイプライン

ボーリング柱状図PDFからLLMを使って情報を構造化抽出し、XMLの正解データと比較して精度を評価するツール群。

## ディレクトリ構成

```
kajima/
├── files/              # データ
│   ├── pdf/            # 元データ（ボーリング柱状図PDF）627件
│   ├── xml/            # 解析データ（UTF-8変換済み）627件
│   ├── parsed/         # PDFテキスト抽出結果（parse_pdfの出力先）
│   │   ├── pymupdf4llm/ # pymupdf4llmでMarkdown変換
│   │   ├── pymupdf/     # pymupdfでテキスト+テーブル検出（Markdown）
│   │   ├── pymupdf_html/    # pymupdf MarkdownからHTML変換
│   │   ├── pymupdf4llm_html/ # pymupdf4llm MarkdownからHTML変換
│   │   └── position/    # pdfplumberで座標付きテキスト
│   ├── results_{model}/ # LLM抽出結果（extract_llmの出力先）
│   │   └── {parse_type}/ # parse_type別のJSON結果
│   └── evaluations_{model}/ # 評価結果（evaluateの出力先）
│       └── {parse_type}.json
├── parse_xml.py        # XMLファイルの汎用dict変換・動的スキーマ生成
├── parse_pdf.py        # PDFからテキスト抽出・保存
├── extract_llm.py      # LLMによる構造化情報抽出（XMLから動的スキーマ生成）
└── evaluate.py         # XML正解データとLLM結果の精度評価
```

PDF/XMLはファイル名でペアになっている（例: `BED01405_080103-012-004IBR.pdf` と `BED01405_080103-012-004IBR.xml`）。

## アーキテクチャ

### 動的スキーマ生成

固定のPydanticスキーマではなく、**対応するXMLの構造を都度解析して動的にJSON Schemaを生成**する。

- `parse_xml.py` がXMLを日本語タグ名をキーとしたネストdictに変換
- `build_json_schema()` がそのdictからJSON Schemaを推論（gensonライブラリ使用）
- LLMにはそのファイルに実際に存在するフィールドだけのスキーマが渡される

これにより：
- ファイルごとに異なるフィールド構成に自動対応
- スキーマのキーが日本語のままなのでLLMが正確にマッピング可能
- 新しいタグが増えてもコード変更不要

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

対応するXMLからスキーマを動的に生成し、そのスキーマに基づいてPDF/テキストから情報を抽出する。

```bash
# PDFを直接Geminiに渡す（デフォルト: --parse-type pdf, --llm gemini）
uv run python -m kajima.extract_llm

# パース済みテキストから（Gemini）
uv run python -m kajima.extract_llm --parse-type pymupdf --llm gemini

# パース済みテキストから（Claude）
uv run python -m kajima.extract_llm --parse-type pymupdf --llm claude

# モデル名を指定
uv run python -m kajima.extract_llm --llm gemini --model gemini-2.5-flash

# 先頭5件のみ
uv run python -m kajima.extract_llm --limit 5

# XMLディレクトリを指定（デフォルト: kajima/files/xml）
uv run python -m kajima.extract_llm --xml-dir kajima/files/xml
```

オプション:

| フラグ | 説明 | デフォルト |
|---|---|---|
| `--parse-type` | 入力タイプ（`pdf`, `pymupdf`, `markdown`, `html`, `pymupdf_html`） | `pdf` |
| `--llm` | 使用するLLM（`gemini`, `claude`） | `gemini` |
| `--model` | モデル名を上書き | Gemini: `gemini-2.5-flash`, Claude: `anthropic.claude-sonnet-4-20250514-v1:0` |
| `--limit` | 処理するファイル数（0=全件） | `0` |
| `--xml-dir` | XMLディレクトリ | `kajima/files/xml` |

入力先: `kajima/files/pdf/`（parse-type=pdf）または `kajima/files/parsed/<parse_type>/`
出力先: `kajima/files/results_{model}/{parse_type}/` に `{ファイル名}.json` として保存。

### Step 3: 精度評価

```bash
# Gemini結果の評価（モデル名は必須）
uv run python -m kajima.evaluate --model gemini-2.5-flash

# Claude結果の評価
uv run python -m kajima.evaluate --model anthropic.claude-sonnet-4-20250514-v1:0

# parse-typeを指定
uv run python -m kajima.evaluate --model gemini-2.5-flash --parse-type pymupdf

# 全parse typeを一括評価（results_<llm>/ 内のサブディレクトリを自動検出）
uv run python -m kajima.evaluate --model gemini-2.5-flash --parse-type all

# XMLディレクトリを指定
uv run python -m kajima.evaluate --model gemini-2.5-flash --xml-dir kajima/files/xml
```

オプション:

| フラグ | 説明 | デフォルト |
|---|---|---|
| `--model` | モデル名（**必須**） | - |
| `--parse-type` | 入力parse type（`pdf`, `pymupdf`, `markdown`, `html`, `pymupdf_html`, `all`） | `pdf` |
| `--xml-dir` | XMLディレクトリ | `kajima/files/xml` |

> **Note**: `--parse-type all` を指定すると、`results_<llm>/` 内に存在する全サブディレクトリ（parse type）を自動検出し、それぞれに対して評価を実行します。

結果ディレクトリ: `kajima/files/results_{model}/{parse_type}/`
出力先: `kajima/files/evaluations_{model}/{parse_type}.json`

### XMLの単独解析

```bash
# XMLをネストdictとしてJSON出力（日本語タグ名がキー）
uv run python -m kajima.parse_xml kajima/files/xml/BED01405_080103-012-004IBR.xml

# 動的に生成されるJSON Schemaを確認
uv run python -m kajima.parse_xml kajima/files/xml/BED01405_080103-012-004IBR.xml --schema
```

## 前提条件

OCR機能を使用するには Tesseract のインストールが必要です。

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-jpn
```

## 環境変数

`.env` ファイルに設定（`extract_llm.py`のCLI実行時に自動読み込み）。

| 変数 | 用途 |
|---|---|
| `PROJECT_ID` | GCP プロジェクトID（Gemini用） |
| `LOCATION` | GCP リージョン（デフォルト: `us-central1`） |

Claude（Bedrock経由）は AWS クレデンシャル（`AWS_PROFILE` 等）で認証。

## 評価ロジック

- XMLとLLM結果をそれぞれフラットなkey-valueペア（ドット記法）に変換して比較
- XMLの値が空でないフィールドのみ評価対象
- LLMが値を返したフィールドのみ正誤判定（precisionベース）、未抽出フィールドは別集計
- NFKC Unicode正規化後に完全一致で判定
- エラー分類: `numeric_close`（数値の微差）、`partial_match`（部分一致）、`wrong_value`（不一致）
- ファイルごとの精度 + 全体の精度 + セクション別分析を算出
