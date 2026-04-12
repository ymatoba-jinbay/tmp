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
│   ├── results_{llm}/  # LLM抽出結果（extract_llmの出力先）
│   │   └── {parse_type}/ # parse_type別のJSON結果
│   └── evaluations_{llm}/ # 評価結果（evaluateの出力先）
│       └── {parse_type}.json
├── parse_xml.py        # XMLファイルの汎用dict変換・動的スキーマ生成
├── parse_pdf.py        # PDFからテキスト抽出・保存
├── extract_llm.py      # LLMによる構造化情報抽出（XMLから動的スキーマ生成）
├── evaluate.py         # XML正解データとLLM結果の精度評価
├── summarize.py        # 全モデルの評価結果をTSVに集約
├── collect_labels.py   # 全予測結果からユニークなフィールドラベルを収集
└── check_overlap.py    # PDFから文字の重なりがないファイルを選定
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
                                                                    ↓
                                                              [summarize] → TSV
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
| `pymupdf4llm` | pymupdf4llmでMarkdown変換（テーブル構造付き、デフォルト） | `parsed/pymupdf4llm/` | `.md` |
| `pymupdf` | pymupdfでテキスト+テーブル検出（OCRなし） | `parsed/pymupdf/` | `.md` |
| `html` | パース済みMarkdown（pymupdf/pymupdf4llm）をHTMLに変換 | `parsed/{source}_html/` | `.html` |
| `position` | pdfplumberで文字座標付きテキスト | `parsed/position/` | `.txt` |

> **Note**: pymupdf系はOCRを使わず埋め込みテキストを直接使用します。HTML変換は事前にpymupdf/pymupdf4llm方式のMarkdown出力が必要です。

出力先: `--output-dir`（デフォルト: `kajima/files/parsed/<extraction_type>/`）

### Step 2: LLMで構造化情報を抽出

対応するXMLからスキーマを動的に生成し、そのスキーマに基づいてPDF/テキストから情報を抽出する。

```bash
# PDFを直接Geminiに渡す（デフォルト: --parse-type pdf, --llm gemini）
uv run python -m kajima.extract_llm

# PDFをJPG画像に変換してからLLMに渡す
uv run python -m kajima.extract_llm --parse-type jpg

# パース済みテキストから（Gemini）
uv run python -m kajima.extract_llm --parse-type pymupdf --llm gemini

# パース済みテキストから（Claude）
uv run python -m kajima.extract_llm --parse-type pymupdf --llm claude

# 座標付きテキスト（空間配置版）
uv run python -m kajima.extract_llm --parse-type position_spatial

# プロンプトを指定
uv run python -m kajima.extract_llm --prompt balanced_schemaless

# 先頭5件のみ（10件スキップ）
uv run python -m kajima.extract_llm --offset 10 --limit 5

# XMLディレクトリを指定（デフォルト: kajima/files/xml）
uv run python -m kajima.extract_llm --xml-dir kajima/files/xml
```

オプション:

| フラグ | 説明 | デフォルト |
|---|---|---|
| `--parse-type` | 入力タイプ（`pdf`, `jpg`, `position`, `position_spatial`, `pymupdf4llm`, `pymupdf`） | `pdf` |
| `--llm` | 使用するLLM（`gemini`, `claude`） | `gemini` |
| `--offset` | スキップするファイル数 | `0` |
| `--limit` | 処理するファイル数（0=全件） | `0` |
| `--xml-dir` | XMLディレクトリ | `kajima/files/xml` |
| `--prompt` | プロンプトファイル名（拡張子なし、例: `simple_schemaless`, `balanced_schemaless`） | なし |

デフォルトモデル: Gemini → `gemini-2.5-pro`, Claude → `jp.anthropic.claude-sonnet-4-6`

入力先: `kajima/files/pdf/`（parse-type=pdf/jpg）または `kajima/files/parsed/<parse_type>/`
出力先: `kajima/files/results_{llm}/{parse_type}/` に `{ファイル名}.json` として保存。

### Step 3: 精度評価

```bash
# Gemini結果の評価（デフォルト: --llm gemini, --parse-type pdf）
uv run python -m kajima.evaluate

# Claude結果の評価
uv run python -m kajima.evaluate --llm claude

# parse-typeを指定
uv run python -m kajima.evaluate --llm gemini --parse-type pymupdf

# 全parse typeを一括評価（results_{llm}/ 内のサブディレクトリを自動検出）
uv run python -m kajima.evaluate --llm gemini --parse-type all

# XMLディレクトリを指定
uv run python -m kajima.evaluate --xml-dir kajima/files/xml
```

オプション:

| フラグ | 説明 | デフォルト |
|---|---|---|
| `--parse-type` | 入力parse type（`pdf`, `jpg`, `position`, `position_spatial`, `pymupdf4llm`, `pymupdf`, `all`） | `pdf` |
| `--llm` | 使用するLLM（`gemini`, `claude`） | `gemini` |
| `--xml-dir` | XMLディレクトリ | `kajima/files/xml` |

> **Note**: `--parse-type all` を指定すると、`results_{llm}/` 内に存在する全サブディレクトリ（parse type）を自動検出し、それぞれに対して評価を実行します。

結果ディレクトリ: `kajima/files/results_{llm}/{parse_type}/`
出力先: `kajima/files/evaluations_{llm}/{parse_type}.json` と `.txt`

### Step 4: サマリー出力

全モデル・全parse typeの評価結果を横断的にTSVにまとめる。

```bash
uv run python -m kajima.summarize
```

`kajima/files/evaluations_*/` 内の全JSONを読み込み、以下のTSVを `kajima/files/` に出力:

| 出力ファイル | 内容 |
|---|---|
| `evaluation_summary_overall.tsv` | モデル×parse type別の全体メトリクス |
| `evaluation_summary_subsection.tsv` | モデル×parse type×セクション別のメトリクス |

### Step 1-4 をまとめて実行: `pipeline`

`collect_labels` → `evaluate`（全LLM × 全parse_type） → `summarize` TSV出力 →
フォーマット済みGoogle Sheet作成 を1コマンドで実行する。

```bash
# 毎回 --google-account を指定する場合
uv run python -m kajima.pipeline --google-account you@example.com

# 環境変数で固定しておくと指定不要
export KAJIMA_GOOGLE_ACCOUNT=you@example.com
uv run python -m kajima.pipeline

# 特定のLLMだけ（未指定なら results_{llm}/ から自動検出）
uv run python -m kajima.pipeline --llm gemini,claude

# 評価はスキップしてTSV+シートだけ再生成
uv run python -m kajima.pipeline --skip-evaluate

# シート作成はスキップ（evaluate + summarize TSVだけ）
uv run python -m kajima.pipeline --no-sheet

# 他メンバーにもeditor権限を付与したい
uv run python -m kajima.pipeline --share-with teammate@example.com
```

オプション:

| フラグ | 説明 | デフォルト |
|---|---|---|
| `--llm` | 対象LLM（カンマ区切り） | `results_{llm}/` から自動検出 |
| `--xml-dir` | XMLディレクトリ | `kajima/files/xml` |
| `--skip-evaluate` | 評価ステップを飛ばす（既存 `evaluations_*/*.json` を再利用） | false |
| `--no-sheet` | Google Sheet作成を飛ばす（evaluate + TSVだけ） | false |
| `--sheet-title` | 作成するSpreadsheetのタイトル | `kajima 精度検証 YYYY-MM-DD HH:MM` |
| `--google-account` | **シート作成の認証に使うGoogleアカウント(OAuth、既定)** | `$KAJIMA_GOOGLE_ACCOUNT` |
| `--share-with` | 追加でeditor権限を付与するGoogleアカウント | `$KAJIMA_SHEET_SHARE_WITH` |
| `--credentials` | GCPサービスアカウントjson（指定時はOAuthではなくSA認証を使う） | なし |

作成されるSpreadsheetは以下の5タブを持ち、参考フォーマット
([精度検証_修正版](https://docs.google.com/spreadsheets/d/1IHoVQaH3Qf8UIC0qVFExAo4CzIMvs50wOAHfEIf1ojQ/edit))
に合わせたヘッダー色・frozen行/列・条件付き書式・basic filterが適用される:

| タブ | 内容 | Frozen | 条件付き書式 |
|---|---|---|---|
| `Summary` | llm × parse_type の全体メトリクス（f1降順） | 先頭2列 | precision/recall/f1 の緑グラデ |
| `Detail` | llm × parse_type × subsection の内訳（precision降順のbasic filter付き） | 先頭1行 | precision/recall/f1 の緑グラデ |
| `Precision` | subsection × (llm_parse_type) の precision ピボット | 先頭1行+1列 | データ全体の緑グラデ |
| `Failed Patterns` | overall precisionより0.2以上低いsubsectionの誤答例（subsectionごとに最大10件） | 先頭1行 | — |
| `Error Types` | エラー種別の件数・比率。末尾に各LLMの「全体」行 | 先頭1行 | 主データ緑／全体行オレンジ |

共通フォーマット: ヘッダー行は青背景+白太字中央寄せ、precision/recall/f1やratio列は `0.0%` 表記、カラム幅はauto-resize。

#### 認証方式（シート作成に必要）

デフォルトは **OAuthユーザー認証** (gog CLI 経由)。`--google-account <email>`
で指定した Google アカウントで作成されるため、そのユーザーのMy Driveルートに
直接ファイルが現れる。

事前に gog CLI でアカウント登録が必要:

```bash
gog auth add you@example.com          # ブラウザでOAuth同意 (初回のみ)
gog auth list                          # 登録状態の確認
```

実装上は以下を行っている (`kajima/sheet_builder.py`):

1. 実行時に `gog auth tokens export <email> --out <tmp>` で refresh token を一時取得
2. `~/.config/gogcli/credentials.json` の OAuth client_id / client_secret と組み合わせて
   `google.oauth2.credentials.Credentials` を生成
3. 一時ファイルを即削除（refresh token を永続化しない）
4. その credentials で Sheets API / Drive API を直接叩く（frozen、条件付き書式、basic filterなど
   batchUpdate で一気に適用）

サービスアカウント認証（`--credentials jinbay-gcp.json`）も残してあるが、
service account は Drive ストレージを持たないため、そのままでは
`storageQuotaExceeded` になる。使う場合は Shared Drive か Domain-Wide Delegation の
設定が前提で、`kajima-dev` プロジェクトで以下2つのAPIが有効である必要もある:

- Google Sheets API: https://console.developers.google.com/apis/api/sheets.googleapis.com/overview?project=844348756373
- Google Drive API: https://console.developers.google.com/apis/api/drive.googleapis.com/overview?project=844348756373

通常は OAuth（`--google-account`）を使うのが簡単。

### XMLの単独解析

```bash
# XMLをネストdictとしてJSON出力（日本語タグ名がキー）
uv run python -m kajima.parse_xml kajima/files/xml/BED01405_080103-012-004IBR.xml

# 動的に生成されるJSON Schemaを確認
uv run python -m kajima.parse_xml kajima/files/xml/BED01405_080103-012-004IBR.xml --schema
```

### ユーティリティ

```bash
# テスト用PDFの選定（文字の重なりがない100件を選出 → test_filenames.txt）
uv run python -m kajima.check_overlap

# 全予測結果からユニークなフィールドラベルを収集 → all_prediction_labels.json
uv run python -m kajima.collect_labels
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
- 配列要素はインデックスではなくフィールド値のベストマッチで対応付け
- NFKC Unicode正規化後に完全一致で判定
- **Precision**: correct / (correct + incorrect)
- **Recall**: correct / (correct + incorrect + not_extracted)
- **F1**: 2 × precision × recall / (precision + recall)

not_extractedとfalse positiveの判定:
- `collect_labels` が全モデル・全parse typeの予測結果からフィールドラベルを収集（expected_labels）
- XMLに値があるがLLMが未抽出 → `not_extracted`（expected_labelsに含まれるもののみ集計）
- XMLに存在しないフィールドをLLMが出力 → `false_positive`（incorrect扱い、expected_labelsに含まれるもののみ）

エラー分類: `false_positive`（LLMのみ出力）、`numeric_close`（数値の微差）、`partial_match`（部分一致）、`wrong_value`（不一致）

ファイルごとの精度 + 全体の精度 + セクション別分析を算出。
