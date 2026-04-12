"""Build a formatted Google Sheet from evaluation results.

Creates a spreadsheet with 5 tabs (Summary / Detail / Precision / Failed Patterns /
Error Types) matching the reference format used for 精度検証:

- 色付きのヘッダー行
- frozen rows/columns
- precision/recall/f1 列への gradient 条件付き書式
- Detail タブの basic filter (precision DESC)
- Error Types の 全体行には別の gradient

## 認証方式

``gog auth tokens export <email>`` で取り出した refresh_token を使い、
そのユーザーのDriveルートにファイルを作成する。
作成後のファイルはユーザー本人が所有するので、"Shared with me" ではなく
My Driveのルートに直接表示される。
``KAJIMA_GOOGLE_ACCOUNT`` 環境変数 または ``--google-account`` で対象emailを指定。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.oauth2.credentials import Credentials as UserCredentials
from googleapiclient.discovery import build

GOG_CREDENTIALS_PATH = Path.home() / ".config" / "gogcli" / "credentials.json"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _load_user_credentials(email: str) -> UserCredentials:
    """Build google OAuth Credentials for ``email`` via ``gog auth tokens export``.

    gog stores its refresh tokens in an encrypted keyring. We call
    ``gog auth tokens export`` into a temp file, combine it with the OAuth
    client ID/secret from ~/.config/gogcli/credentials.json, then delete
    the temp file (since it contains the refresh token).
    """
    if not GOG_CREDENTIALS_PATH.exists():
        raise RuntimeError(
            f"{GOG_CREDENTIALS_PATH} not found. Run `gog auth credentials set`."
        )
    client_creds = json.loads(GOG_CREDENTIALS_PATH.read_text())

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
        token_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "gog",
                "auth",
                "tokens",
                "export",
                email,
                "--out",
                str(token_path),
                "--overwrite",
            ],
            check=True,
            capture_output=True,
        )
        token_data = json.loads(token_path.read_text())
    finally:
        try:
            token_path.unlink()
        except FileNotFoundError:
            pass

    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        raise RuntimeError(f"gog export for {email} did not contain a refresh_token")

    return UserCredentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_creds["client_id"],
        client_secret=client_creds["client_secret"],
        scopes=SCOPES,
    )


def _load_credentials(
    *,
    google_account: str | None,
):
    """Return OAuth user credentials for the given Google account."""
    if not google_account:
        google_account = os.environ.get("KAJIMA_GOOGLE_ACCOUNT")
    if not google_account:
        raise RuntimeError(
            "No auth specified. Pass --google-account <email> or "
            "set KAJIMA_GOOGLE_ACCOUNT."
        )
    return _load_user_credentials(google_account)


# ----- Colors (matches reference sheet) -----
HEADER_BG = {"red": 0.26, "green": 0.52, "blue": 0.96}  # Google blue
HEADER_FG = {"red": 1.0, "green": 1.0, "blue": 1.0}
GRADIENT_MAX_GREEN = {
    "red": 0.34117648,
    "green": 0.73333335,
    "blue": 0.5411765,
}
GRADIENT_MAX_BLUE = {
    "red": 0.2901961,
    "green": 0.5254902,
    "blue": 0.9098039,
}
GRADIENT_MIN_WHITE = {"red": 1.0, "green": 1.0, "blue": 1.0}


@dataclass
class SheetSpec:
    """Declarative spec for a single tab in the output spreadsheet."""

    title: str
    headers: list[str]
    rows: list[list[Any]]
    frozen_rows: int = 1
    frozen_cols: int = 0
    # List of (start_col, end_col) tuples (0-indexed, end exclusive) where to
    # render a white→green gradient across numeric data rows.
    gradient_cols: list[tuple[int, int]] | None = None
    # Same but for the "summary" rows at the bottom (different color).
    summary_gradient: tuple[int, int, int] | None = (
        None  # (start_col, end_col, summary_row_count)
    )
    # Columns (0-indexed) to format as percent (e.g. 0.873 -> "87.3%").
    percent_cols: list[int] | None = None
    # Enable basic filter sorted by this column (0-indexed, DESC).
    sort_col: int | None = None


def build_spreadsheet(
    title: str,
    specs: list[SheetSpec],
    *,
    google_account: str | None = None,
    share_with: str | None = None,
) -> dict[str, str]:
    """Create a spreadsheet with the given specs and return {id, url}."""
    creds = _load_credentials(
        google_account=google_account,
    )
    sheets_api = build("sheets", "v4", credentials=creds, cache_discovery=False)
    drive_api = build("drive", "v3", credentials=creds, cache_discovery=False)

    # --- 1. create the spreadsheet with all tabs up-front ---
    create_body = {
        "properties": {"title": title, "locale": "ja_JP"},
        "sheets": [
            {
                "properties": {
                    "title": spec.title,
                    "index": i,
                    "gridProperties": {
                        "rowCount": max(len(spec.rows) + 10, 100),
                        "columnCount": max(len(spec.headers), 10),
                        "frozenRowCount": spec.frozen_rows,
                        "frozenColumnCount": spec.frozen_cols,
                    },
                }
            }
            for i, spec in enumerate(specs)
        ],
    }
    created = sheets_api.spreadsheets().create(body=create_body).execute()
    spreadsheet_id = created["spreadsheetId"]
    sheet_ids: dict[str, int] = {
        s["properties"]["title"]: s["properties"]["sheetId"] for s in created["sheets"]
    }

    # --- 2. upload values for each tab ---
    data_updates = [
        {
            "range": f"'{spec.title}'!A1",
            "values": [spec.headers] + [list(r) for r in spec.rows],
        }
        for spec in specs
    ]
    sheets_api.spreadsheets().values().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"valueInputOption": "RAW", "data": data_updates},
    ).execute()

    # --- 3. apply formatting in one batchUpdate ---
    requests: list[dict] = []
    for spec in specs:
        sheet_id = sheet_ids[spec.title]
        n_cols = len(spec.headers)
        n_rows = len(spec.rows) + 1  # +1 for header

        # Header: bold, white text, blue background
        requests.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": 1,
                        "startColumnIndex": 0,
                        "endColumnIndex": n_cols,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": HEADER_BG,
                            "textFormat": {
                                "foregroundColor": HEADER_FG,
                                "bold": True,
                            },
                            "horizontalAlignment": "CENTER",
                        }
                    },
                    "fields": (
                        "userEnteredFormat(backgroundColor,textFormat,"
                        "horizontalAlignment)"
                    ),
                }
            }
        )

        # Percent columns
        for col in spec.percent_cols or []:
            requests.append(
                {
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 1,
                            "endRowIndex": n_rows,
                            "startColumnIndex": col,
                            "endColumnIndex": col + 1,
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "numberFormat": {
                                    "type": "PERCENT",
                                    "pattern": "0.0%",
                                }
                            }
                        },
                        "fields": "userEnteredFormat.numberFormat",
                    }
                }
            )

        # Gradient conditional formats (primary: green)
        summary_rows = 0
        if spec.summary_gradient is not None:
            summary_rows = spec.summary_gradient[2]

        primary_end_row = n_rows - summary_rows
        for start_col, end_col in spec.gradient_cols or []:
            requests.append(
                {
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [
                                {
                                    "sheetId": sheet_id,
                                    "startRowIndex": 1,
                                    "endRowIndex": primary_end_row,
                                    "startColumnIndex": start_col,
                                    "endColumnIndex": end_col,
                                }
                            ],
                            "gradientRule": {
                                "minpoint": {
                                    "color": GRADIENT_MIN_WHITE,
                                    "type": "MIN",
                                },
                                "maxpoint": {
                                    "color": GRADIENT_MAX_GREEN,
                                    "type": "MAX",
                                },
                            },
                        },
                        "index": 0,
                    }
                }
            )

        # Summary rows gradient (alternate color)
        if spec.summary_gradient is not None:
            start_col, end_col, _ = spec.summary_gradient
            requests.append(
                {
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [
                                {
                                    "sheetId": sheet_id,
                                    "startRowIndex": primary_end_row,
                                    "endRowIndex": n_rows,
                                    "startColumnIndex": start_col,
                                    "endColumnIndex": end_col,
                                }
                            ],
                            "gradientRule": {
                                "minpoint": {
                                    "color": GRADIENT_MIN_WHITE,
                                    "type": "MIN",
                                },
                                "maxpoint": {
                                    "color": GRADIENT_MAX_BLUE,
                                    "type": "MAX",
                                },
                            },
                        },
                        "index": 0,
                    }
                }
            )

        # Basic filter (+ optional sort)
        if spec.sort_col is not None:
            requests.append(
                {
                    "setBasicFilter": {
                        "filter": {
                            "range": {
                                "sheetId": sheet_id,
                                "startRowIndex": 0,
                                "endRowIndex": n_rows,
                                "startColumnIndex": 0,
                                "endColumnIndex": n_cols,
                            },
                            "sortSpecs": [
                                {
                                    "dimensionIndex": spec.sort_col,
                                    "sortOrder": "DESCENDING",
                                }
                            ],
                        }
                    }
                }
            )

        # Auto-resize columns
        requests.append(
            {
                "autoResizeDimensions": {
                    "dimensions": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": 0,
                        "endIndex": n_cols,
                    }
                }
            }
        )

    sheets_api.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id, body={"requests": requests}
    ).execute()

    # --- 4. share with user if requested ---
    if share_with:
        drive_api.permissions().create(
            fileId=spreadsheet_id,
            body={"type": "user", "role": "writer", "emailAddress": share_with},
            sendNotificationEmail=False,
            fields="id",
        ).execute()

    return {
        "id": spreadsheet_id,
        "url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
    }
