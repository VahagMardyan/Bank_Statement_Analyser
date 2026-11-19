"""Monthly report: several tables in one .xlsx file, one sheet each.

The caller passes tables that are already translated; this module only lays
them out (bold headers, frozen header row, readable column widths).
"""

from __future__ import annotations

import io
import re

import pandas as pd
from openpyxl.styles import Font

MAX_WIDTH = 60
MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def sheet_name(name: str) -> str:
    """Excel: at most 31 characters, none of []:*?/\\"""
    return re.sub(r"[\[\]:*?/\\]", " ", name)[:31] or "Sheet"


def build_workbook(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, table in sheets.items():
            title = sheet_name(name)
            table.to_excel(writer, sheet_name=title, index=False)
            ws = writer.sheets[title]
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for column in ws.columns:
                longest = max((len(str(c.value)) for c in column if c.value is not None), default=0)
                ws.column_dimensions[column[0].column_letter].width = min(longest + 2, MAX_WIDTH)
    return buffer.getvalue()
