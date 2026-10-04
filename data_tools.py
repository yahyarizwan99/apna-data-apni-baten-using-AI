"""Loading and profiling for Apna Data Apni Baten."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd

SUPPORTED = ["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"]


# ---------------------------------------------------------------- loading
def load_data(upload) -> pd.DataFrame:
    """Read an uploaded file into a DataFrame. Raises ValueError with a readable message."""
    name = getattr(upload, "name", "file")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    raw = upload.read() if hasattr(upload, "read") else open(upload, "rb").read()

    try:
        if ext in ("csv", "tsv", "txt"):
            sep = "\t" if ext == "tsv" else None
            df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(raw))
        elif ext == "json":
            try:
                df = pd.read_json(io.BytesIO(raw))
            except ValueError:
                df = pd.read_json(io.BytesIO(raw), lines=True)
        elif ext == "parquet":
            df = pd.read_parquet(io.BytesIO(raw))
        else:
            raise ValueError(f"'{ext}' files aren't supported. Use: {', '.join(SUPPORTED)}.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read {name}. {exc}") from exc

    if df.empty:
        raise ValueError(f"{name} has no rows in it.")
    return tidy(df)


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Trim column names and parse anything that looks like a date."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        # pandas 3 reports text columns as "str"; pandas 2 as "object"
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            continue
        sample = df[col].dropna().astype(str).head(40)
        if sample.empty:
            continue
        looks_datey = sample.str.contains(r"\d{4}-\d{2}|\d{1,2}/\d{1,2}/", regex=True).mean()
        if looks_datey > 0.7:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                df[col] = parsed
    return df


# ---------------------------------------------------------------- profiling
def _kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def profile_data(df: pd.DataFrame) -> dict:
    """Everything the dashboard and the AI need to know about this dataset."""
    rows, cols = df.shape
    cells = max(rows * cols, 1)
    missing_cells = int(df.isna().sum().sum())
    dupes = int(df.duplicated().sum())

    columns = []
    for name in df.columns:
        s = df[name]
        miss = int(s.isna().sum())
        nunique = int(s.nunique(dropna=True))
        try:
            top = s.mode(dropna=True)
            top = None if top.empty else top.iloc[0]
        except Exception:
            top = None
        columns.append({
            "column": name,
            "kind": _kind(s),
            "dtype": str(s.dtype),
            "missing": miss,
            "missing_pct": round(miss / max(rows, 1) * 100, 2),
            "unique": nunique,
            "unique_pct": round(nunique / max(rows, 1) * 100, 2),
            "most_frequent": None if top is None else str(top)[:60],
        })

    alerts = _alerts(df, rows, dupes, columns)

    score = 100.0
    score -= min(missing_cells / cells * 100, 40)
    score -= min(dupes / max(rows, 1) * 100 * 2, 20)
    score -= min(len([a for a in alerts if a["level"] == "warn"]) * 4, 25)
    score = int(max(0, round(score)))

    return {
        "rows": rows,
        "cols": cols,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_cells / cells * 100, 2),
        "duplicate_rows": dupes,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 2),
        "quality_score": score,
        "columns": columns,
        "alerts": alerts,
        "kind_counts": pd.Series([c["kind"] for c in columns]).value_counts().to_dict(),
    }


def _alerts(df, rows, dupes, columns) -> list[dict]:
    out = []
    if dupes:
        out.append({"level": "warn", "text": f"{dupes} duplicate rows ({dupes/max(rows,1)*100:.1f}% of the data)."})

    for c in columns:
        name = c["column"]
        if c["missing_pct"] >= 20:
            out.append({"level": "warn", "text": f"{name} — {c['missing_pct']}% of values are missing."})
        elif c["missing_pct"] > 0:
            out.append({"level": "info", "text": f"{name} — {c['missing_pct']}% missing."})

        if c["unique"] == 1:
            out.append({"level": "warn", "text": f"{name} — every row has the same value."})
        elif c["kind"] == "categorical" and c["unique_pct"] > 95 and rows > 20:
            out.append({"level": "info", "text": f"{name} — nearly all values are unique, so it looks like an ID."})

    num = df.select_dtypes(include=np.number)
    for name in num.columns:
        s = num[name].dropna()
        if len(s) < 10:
            continue
        try:
            sk = float(s.skew())
        except Exception:
            continue
        if abs(sk) > 2:
            out.append({"level": "info", "text": f"{name} — heavily skewed (skew {sk:.2f})."})
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            if n_out / len(s) > 0.05:
                out.append({"level": "info", "text": f"{name} — {n_out} outliers by the IQR rule."})

    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True).abs()
        seen = set()
        for a in corr.columns:
            for b in corr.columns:
                if a != b and (b, a) not in seen and corr.loc[a, b] >= 0.9:
                    seen.add((a, b))
                    out.append({"level": "info", "text": f"{a} and {b} are almost the same column (|r| ≥ 0.9)."})

    return out[:14]


def profile_to_text(df: pd.DataFrame, p: dict) -> str:
    """Compact text version of the profile — this is what the AI reads."""
    lines = [
        f"DATASET: {p['rows']} rows x {p['cols']} columns.",
        f"Missing {p['missing_pct']}% of cells. {p['duplicate_rows']} duplicate rows. Quality score {p['quality_score']}/100.",
        "",
        "COLUMNS:",
    ]
    for c in p["columns"]:
        bit = f"- {c['column']} ({c['kind']}, {c['dtype']}) missing {c['missing_pct']}%, {c['unique']} unique"
        if c["most_frequent"] is not None:
            bit += f", most common: {c['most_frequent']}"
        lines.append(bit)

    num = df.select_dtypes(include=np.number)
    if not num.empty:
        lines += ["", "NUMERIC SUMMARY:", num.describe().round(2).to_string()]

    if p["alerts"]:
        lines += ["", "DATA QUALITY NOTES:"] + [f"- {a['text']}" for a in p["alerts"]]

    lines += ["", "FIRST 5 ROWS:", df.head().to_string()]
    return "\n".join(lines)


def profile_to_markdown(p: dict) -> str:
    """Downloadable report."""
    out = [
        "# Data profile", "",
        f"- Rows: **{p['rows']}**", f"- Columns: **{p['cols']}**",
        f"- Missing cells: **{p['missing_pct']}%**",
        f"- Duplicate rows: **{p['duplicate_rows']}**",
        f"- Memory: **{p['memory_mb']} MB**",
        f"- Quality score: **{p['quality_score']}/100**", "",
        "## Columns", "",
        "| Column | Kind | Missing % | Unique | Most frequent |",
        "|---|---|---|---|---|",
    ]
    for c in p["columns"]:
        out.append(f"| {c['column']} | {c['kind']} | {c['missing_pct']}% | {c['unique']} | {c['most_frequent']} |")
    if p["alerts"]:
        out += ["", "## Quality notes", ""] + [f"- {a['text']}" for a in p["alerts"]]
    return "\n".join(out)
