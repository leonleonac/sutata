from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "regression_app.db"
STATIC_DIR = APP_DIR / "static"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".dta"}

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None


@dataclass
class DatasetBundle:
    filename: str
    rows: list[dict[str, Any]]
    schema: dict[str, Any]
    created_at: datetime


DATASETS: dict[str, DatasetBundle] = {}


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                export_csv TEXT NOT NULL,
                export_markdown TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_groups (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                title TEXT NOT NULL,
                model_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                closed_at TEXT
            )
            """
        )
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(model_runs)").fetchall()
        }
        for name, col_type in (
            ("dataset_id", "TEXT NOT NULL DEFAULT ''"),
            ("result_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("export_markdown", "TEXT NOT NULL DEFAULT ''"),
        ):
            if name not in cols:
                conn.execute(f"ALTER TABLE model_runs ADD COLUMN {name} {col_type}")
        group_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(run_groups)").fetchall()
        }
        for name, col_type in (
            ("dataset_id", "TEXT NOT NULL DEFAULT ''"),
            ("title", "TEXT NOT NULL DEFAULT ''"),
            ("model_id", "TEXT NOT NULL DEFAULT ''"),
            ("payload_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("summary_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("created_at", "TEXT NOT NULL DEFAULT ''"),
            ("closed_at", "TEXT"),
        ):
            if name not in group_cols:
                conn.execute(f"ALTER TABLE run_groups ADD COLUMN {name} {col_type}")


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if hasattr(value, "to_pydatetime"):
        dt = value.to_pydatetime()
        return dt.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return _clean_value(value.item())
    return value


def _clean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: _clean_value(v) for k, v in row.items()} for row in rows]


def _read_csv_with_fallback(raw_bytes: bytes) -> list[dict[str, Any]]:
    errors: list[str] = []
    for enc in ("utf-8", "gbk"):
        try:
            text = raw_bytes.decode(enc)
            reader = csv.DictReader(io.StringIO(text))
            rows = [dict(r) for r in reader]
            return rows
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{enc}: {exc}")
    raise ValueError(f"CSV 瑙ｆ瀽澶辫触: {' | '.join(errors)}")


def _normalize_stata_value_labels(raw_value_labels: dict[Any, dict[Any, Any]]) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    for var_name, mapping in raw_value_labels.items():
        name = str(var_name)
        normalized[name] = {}
        for key, label in mapping.items():
            norm_key = _clean_value(key)
            normalized[name][str(norm_key)] = str(label)
    return normalized


def _parse_file(filename: str, content: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"涓嶆敮鎸佺殑鏂囦欢绫诲瀷: {suffix}锛屼粎鏀寔 {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    if pd is None:
        if suffix != ".csv":
            raise ValueError("pandas is required for Excel/DTA files; only CSV is supported without pandas")
        return _read_csv_with_fallback(content), {}

    source_meta: dict[str, Any] = {}
    if suffix == ".csv":
        last_error: Exception | None = None
        for enc in ("utf-8", "gbk"):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        else:
            raise ValueError(f"CSV 瑙ｆ瀽澶辫触: {last_error}") from last_error
    elif suffix in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(io.BytesIO(content))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Excel 瑙ｆ瀽澶辫触: {exc}") from exc
    elif suffix == ".dta":
        try:
            df = pd.read_stata(io.BytesIO(content))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"DTA 瑙ｆ瀽澶辫触: {exc}") from exc
        try:
            with pd.read_stata(io.BytesIO(content), iterator=True) as reader:
                variable_labels = {
                    str(k): str(v)
                    for k, v in dict(reader.variable_labels()).items()
                    if str(v).strip()
                }
                value_labels = _normalize_stata_value_labels(dict(reader.value_labels()))
                data_label = str(getattr(reader, "data_label", "") or "")
            source_meta["stata_meta"] = {
                "data_label": data_label,
                "variable_labels": variable_labels,
                "value_labels": value_labels,
            }
        except Exception:
            # Label extraction errors should not block file import.
            pass
    else:  # pragma: no cover
        raise ValueError(f"涓嶆敮鎸佺殑鏂囦欢绫诲瀷: {suffix}")

    if df.empty:
        return [], source_meta

    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    rows = df.where(pd.notnull(df), None).to_dict(orient="records")
    return _clean_rows(rows), source_meta


def _is_numeric(value: Any) -> bool:
    if value in (None, ""):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_datetime_like(value: Any) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, datetime):
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    text = text.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


def _infer_column_type(values: list[Any]) -> str:
    non_missing = [v for v in values if v not in (None, "")]
    if not non_missing:
        return "string"

    numeric_count = sum(1 for v in non_missing if _is_numeric(v))
    if numeric_count == len(non_missing):
        return "numeric"

    datetime_count = sum(1 for v in non_missing if _is_datetime_like(v))
    if datetime_count == len(non_missing):
        return "datetime"

    unique_count = len(set(str(v) for v in non_missing))
    if unique_count <= min(50, max(2, int(len(non_missing) * 0.3))):
        return "category"
    return "string"


def _build_schema(rows: list[dict[str, Any]], source_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    source_meta = source_meta or {}
    stata_meta = source_meta.get("stata_meta") if isinstance(source_meta, dict) else None
    variable_labels = {}
    if isinstance(stata_meta, dict):
        variable_labels = dict(stata_meta.get("variable_labels") or {})

    if not rows:
        base = {
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "numeric_columns": [],
        }
        if stata_meta:
            base["stata_meta"] = stata_meta
        return base

    columns = list(rows[0].keys())
    info: list[dict[str, Any]] = []
    numeric_columns: list[str] = []
    row_count = len(rows)

    for col in columns:
        values = [row.get(col) for row in rows]
        missing = sum(1 for v in values if v in (None, ""))
        inferred_type = _infer_column_type(values)
        if inferred_type == "numeric":
            numeric_columns.append(col)
        info.append(
            {
                "name": col,
                "label": variable_labels.get(col),
                "missing_rate": missing / row_count,
                "inferred_type": inferred_type,
            }
        )

    schema = {
        "row_count": row_count,
        "column_count": len(columns),
        "columns": info,
        "numeric_columns": numeric_columns,
    }
    if stata_meta:
        schema["stata_meta"] = stata_meta
    return schema


def upload_dataset(filename: str, content: bytes) -> dict[str, Any]:
    rows, source_meta = _parse_file(filename, content)
    if not rows:
        raise ValueError("绌烘暟鎹泦")

    schema = _build_schema(rows, source_meta=source_meta)
    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = DatasetBundle(
        filename=filename,
        rows=rows,
        schema=schema,
        created_at=datetime.now(timezone.utc),
    )
    return {
        "dataset_id": dataset_id,
        "schema": schema,
        "preview": rows[:200],
    }


def _to_float(value: Any) -> float:
    if value in (None, ""):
        raise ValueError("missing value")
    return float(value)


def _transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*a)]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows, cols, inner = len(a), len(b[0]), len(b)
    out = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = a[i][k]
            for j in range(cols):
                out[i][j] += aik * b[k][j]
    return out


def _inverse(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]

    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[pivot][i]) < 1e-12:
            raise ValueError("matrix is singular; possible perfect multicollinearity")
        aug[i], aug[pivot] = aug[pivot], aug[i]
        div = aug[i][i]
        for j in range(2 * n):
            aug[i][j] /= div
        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            for c in range(2 * n):
                aug[r][c] -= factor * aug[i][c]

    return [row[n:] for row in aug]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _run_ols_manual(
    rows: list[dict[str, Any]],
    y_col: str,
    regressors: list[str],
    intercept: bool,
    robust_se: str,
) -> dict[str, Any]:
    data_y: list[float] = []
    data_x: list[list[float]] = []
    dropped = 0
    for row in rows:
        try:
            y = _to_float(row.get(y_col))
            x = [_to_float(row.get(c)) for c in regressors]
            if intercept:
                x = [1.0] + x
            data_y.append(y)
            data_x.append(x)
        except Exception:
            dropped += 1

    n = len(data_y)
    if n < 5:
        raise ValueError("鍒犻櫎缂哄け鍚庢牱鏈暟灏忎簬 5")

    k = len(data_x[0])
    if n <= k:
        raise ValueError(f"insufficient observations (n={n}) for parameter count (k={k})")

    y_mat = [[v] for v in data_y]
    xt = _transpose(data_x)
    xtx = _matmul(xt, data_x)
    xtx_inv = _inverse(xtx)
    beta = _matmul(_matmul(xtx_inv, xt), y_mat)

    y_hat = [sum(data_x[i][j] * beta[j][0] for j in range(k)) for i in range(n)]
    resid = [data_y[i] - y_hat[i] for i in range(n)]
    sse = sum(e * e for e in resid)
    y_mean = sum(data_y) / n
    tss = sum((v - y_mean) ** 2 for v in data_y)
    r2 = 1 - sse / tss if tss else 0.0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else 0.0

    if robust_se == "HC1":
        meat = [[0.0 for _ in range(k)] for _ in range(k)]
        scale = n / (n - k)
        for i in range(n):
            xi = data_x[i]
            u2 = resid[i] ** 2
            for a in range(k):
                for b in range(k):
                    meat[a][b] += xi[a] * xi[b] * u2
        cov = _matmul(_matmul(xtx_inv, meat), xtx_inv)
        cov = [[v * scale for v in row] for row in cov]
    else:
        sigma2 = sse / (n - k)
        cov = [[xtx_inv[i][j] * sigma2 for j in range(k)] for i in range(k)]

    var_names = (["const"] if intercept else []) + regressors
    coefficients: list[dict[str, Any]] = []
    for i, name in enumerate(var_names):
        coef = beta[i][0]
        se = math.sqrt(max(cov[i][i], 0.0))
        t = coef / se if se > 0 else 0.0
        p = 2 * (1 - _normal_cdf(abs(t)))
        star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        coefficients.append(
            {
                "variable": name,
                "coef": coef,
                "std_err": se,
                "t": t,
                "p": p,
                "ci_lower": coef - 1.96 * se,
                "ci_upper": coef + 1.96 * se,
                "significance": star,
            }
        )

    warnings: list[str] = []
    if dropped:
        warnings.append(f"dropped {dropped} rows due to missing values")
    if n <= 2 * k:
        warnings.append("sample size is relatively small compared with variable count")

    f_stat = None
    prob_f = None
    if k > 1 and n > k:
        ess = tss - sse
        f_stat = (ess / (k - 1)) / (sse / (n - k)) if sse > 0 else None
        if f_stat is not None:
            prob_f = 2 * (1 - _normal_cdf(math.sqrt(abs(f_stat))))

    summary = {
        "r_squared": r2,
        "adj_r_squared": adj_r2,
        "f_stat": f_stat,
        "prob_f": prob_f,
        "n": n,
    }
    return {"summary": summary, "coefficients": coefficients, "warnings": warnings}


def _run_ols_statsmodels(
    rows: list[dict[str, Any]],
    y_col: str,
    regressors: list[str],
    intercept: bool,
    robust_se: str,
) -> dict[str, Any]:
    if pd is None or sm is None:  # pragma: no cover
        raise RuntimeError("缂哄皯 pandas/statsmodels")

    model_vars = [y_col, *regressors]
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("绌烘暟鎹泦")

    for col in model_vars:
        if col not in df.columns:
            raise ValueError(f"瀛楁涓嶅瓨鍦? {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    sliced = df[model_vars].dropna(axis=0)
    dropped = n_before - len(sliced)
    n = len(sliced)
    if n < 5:
        raise ValueError("鍒犻櫎缂哄け鍚庢牱鏈暟灏忎簬 5")

    y = sliced[y_col]
    X = sliced[regressors]
    if intercept:
        X = sm.add_constant(X, has_constant="add")

    try:
        model = sm.OLS(y, X).fit(cov_type="HC1" if robust_se == "HC1" else "nonrobust")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"OLS 璁＄畻澶辫触: {exc}") from exc

    ci = model.conf_int()
    coefficients: list[dict[str, Any]] = []
    for name in model.params.index:
        p_value = float(model.pvalues[name])
        star = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
        coefficients.append(
            {
                "variable": str(name),
                "coef": float(model.params[name]),
                "std_err": float(model.bse[name]),
                "t": float(model.tvalues[name]),
                "p": p_value,
                "ci_lower": float(ci.loc[name, 0]),
                "ci_upper": float(ci.loc[name, 1]),
                "significance": star,
            }
        )

    warnings: list[str] = []
    if dropped > 0:
        warnings.append(f"dropped {dropped} rows due to missing values")
    if n <= 2 * len(coefficients):
        warnings.append("sample size is relatively small compared with variable count")
    cond = getattr(model, "condition_number", None)
    if isinstance(cond, (int, float)) and cond > 1e4:
        warnings.append(f"potential multicollinearity detected (condition_number={cond:.2f})")

    def _safe_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number):
            return None
        return number

    summary = {
        "r_squared": _safe_float(model.rsquared),
        "adj_r_squared": _safe_float(model.rsquared_adj),
        "f_stat": _safe_float(model.fvalue),
        "prob_f": _safe_float(model.f_pvalue),
        "n": int(model.nobs),
    }
    return {"summary": summary, "coefficients": coefficients, "warnings": warnings}


def run_ols(
    rows: list[dict[str, Any]],
    y_col: str,
    x_cols: list[str],
    c_cols: list[str],
    intercept: bool,
    robust_se: str,
) -> dict[str, Any]:
    regressors: list[str] = []
    seen = set()
    for col in [*x_cols, *c_cols]:
        if col not in seen:
            regressors.append(col)
            seen.add(col)

    if not regressors:
        raise ValueError("鑷冲皯閫夋嫨涓€涓?X 鎴?C 鍙橀噺")
    if robust_se not in {"none", "HC1"}:
        raise ValueError("fit_options.robust_se 浠呮敮鎸?'none' 鎴?'HC1'")
    if y_col in regressors:
        raise ValueError("y_col 涓嶈兘涓?X/C 鍙橀噺閲嶅")

    if sm is not None and pd is not None:
        return _run_ols_statsmodels(rows, y_col, regressors, intercept, robust_se)
    return _run_ols_manual(rows, y_col, regressors, intercept, robust_se)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _render_model_csv(summary: dict[str, Any], coefficients: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    buf.write("r_squared,adj_r_squared,f_stat,prob_f,n\n")
    buf.write(
        ",".join(
            _format_value(summary.get(k))
            for k in ("r_squared", "adj_r_squared", "f_stat", "prob_f", "n")
        )
    )
    buf.write("\n\n")

    if coefficients:
        fieldnames = list(coefficients[0].keys())
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in coefficients:
            writer.writerow({k: _format_value(v) for k, v in row.items()})
    return buf.getvalue()


def _render_markdown_table(summary: dict[str, Any], coefficients: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key in ("r_squared", "adj_r_squared", "f_stat", "prob_f", "n"):
        lines.append(f"| {key} | {_format_value(summary.get(key))} |")
    lines.append("")

    if coefficients:
        header = list(coefficients[0].keys())
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for row in coefficients:
            lines.append("| " + " | ".join(_format_value(row.get(h)) for h in header) + " |")
    return "\n".join(lines)


def _model_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _unique_preserve(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _build_model_sample(rows: list[dict[str, Any]], y_col: str, regressors: list[str]) -> tuple[list[dict[str, float]], int]:
    sample: list[dict[str, float]] = []
    dropped = 0
    vars_in_model = [y_col, *regressors]
    for row in rows:
        parsed: dict[str, float] = {}
        ok = True
        for col in vars_in_model:
            try:
                parsed[col] = _to_float(row.get(col))
            except Exception:
                ok = False
                break
        if ok:
            sample.append(parsed)
        else:
            dropped += 1
    return sample, dropped


def _compute_descriptive_stats(sample_rows: list[dict[str, float]], variable_names: list[str]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for var in variable_names:
        values = [row[var] for row in sample_rows]
        n = len(values)
        if n == 0:
            stats.append(
                {
                    "variable": var,
                    "n": 0,
                    "mean": None,
                    "variance": None,
                    "median": None,
                    "min": None,
                    "max": None,
                }
            )
            continue
        values_sorted = sorted(values)
        mean = sum(values) / n
        if n % 2 == 1:
            median = values_sorted[n // 2]
        else:
            median = (values_sorted[(n // 2) - 1] + values_sorted[n // 2]) / 2
        variance = None
        if n > 1:
            variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        stats.append(
            {
                "variable": var,
                "n": n,
                "mean": mean,
                "variance": variance,
                "median": median,
                "min": values_sorted[0],
                "max": values_sorted[-1],
            }
        )
    return stats


def _ensure_str_list(name: str, value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} 蹇呴』鏄瓧绗︿覆鍒楄〃")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} 浠呭厑璁搁潪绌哄瓧绗︿覆")
    return value


def _apply_preprocess(rows: list[dict[str, Any]], preprocess: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    if not rows:
        return rows, {"log1p_cols_applied": [], "zscore_cols_applied": []}, []

    available_columns = set(rows[0].keys())
    log1p_cols = _ensure_str_list("preprocess.log1p_cols", preprocess.get("log1p_cols"))
    zscore_cols = _ensure_str_list("preprocess.zscore_cols", preprocess.get("zscore_cols"))

    unknown = [c for c in [*log1p_cols, *zscore_cols] if c not in available_columns]
    if unknown:
        raise ValueError(f"preprocess 鍖呭惈涓嶅瓨鍦ㄧ殑鍒? {', '.join(sorted(set(unknown)))}")

    transformed = [dict(row) for row in rows]
    warnings: list[str] = []
    applied_log_cols: list[str] = []
    applied_z_cols: list[str] = []

    # Process in fixed order: log1p first, then z-score.
    for col in log1p_cols:
        values: list[float] = []
        for row in transformed:
            raw = row.get(col)
            if raw in (None, ""):
                continue
            try:
                numeric = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"preprocess.log1p_cols 涓垪 {col} 鍚潪鏁板€硷紝鏃犳硶杞崲") from exc
            if numeric < 0:
                raise ValueError(f"preprocess.log1p_cols 涓垪 {col} 瀛樺湪璐熷€硷紝鏃犳硶鎵ц log(x+1)")
            values.append(numeric)

        if not values:
            warnings.append(f"log1p skipped for {col} (no usable numeric values)")
            continue

        for row in transformed:
            raw = row.get(col)
            if raw in (None, ""):
                continue
            row[col] = math.log1p(float(raw))
        applied_log_cols.append(col)

    for col in zscore_cols:
        values = []
        for row in transformed:
            raw = row.get(col)
            if raw in (None, ""):
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"preprocess.zscore_cols column {col} contains non-numeric values") from exc

        if len(values) < 2:
            warnings.append(f"z-score skipped for {col} (fewer than 2 valid samples)")
            continue

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = math.sqrt(variance)
        if std == 0:
            warnings.append(f"z-score skipped for {col} (standard deviation is 0)")
            continue

        for row in transformed:
            raw = row.get(col)
            if raw in (None, ""):
                continue
            row[col] = (float(raw) - mean) / std
        applied_z_cols.append(col)

    report = {
        "log1p_cols_applied": applied_log_cols,
        "zscore_cols_applied": applied_z_cols,
    }
    return transformed, report, warnings


def run_model(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise ValueError("缂哄皯 dataset_id")
    bundle = DATASETS.get(dataset_id)
    if bundle is None:
        raise ValueError("dataset_id does not exist")

    y_col = payload.get("y_col")
    x_cols = _ensure_str_list("x_cols", payload.get("x_cols", []))
    c_cols = _ensure_str_list("c_cols", payload.get("c_cols", []))
    fit_options = payload.get("fit_options", {})
    preprocess = payload.get("preprocess", {})
    intercept = bool(fit_options.get("intercept", True))
    robust_se = fit_options.get("robust_se", "none")

    if not isinstance(y_col, str) or not y_col.strip():
        raise ValueError("y_col 蹇呴』鏄潪绌哄瓧绗︿覆")
    if robust_se not in {"none", "HC1"}:
        raise ValueError("fit_options.robust_se 浠呮敮鎸?'none' 鎴?'HC1'")
    if not isinstance(preprocess, dict):
        raise ValueError("preprocess must be an object")

    preprocess_log_cols = _ensure_str_list("preprocess.log1p_cols", preprocess.get("log1p_cols"))
    preprocess_zscore_cols = _ensure_str_list("preprocess.zscore_cols", preprocess.get("zscore_cols"))
    processed_rows, preprocess_report, preprocess_warnings = _apply_preprocess(
        bundle.rows,
        {"log1p_cols": preprocess_log_cols, "zscore_cols": preprocess_zscore_cols},
    )
    normalized_payload = {
        "dataset_id": dataset_id,
        "y_col": y_col,
        "x_cols": x_cols,
        "c_cols": c_cols,
        "fit_options": {"intercept": intercept, "robust_se": robust_se},
        "preprocess": {
            "log1p_cols": preprocess_log_cols,
            "zscore_cols": preprocess_zscore_cols,
        },
    }
    model_id = _model_fingerprint(normalized_payload)
    regressors = _unique_preserve([*x_cols, *c_cols])

    result = run_ols(
        rows=processed_rows,
        y_col=y_col,
        x_cols=x_cols,
        c_cols=c_cols,
        intercept=intercept,
        robust_se=robust_se,
    )
    model_sample, _ = _build_model_sample(processed_rows, y_col=y_col, regressors=regressors)
    descriptive_stats = _compute_descriptive_stats(model_sample, [y_col, *regressors])
    summary = result["summary"]
    coefficients = result["coefficients"]
    warnings = [*preprocess_warnings, *result["warnings"]]

    output = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "coefficients": coefficients,
        "warnings": warnings,
        "fit_options": normalized_payload["fit_options"],
        "preprocess": preprocess_report,
        "descriptive_stats": descriptive_stats,
    }
    export_csv = _render_model_csv(summary, coefficients)
    export_markdown = _render_markdown_table(summary, coefficients)

    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO model_runs (id, created_at, dataset_id, payload_json, result_json, export_csv, export_markdown)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                created_at=excluded.created_at,
                dataset_id=excluded.dataset_id,
                payload_json=excluded.payload_json,
                result_json=excluded.result_json,
                export_csv=excluded.export_csv,
                export_markdown=excluded.export_markdown
            """,
            (
                model_id,
                datetime.now(timezone.utc).isoformat(),
                dataset_id,
                json.dumps(normalized_payload, ensure_ascii=False),
                json.dumps(output, ensure_ascii=False),
                export_csv,
                export_markdown,
            ),
        )

    return output


def export_model_csv(model_id: str) -> str:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT export_csv FROM model_runs WHERE id = ?", (model_id,)).fetchone()
    if not row:
        raise ValueError("model not found")
    return str(row[0])


def export_model_markdown(model_id: str) -> str:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT export_markdown FROM model_runs WHERE id = ?", (model_id,)).fetchone()
    if not row:
        raise ValueError("model not found")
    return str(row[0])


def get_model_snapshot(model_id: str) -> dict[str, Any]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT result_json FROM model_runs WHERE id = ?", (model_id,)).fetchone()
    if not row:
        raise ValueError("model not found")
    return json.loads(str(row[0]))


def _next_group_title(conn: sqlite3.Connection, dataset_id: str) -> str:
    row = conn.execute(
        "SELECT COUNT(*) FROM run_groups WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchone()
    count = int(row[0]) if row else 0
    return f"Model {count + 1}"


def create_run_group(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise ValueError("缂哄皯 dataset_id")

    group_title = payload.get("group_title")
    if group_title is not None:
        if not isinstance(group_title, str) or not group_title.strip():
            raise ValueError("group_title 蹇呴』鏄潪绌哄瓧绗︿覆")
        group_title = group_title.strip()

    model_payload = {k: v for k, v in payload.items() if k != "group_title"}
    result = run_model(model_payload)

    init_db()
    group_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        title = group_title or _next_group_title(conn, dataset_id)
        conn.execute(
            """
            INSERT INTO run_groups (id, dataset_id, title, model_id, payload_json, summary_json, created_at, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                group_id,
                dataset_id,
                title,
                result["model_id"],
                json.dumps(model_payload, ensure_ascii=False),
                json.dumps(result["summary"], ensure_ascii=False),
                now,
            ),
        )

    return {
        "group_id": group_id,
        "dataset_id": dataset_id,
        "title": title,
        "model_id": result["model_id"],
        "summary": result["summary"],
        "created_at": now,
        "closed_at": None,
        "payload": model_payload,
        "result": result,
    }


def list_run_groups(dataset_id: str, include_closed: bool = False) -> list[dict[str, Any]]:
    if not dataset_id:
        raise ValueError("缂哄皯 dataset_id")
    init_db()
    query = """
        SELECT id, dataset_id, title, model_id, payload_json, summary_json, created_at, closed_at
        FROM run_groups
        WHERE dataset_id = ?
    """
    params: tuple[Any, ...] = (dataset_id,)
    if not include_closed:
        query += " AND closed_at IS NULL"
    query += " ORDER BY created_at DESC"

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(query, params).fetchall()

    groups: list[dict[str, Any]] = []
    for row in rows:
        groups.append(
            {
                "group_id": str(row[0]),
                "dataset_id": str(row[1]),
                "title": str(row[2]),
                "model_id": str(row[3]),
                "payload": json.loads(str(row[4])),
                "summary": json.loads(str(row[5])),
                "created_at": str(row[6]),
                "closed_at": None if row[7] is None else str(row[7]),
            }
        )
    return groups


def close_run_group(group_id: str) -> dict[str, Any]:
    if not group_id:
        raise ValueError("缂哄皯 group_id")
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, closed_at FROM run_groups WHERE id = ?",
            (group_id,),
        ).fetchone()
        if not row:
            raise ValueError("鍥炲綊缁勪笉瀛樺湪")
        if row[1] is not None:
            return {
                "group_id": group_id,
                "closed_at": str(row[1]),
                "already_closed": True,
            }
        closed_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE run_groups SET closed_at = ? WHERE id = ?",
            (closed_at, group_id),
        )
    return {"group_id": group_id, "closed_at": closed_at, "already_closed": False}


def _rtf_escape(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "{":
            out.append("\\{")
        elif ch == "}":
            out.append("\\}")
        else:
            code = ord(ch)
            if 32 <= code <= 126:
                out.append(ch)
            else:
                out.append(f"\\u{code}?")
    return "".join(out)


def _rtf_format_number(value: Any, digits: int = 4, *, thousands_sep: bool = False) -> str:
    if value is None:
        return ""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(num):
        return ""
    if thousands_sep:
        if digits == 0:
            return f"{int(round(num)):,}"
        return f"{num:,.{digits}f}"
    if digits == 0:
        return str(int(round(num)))
    return f"{num:.{digits}f}"


def _ensure_int_range(name: str, value: Any, default: int, min_value: int, max_value: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer in [{min_value}, {max_value}]")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer in [{min_value}, {max_value}]")
        value = int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        if not text.isdigit():
            raise ValueError(f"{name} must be an integer in [{min_value}, {max_value}]")
        value = int(text)
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer in [{min_value}, {max_value}]")
    if value < min_value or value > max_value:
        raise ValueError(f"{name} must be between {min_value} and {max_value}")
    return value


def _ensure_bool(name: str, value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean (true/false)")


def _normalize_rtf_export_format(format_options: dict[str, Any] | None) -> dict[str, Any]:
    raw = format_options or {}
    if not isinstance(raw, dict):
        raise ValueError("format_options must be an object")
    return {
        "coef_digits": _ensure_int_range("format_options.coef_digits", raw.get("coef_digits"), 3, 0, 8),
        "t_digits": _ensure_int_range("format_options.t_digits", raw.get("t_digits"), 2, 0, 8),
        "stat_digits": _ensure_int_range("format_options.stat_digits", raw.get("stat_digits"), 3, 0, 8),
        "desc_digits": _ensure_int_range("format_options.desc_digits", raw.get("desc_digits"), 3, 0, 8),
        "thousands_sep": _ensure_bool("format_options.thousands_sep", raw.get("thousands_sep"), True),
        "align_significance": _ensure_bool("format_options.align_significance", raw.get("align_significance"), True),
    }


def _rtf_table(
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[int],
    *,
    bottom_border_rows: set[int] | None = None,
    align_first_left: bool = True,
    body_alignment: str = "\\qc",
) -> str:
    if len(headers) != len(col_widths):
        raise ValueError("RTF table header count does not match col width count")
    if body_alignment not in {"\\ql", "\\qc", "\\qr"}:
        raise ValueError("body_alignment must be one of \\ql / \\qc / \\qr")

    lines: list[str] = []
    all_rows = [headers, *rows]
    if bottom_border_rows is None:
        bottom_border_rows = {len(all_rows) - 1}
    for i, row in enumerate(all_rows):
        if len(row) != len(headers):
            continue
        row_def = "\\trowd\\trgaph90\\trleft0"
        if i == 0:
            row_def += "\\trbrdrt\\brdrs\\brdrw20"
        if i in bottom_border_rows:
            row_def += "\\trbrdrb\\brdrs\\brdrw20"
        lines.append(row_def)

        x = 0
        for width in col_widths:
            x += width
            lines.append(f"\\cellx{x}")

        for cidx, cell in enumerate(row):
            if align_first_left and cidx == 0:
                lines.append(f"\\pard\\intbl\\ql {_rtf_escape(cell)}\\cell")
            else:
                lines.append(f"\\pard\\intbl{body_alignment} {_rtf_escape(cell)}\\cell")
        lines.append("\\row")
    return "\n".join(lines) + "\n"


def _format_coef_cell(
    coef_item: dict[str, Any],
    *,
    coef_digits: int,
    thousands_sep: bool,
    align_significance: bool,
    coef_width: int,
) -> str:
    coef_text = _rtf_format_number(coef_item.get("coef"), digits=coef_digits, thousands_sep=thousands_sep)
    sig = str(coef_item.get("significance") or "")
    if not coef_text:
        return ""
    if not align_significance:
        return f"{coef_text}{sig}"
    fig_space = "\u2007"
    width = max(coef_width, len(coef_text))
    padded_coef = coef_text.rjust(width, fig_space)
    sig_block = sig[:3].ljust(3, fig_space)
    return f"{padded_coef}{sig_block}"


def export_groups_rtf(
    dataset_id: str,
    group_ids: list[str],
    format_options: dict[str, Any] | None = None,
) -> tuple[str, str]:
    if not dataset_id:
        raise ValueError("缂哄皯 dataset_id")
    if not isinstance(group_ids, list) or not group_ids or any(not isinstance(g, str) or not g for g in group_ids):
        raise ValueError("group_ids 蹇呴』鏄潪绌哄瓧绗︿覆鍒楄〃")
    fmt = _normalize_rtf_export_format(format_options)
    coef_digits = int(fmt["coef_digits"])
    t_digits = int(fmt["t_digits"])
    stat_digits = int(fmt["stat_digits"])
    desc_digits = int(fmt["desc_digits"])
    thousands_sep = bool(fmt["thousands_sep"])
    align_significance = bool(fmt["align_significance"])

    init_db()
    placeholders = ",".join("?" for _ in group_ids)
    params: tuple[Any, ...] = (dataset_id, *group_ids)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"""
            SELECT id, dataset_id, title, model_id, payload_json, summary_json, created_at, closed_at
            FROM run_groups
            WHERE dataset_id = ? AND id IN ({placeholders})
            """,
            params,
        ).fetchall()

    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_id[str(row[0])] = {
            "group_id": str(row[0]),
            "dataset_id": str(row[1]),
            "title": str(row[2]),
            "model_id": str(row[3]),
            "payload": json.loads(str(row[4])),
            "summary": json.loads(str(row[5])),
            "created_at": str(row[6]),
            "closed_at": None if row[7] is None else str(row[7]),
        }

    missing = [gid for gid in group_ids if gid not in by_id]
    if missing:
        raise ValueError(f"浠ヤ笅鍥炲綊缁勪笉瀛樺湪鎴栦笉灞炰簬褰撳墠鏁版嵁闆? {', '.join(missing)}")

    ordered_groups = [by_id[gid] for gid in group_ids]
    model_results = [get_model_snapshot(group["model_id"]) for group in ordered_groups]
    now_text = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc: list[str] = []
    doc.append("{\\rtf1\\ansi\\deff0")
    doc.append("{\\fonttbl{\\f0 Times New Roman;}}")
    doc.append("\\viewkind4\\uc1\\pard\\f0\\fs22")
    doc.append(f"\\b {_rtf_escape('Regression & Descriptive Statistics Report')}\\b0\\par")
    doc.append(f"{_rtf_escape(f'Dataset ID: {dataset_id}')}\\par")
    group_titles = " | ".join(f"({idx}) {g['title']}" for idx, g in enumerate(ordered_groups, start=1))
    doc.append(f"{_rtf_escape(f'Models: {group_titles}')}\\par")
    doc.append(f"{_rtf_escape(f'Generated: {now_text}')}\\par\\par")

    # Panel A: descriptive statistics for the union of variables across selected models.
    # Keep the first-seen order to mimic estout's stable listing.
    desc_map: dict[str, dict[str, Any]] = {}
    desc_order: list[str] = []
    for model in model_results:
        for row in model.get("descriptive_stats", []):
            var = str(row.get("variable", ""))
            if not var:
                continue
            if var not in desc_map:
                desc_order.append(var)
                desc_map[var] = row
    doc.append(f"\\b {_rtf_escape('Panel A: Descriptive statistics')}\\b0\\par")
    desc_headers = ["VARIABLES", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"]
    desc_rows: list[list[str]] = [
        ["", "N", "mean", "variance", "median", "min", "max"]
    ]
    desc_rows.extend(
        [
            [
                var,
                _rtf_format_number(desc_map[var].get("n"), digits=0, thousands_sep=thousands_sep),
                _rtf_format_number(desc_map[var].get("mean"), digits=desc_digits, thousands_sep=thousands_sep),
                _rtf_format_number(desc_map[var].get("variance"), digits=desc_digits, thousands_sep=thousands_sep),
                _rtf_format_number(desc_map[var].get("median"), digits=desc_digits, thousands_sep=thousands_sep),
                _rtf_format_number(desc_map[var].get("min"), digits=desc_digits, thousands_sep=thousands_sep),
                _rtf_format_number(desc_map[var].get("max"), digits=desc_digits, thousands_sep=thousands_sep),
            ]
            for var in desc_order
        ]
    )
    desc_bottom_rows = {1, len(desc_rows)}
    doc.append(
        _rtf_table(
            desc_headers,
            desc_rows,
            [2600, 850, 1150, 1400, 1300, 1100, 1100],
            bottom_border_rows=desc_bottom_rows,
            align_first_left=True,
            body_alignment="\\qr",
        )
    )
    doc.append("\\par")

    # Panel B: regression results in outreg2/estout style
    doc.append(f"\\b {_rtf_escape('Panel B: Regression results')}\\b0\\par")
    reg_headers = ["VARIABLES"] + [f"({idx})" for idx in range(1, len(ordered_groups) + 1)]
    reg_rows: list[list[str]] = [
        [""] + [str(group["title"]) for group in ordered_groups]
    ]

    coef_maps: list[dict[str, dict[str, Any]]] = []
    variable_order: list[str] = []
    seen_vars: set[str] = set()
    for model in model_results:
        cmap: dict[str, dict[str, Any]] = {}
        for row in model.get("coefficients", []):
            var = str(row.get("variable", ""))
            cmap[var] = row
            if var not in seen_vars:
                variable_order.append(var)
                seen_vars.add(var)
        coef_maps.append(cmap)
    coef_widths: list[int] = []
    for cmap in coef_maps:
        texts = [
            _rtf_format_number(item.get("coef"), digits=coef_digits, thousands_sep=thousands_sep)
            for item in cmap.values()
        ]
        coef_widths.append(max((len(text) for text in texts if text), default=0))

    for var in variable_order:
        label = "Constant" if var == "const" else var
        coef_line = [label]
        t_line = [""]
        for model_idx, cmap in enumerate(coef_maps):
            coef_item = cmap.get(var)
            if not coef_item:
                coef_line.append("")
                t_line.append("")
                continue
            coef_line.append(
                _format_coef_cell(
                    coef_item,
                    coef_digits=coef_digits,
                    thousands_sep=thousands_sep,
                    align_significance=align_significance,
                    coef_width=coef_widths[model_idx],
                )
            )
            t_text = _rtf_format_number(coef_item.get("t"), digits=t_digits, thousands_sep=thousands_sep)
            t_line.append(f"({t_text})" if t_text else "")
        reg_rows.append(coef_line)
        reg_rows.append(t_line)

    summary_rows = [
        ("Observations", "n", 0),
        ("R-squared", "r_squared", stat_digits),
        ("Adj. R-squared", "adj_r_squared", stat_digits),
    ]
    for title, key, digits in summary_rows:
        row = [title]
        for model in model_results:
            row.append(
                _rtf_format_number(
                    model.get("summary", {}).get(key),
                    digits=digits,
                    thousands_sep=thousands_sep,
                )
            )
        reg_rows.append(row)

    reg_bottom_rows = {1, len(reg_rows)}
    reg_col_widths = [3000] + [int((8600 - 3000) / max(len(ordered_groups), 1)) for _ in ordered_groups]
    doc.append(
        _rtf_table(
            reg_headers,
            reg_rows,
            reg_col_widths,
            bottom_border_rows=reg_bottom_rows,
            align_first_left=True,
            body_alignment="\\qr",
        )
    )
    robust_any = any(
        str(((group.get("payload") or {}).get("fit_options") or {}).get("robust_se") or "none").lower() != "none"
        for group in ordered_groups
    )
    t_note = "Robust t-statistics in parentheses" if robust_any else "t-statistics in parentheses"
    doc.append(f"\\pard\\qc {_rtf_escape(t_note)}\\par")
    doc.append(f"\\pard\\qc {_rtf_escape('*** p<0.01, ** p<0.05, * p<0.1')}\\par")
    doc.append("\\par")

    doc.append("}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"regression_groups_{timestamp}.rtf"
    return "\n".join(doc), filename


def create_fastapi_app() -> Any:
    try:
        from fastapi import FastAPI, File, HTTPException
        from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FastAPI 鏈畨瑁咃紝鏃犳硶鍒涘缓 API 搴旂敤") from exc

    api = FastAPI(title="Regression MVP API", version="1.0.0")

    @api.get("/", response_class=HTMLResponse)
    async def api_home() -> Any:
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return HTMLResponse(
            "<h3>Regression MVP API is running.</h3><p>Place app/static/index.html to enable UI.</p>"
        )

    @api.post("/api/datasets/upload")
    async def api_upload_dataset(file: Any = File(...)) -> dict[str, Any]:
        try:
            content = await file.read()
            filename = getattr(file, "filename", None) or "uploaded.csv"
            return upload_dataset(filename, content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"涓婁紶澶辫触: {exc}") from exc

    @api.post("/api/models/run")
    async def api_run_model(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return run_model(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"妯″瀷杩愯澶辫触: {exc}") from exc

    @api.post("/api/groups/create-and-run")
    async def api_create_group_and_run(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return create_run_group(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"鍒涘缓鍥炲綊缁勫け璐? {exc}") from exc

    @api.get("/api/datasets/{dataset_id}/groups")
    async def api_list_groups(dataset_id: str, include_closed: bool = False) -> dict[str, Any]:
        try:
            groups = list_run_groups(dataset_id, include_closed=include_closed)
            return {"dataset_id": dataset_id, "groups": groups}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"璇诲彇鍥炲綊缁勫け璐? {exc}") from exc

    @api.delete("/api/groups/{group_id}")
    async def api_close_group(group_id: str) -> dict[str, Any]:
        try:
            return close_run_group(group_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"鍏抽棴鍥炲綊缁勫け璐? {exc}") from exc

    @api.post("/api/groups/export.rtf")
    async def api_export_groups_rtf(payload: dict[str, Any]) -> Response:
        try:
            dataset_id = str(payload.get("dataset_id") or "")
            group_ids = _ensure_str_list("group_ids", payload.get("group_ids"))
            format_options = payload.get("format_options")
            if format_options is not None and not isinstance(format_options, dict):
                raise ValueError("format_options must be an object")
            rtf_text, filename = export_groups_rtf(dataset_id, group_ids, format_options=format_options)
            return Response(
                content=rtf_text,
                media_type="application/rtf; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"瀵煎嚭 RTF 澶辫触: {exc}") from exc

    @api.get("/api/models/{model_id}/export.csv")
    async def api_export_csv(model_id: str) -> PlainTextResponse:
        try:
            csv_text = export_model_csv(model_id)
            return PlainTextResponse(
                csv_text,
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="model_{model_id}.csv"'},
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"瀵煎嚭澶辫触: {exc}") from exc

    @api.get("/api/models/{model_id}/export.md")
    async def api_export_markdown(model_id: str) -> PlainTextResponse:
        try:
            markdown_text = export_model_markdown(model_id)
            return PlainTextResponse(
                markdown_text,
                media_type="text/markdown; charset=utf-8",
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"瀵煎嚭澶辫触: {exc}") from exc

    @api.get("/api/models/{model_id}")
    async def api_get_model(model_id: str) -> dict[str, Any]:
        try:
            return get_model_snapshot(model_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"璇诲彇澶辫触: {exc}") from exc

    return api


try:  # pragma: no cover
    app = create_fastapi_app()
except RuntimeError:
    app = None

