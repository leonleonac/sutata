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
    raise ValueError(f"CSV 解析失败: {' | '.join(errors)}")


def _parse_file(filename: str, content: bytes) -> list[dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {suffix}，仅支持 {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    if pd is None:
        if suffix != ".csv":
            raise ValueError("当前环境缺少 pandas，仅支持 CSV；Excel/DTA 请安装 pandas 后重试")
        return _read_csv_with_fallback(content)

    if suffix == ".csv":
        last_error: Exception | None = None
        for enc in ("utf-8", "gbk"):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        else:
            raise ValueError(f"CSV 解析失败: {last_error}") from last_error
    elif suffix in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(io.BytesIO(content))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Excel 解析失败: {exc}") from exc
    elif suffix == ".dta":
        try:
            df = pd.read_stata(io.BytesIO(content))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"DTA 解析失败: {exc}") from exc
    else:  # pragma: no cover
        raise ValueError(f"不支持的文件类型: {suffix}")

    if df.empty:
        return []

    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    rows = df.where(pd.notnull(df), None).to_dict(orient="records")
    return _clean_rows(rows)


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


def _build_schema(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "numeric_columns": [],
        }

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
                "missing_rate": missing / row_count,
                "inferred_type": inferred_type,
            }
        )

    return {
        "row_count": row_count,
        "column_count": len(columns),
        "columns": info,
        "numeric_columns": numeric_columns,
    }


def upload_dataset(filename: str, content: bytes) -> dict[str, Any]:
    rows = _parse_file(filename, content)
    if not rows:
        raise ValueError("空数据集")

    schema = _build_schema(rows)
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
        raise ValueError("缺失值")
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
            raise ValueError("矩阵不可逆，可能存在完全共线性")
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
        raise ValueError("删除缺失后样本数小于 5")

    k = len(data_x[0])
    if n <= k:
        raise ValueError(f"样本数不足（n={n}）无法估计参数（k={k}）")

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
        warnings.append(f"因缺失值删除样本 {dropped} 行。")
    if n <= 2 * k:
        warnings.append("样本数相对变量数偏少，估计结果可能不稳定。")

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
        raise RuntimeError("缺少 pandas/statsmodels")

    model_vars = [y_col, *regressors]
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("空数据集")

    for col in model_vars:
        if col not in df.columns:
            raise ValueError(f"字段不存在: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    sliced = df[model_vars].dropna(axis=0)
    dropped = n_before - len(sliced)
    n = len(sliced)
    if n < 5:
        raise ValueError("删除缺失后样本数小于 5")

    y = sliced[y_col]
    X = sliced[regressors]
    if intercept:
        X = sm.add_constant(X, has_constant="add")

    try:
        model = sm.OLS(y, X).fit(cov_type="HC1" if robust_se == "HC1" else "nonrobust")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"OLS 计算失败: {exc}") from exc

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
        warnings.append(f"因缺失值删除样本 {dropped} 行。")
    if n <= 2 * len(coefficients):
        warnings.append("样本数相对变量数偏少，估计结果可能不稳定。")
    cond = getattr(model, "condition_number", None)
    if isinstance(cond, (int, float)) and cond > 1e4:
        warnings.append(f"检测到较强多重共线性风险（condition_number={cond:.2f}）。")

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
        raise ValueError("至少选择一个 X 或 C 变量")
    if robust_se not in {"none", "HC1"}:
        raise ValueError("fit_options.robust_se 仅支持 'none' 或 'HC1'")
    if y_col in regressors:
        raise ValueError("y_col 不能与 X/C 变量重复")

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


def _ensure_str_list(name: str, value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} 必须是字符串列表")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} 仅允许非空字符串")
    return value


def _apply_preprocess(rows: list[dict[str, Any]], preprocess: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    if not rows:
        return rows, {"log1p_cols_applied": [], "zscore_cols_applied": []}, []

    available_columns = set(rows[0].keys())
    log1p_cols = _ensure_str_list("preprocess.log1p_cols", preprocess.get("log1p_cols"))
    zscore_cols = _ensure_str_list("preprocess.zscore_cols", preprocess.get("zscore_cols"))

    unknown = [c for c in [*log1p_cols, *zscore_cols] if c not in available_columns]
    if unknown:
        raise ValueError(f"preprocess 包含不存在的列: {', '.join(sorted(set(unknown)))}")

    transformed = [dict(row) for row in rows]
    warnings: list[str] = []
    applied_log_cols: list[str] = []
    applied_z_cols: list[str] = []

    # 按固定顺序处理：先 log1p，再标准化。
    for col in log1p_cols:
        values: list[float] = []
        for row in transformed:
            raw = row.get(col)
            if raw in (None, ""):
                continue
            try:
                numeric = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"preprocess.log1p_cols 中列 {col} 含非数值，无法转换") from exc
            if numeric < 0:
                raise ValueError(f"preprocess.log1p_cols 中列 {col} 存在负值，无法执行 log(x+1)")
            values.append(numeric)

        if not values:
            warnings.append(f"log1p 跳过列 {col}（无可用数值）。")
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
                raise ValueError(f"preprocess.zscore_cols 中列 {col} 含非数值，无法标准化") from exc

        if len(values) < 2:
            warnings.append(f"z-score 跳过列 {col}（有效样本不足 2）。")
            continue

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = math.sqrt(variance)
        if std == 0:
            warnings.append(f"z-score 跳过列 {col}（标准差为 0）。")
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
        raise ValueError("缺少 dataset_id")
    bundle = DATASETS.get(dataset_id)
    if bundle is None:
        raise ValueError("dataset_id 不存在")

    y_col = payload.get("y_col")
    x_cols = _ensure_str_list("x_cols", payload.get("x_cols", []))
    c_cols = _ensure_str_list("c_cols", payload.get("c_cols", []))
    fit_options = payload.get("fit_options", {})
    preprocess = payload.get("preprocess", {})
    intercept = bool(fit_options.get("intercept", True))
    robust_se = fit_options.get("robust_se", "none")

    if not isinstance(y_col, str) or not y_col.strip():
        raise ValueError("y_col 必须是非空字符串")
    if robust_se not in {"none", "HC1"}:
        raise ValueError("fit_options.robust_se 仅支持 'none' 或 'HC1'")
    if not isinstance(preprocess, dict):
        raise ValueError("preprocess 必须是对象")

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

    result = run_ols(
        rows=processed_rows,
        y_col=y_col,
        x_cols=x_cols,
        c_cols=c_cols,
        intercept=intercept,
        robust_se=robust_se,
    )
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
        raise ValueError("模型不存在")
    return str(row[0])


def export_model_markdown(model_id: str) -> str:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT export_markdown FROM model_runs WHERE id = ?", (model_id,)).fetchone()
    if not row:
        raise ValueError("模型不存在")
    return str(row[0])


def get_model_snapshot(model_id: str) -> dict[str, Any]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT result_json FROM model_runs WHERE id = ?", (model_id,)).fetchone()
    if not row:
        raise ValueError("模型不存在")
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
        raise ValueError("缺少 dataset_id")

    group_title = payload.get("group_title")
    if group_title is not None:
        if not isinstance(group_title, str) or not group_title.strip():
            raise ValueError("group_title 必须是非空字符串")
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
        raise ValueError("缺少 dataset_id")
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
        raise ValueError("缺少 group_id")
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, closed_at FROM run_groups WHERE id = ?",
            (group_id,),
        ).fetchone()
        if not row:
            raise ValueError("回归组不存在")
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


def create_fastapi_app() -> Any:
    try:
        from fastapi import FastAPI, File, HTTPException
        from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FastAPI 未安装，无法创建 API 应用") from exc

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
            raise HTTPException(status_code=500, detail=f"上传失败: {exc}") from exc

    @api.post("/api/models/run")
    async def api_run_model(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return run_model(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"模型运行失败: {exc}") from exc

    @api.post("/api/groups/create-and-run")
    async def api_create_group_and_run(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return create_run_group(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"创建回归组失败: {exc}") from exc

    @api.get("/api/datasets/{dataset_id}/groups")
    async def api_list_groups(dataset_id: str, include_closed: bool = False) -> dict[str, Any]:
        try:
            groups = list_run_groups(dataset_id, include_closed=include_closed)
            return {"dataset_id": dataset_id, "groups": groups}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"读取回归组失败: {exc}") from exc

    @api.delete("/api/groups/{group_id}")
    async def api_close_group(group_id: str) -> dict[str, Any]:
        try:
            return close_run_group(group_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"关闭回归组失败: {exc}") from exc

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
            raise HTTPException(status_code=500, detail=f"导出失败: {exc}") from exc

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
            raise HTTPException(status_code=500, detail=f"导出失败: {exc}") from exc

    @api.get("/api/models/{model_id}")
    async def api_get_model(model_id: str) -> dict[str, Any]:
        try:
            return get_model_snapshot(model_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"读取失败: {exc}") from exc

    return api


try:  # pragma: no cover
    app = create_fastapi_app()
except RuntimeError:
    app = None
