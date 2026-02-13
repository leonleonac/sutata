from __future__ import annotations

import csv
import io
import json
import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "regression_app.db"


@dataclass
class DatasetBundle:
    filename: str
    rows: list[dict[str, Any]]
    created_at: datetime


DATASETS: dict[str, DatasetBundle] = {}


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                export_csv TEXT NOT NULL
            )
            """
        )


def _read_csv_with_fallback(raw_bytes: bytes) -> list[dict[str, Any]]:
    errors: list[str] = []
    for enc in ("utf-8", "gbk"):
        try:
            text = raw_bytes.decode(enc)
            reader = csv.DictReader(io.StringIO(text))
            return [dict(r) for r in reader]
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{enc}: {exc}")
    raise ValueError(f"CSV 解析失败: {' | '.join(errors)}")


def _to_float(value: Any) -> float:
    if value is None or value == "":
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


def run_ols(rows: list[dict[str, Any]], y_col: str, x_cols: list[str], c_cols: list[str], intercept: bool, robust_se: str) -> dict[str, Any]:
    regressors = x_cols + c_cols
    if not regressors:
        raise ValueError("至少选择一个 X 或 C 变量")

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
    coefficients = []
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

    # 简化 F 统计量
    f_stat = None
    prob_f = None
    if k > 1 and n > k:
        ess = tss - sse
        f_stat = (ess / (k - 1)) / (sse / (n - k)) if sse > 0 else None
        if f_stat is not None:
            prob_f = 2 * (1 - _normal_cdf(math.sqrt(abs(f_stat))))

    summary = {"r_squared": r2, "adj_r_squared": adj_r2, "f_stat": f_stat, "prob_f": prob_f, "n": n}
    return {"summary": summary, "coefficients": coefficients, "warnings": warnings}


def upload_dataset(filename: str, content: bytes) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix != ".csv":
        raise ValueError("当前离线环境仅支持 CSV；Excel/DTA 需要安装 pandas 依赖")
    rows = _read_csv_with_fallback(content)
    if not rows:
        raise ValueError("空数据集")

    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = DatasetBundle(filename=filename, rows=rows, created_at=datetime.utcnow())

    columns = list(rows[0].keys())
    missing_rate = {}
    for c in columns:
        missing = sum(1 for r in rows if r.get(c) in (None, ""))
        missing_rate[c] = missing / len(rows)

    preview = rows[:200]
    numeric_columns = []
    for c in columns:
        try:
            for r in rows[:20]:
                if r.get(c) not in (None, ""):
                    float(r[c])
            numeric_columns.append(c)
        except Exception:
            pass

    return {
        "dataset_id": dataset_id,
        "schema": {
            "row_count": len(rows),
            "column_count": len(columns),
            "columns": [{"name": c, "missing_rate": missing_rate[c]} for c in columns],
            "numeric_columns": numeric_columns,
        },
        "preview": preview,
    }


def run_model(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_id = payload["dataset_id"]
    bundle = DATASETS.get(dataset_id)
    if not bundle:
        raise ValueError("dataset_id 不存在")

    result = run_ols(
        rows=bundle.rows,
        y_col=payload["y_col"],
        x_cols=payload.get("x_cols", []),
        c_cols=payload.get("c_cols", []),
        intercept=payload.get("fit_options", {}).get("intercept", True),
        robust_se=payload.get("fit_options", {}).get("robust_se", "none"),
    )

    model_id = str(uuid.uuid4())
    output = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "summary": result["summary"],
        "coefficients": result["coefficients"],
        "warnings": result["warnings"],
        "fit_options": payload.get("fit_options", {"intercept": True, "robust_se": "none"}),
    }

    csv_buf = io.StringIO()
    csv_buf.write("r_squared,adj_r_squared,f_stat,prob_f,n\n")
    s = output["summary"]
    csv_buf.write(f"{s['r_squared']},{s['adj_r_squared']},{s['f_stat']},{s['prob_f']},{s['n']}\n\n")
    writer = csv.DictWriter(csv_buf, fieldnames=list(output["coefficients"][0].keys()))
    writer.writeheader()
    writer.writerows(output["coefficients"])

    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO model_runs (id, created_at, payload_json, export_csv) VALUES (?, ?, ?, ?)",
            (model_id, datetime.utcnow().isoformat(), json.dumps(output, ensure_ascii=False), csv_buf.getvalue()),
        )

    return output


def export_model_csv(model_id: str) -> str:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT export_csv FROM model_runs WHERE id = ?", (model_id,)).fetchone()
    if not row:
        raise ValueError("模型不存在")
    return str(row[0])
