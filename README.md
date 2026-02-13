# 数据分析回归软件（离线可运行修复版）

为解决当前环境无法安装第三方依赖（代理 403）导致程序不可运行的问题，现提供**标准库实现**：

- CSV 数据上传与预览（`upload_dataset`）
- OLS 回归（`run_model`，支持常数项与 HC1）
- 结果导出（`export_model_csv`）
- SQLite 持久化模型快照

> 说明：在离线环境中优先保证可运行与可测试。Excel/DTA、FastAPI Web 接口需要联网安装第三方依赖后再启用。

## 运行测试

```bash
pytest -q
```

## 示例

```python
from app.main import upload_dataset, run_model, export_model_csv

uploaded = upload_dataset("sample.csv", b"y,x1\n1,1\n2,2\n3,3\n4,4\n5,5\n")
res = run_model({
    "dataset_id": uploaded["dataset_id"],
    "y_col": "y",
    "x_cols": ["x1"],
    "c_cols": [],
    "fit_options": {"intercept": True, "robust_se": "none"},
})
print(export_model_csv(res["model_id"]))
```
