# 数据分析回归软件（后端 MVP）

基于 `项目大纲.md` 实现的可运行后端核心：

- 数据导入：`upload_dataset`（支持 `.csv/.xlsx/.xls/.dta`，自动返回前 200 行预览）
- DTA 标签：导入 `.dta` 时返回变量标签与取值标签（若文件内存在）
- 变量点击查看：在 Y/X/C 中点击变量可查看各取值标签与建议 dummy 映射
- 字段推断：返回每列缺失率与 `numeric/category/datetime/string` 推断
- 回归引擎：`run_model`（OLS，支持 `intercept` 与 `robust_se=HC1`）
- 预处理：支持 `log(x+1)` 与 `z-score`（按列配置）
- 导出能力：`export_model_csv`、`export_model_markdown`
- 快照存储：SQLite 持久化模型结果（幂等 `model_id`）
- 内置页面：`/` 可直接上传文件并运行回归
- 多回归组：同一数据可创建多个组并保留展示，可随时关闭
- 描述统计：按组变量展示样本数、均值、方差、中位数、最小值、最大值
- RTF 导出：可多选回归组，一键导出顶刊三线表（描述统计 + 回归结果）

## Quick Start

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动服务

```bash
python -m app.server
```

3. 打开浏览器

```text
http://127.0.0.1:8000/
```

## 主要函数

- `upload_dataset(filename: str, content: bytes) -> dict`
- `run_model(payload: dict) -> dict`
- `export_model_csv(model_id: str) -> str`
- `export_model_markdown(model_id: str) -> str`
- `get_model_snapshot(model_id: str) -> dict`

## run_model 入参示例

```python
payload = {
    "dataset_id": "...",
    "y_col": "y",
    "x_cols": ["x1", "x2"],
    "c_cols": ["c1"],
    "fit_options": {"intercept": True, "robust_se": "HC1"},
    "preprocess": {
        "log1p_cols": ["x1"],     # 先做 log(x+1)
        "zscore_cols": ["x1"],    # 再做 z-score
    },
}
```

## API 启动

```bash
python -m app.server
```

可选环境变量：

- `APP_HOST`（默认 `127.0.0.1`）
- `APP_PORT`（默认 `8000`）
- `APP_RELOAD`（`1/true` 开启热更新）

主要端点：

- `POST /api/datasets/upload`
- `POST /api/models/run`
- `POST /api/groups/create-and-run`
- `POST /api/groups/export.rtf`
- `GET /api/datasets/{dataset_id}/groups`
- `DELETE /api/groups/{group_id}`
- `GET /api/models/{id}`
- `GET /api/models/{id}/export.csv`
- `GET /api/models/{id}/export.md`

## UI 功能

- 上传 `.csv/.xlsx/.xls/.dta`
- 选择 `Y/X/C` 并设置 `intercept`、`robust_se`
- 配置 `log1p`、`z-score` 预处理列
- 创建多个回归组并在侧边面板切换查看
- 关闭不需要的回归组
- 多选回归组并导出 `.rtf` 三线表
- 查看 summary / 系数表 / warnings
- 一键打开 CSV 与 Markdown 导出
