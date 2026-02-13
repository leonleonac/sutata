from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import export_model_csv, export_model_markdown, run_model, upload_dataset


def test_upload_run_export_offline_flow():
    csv_content = b"y,x1,c1\n1,1,0\n2,2,1\n3,3,0\n4,4,1\n5,5,0\n6,6,1\n"
    uploaded = upload_dataset("sample.csv", csv_content)
    dataset_id = uploaded["dataset_id"]
    assert uploaded["schema"]["row_count"] == 6
    assert uploaded["schema"]["column_count"] == 3
    assert "y" in uploaded["schema"]["numeric_columns"]

    result = run_model(
        {
            "dataset_id": dataset_id,
            "y_col": "y",
            "x_cols": ["x1"],
            "c_cols": ["c1"],
            "fit_options": {"intercept": True, "robust_se": "HC1"},
        }
    )
    assert result["summary"]["n"] == 6
    assert any(c["variable"] == "x1" for c in result["coefficients"])

    exported = export_model_csv(result["model_id"])
    assert "variable,coef,std_err" in exported

    exported_md = export_model_markdown(result["model_id"])
    assert "| variable | coef | std_err |" in exported_md


def test_same_payload_is_idempotent():
    csv_content = b"y,x1\n1,1\n2,2\n3,3\n4,4\n5,5\n6,6\n"
    uploaded = upload_dataset("idem.csv", csv_content)
    payload = {
        "dataset_id": uploaded["dataset_id"],
        "y_col": "y",
        "x_cols": ["x1"],
        "c_cols": [],
        "fit_options": {"intercept": True, "robust_se": "none"},
    }

    first = run_model(payload)
    second = run_model(payload)
    assert first["model_id"] == second["model_id"]


def test_preprocess_log1p_and_zscore():
    csv_content = b"y,x1,x2\n1,1,10\n2,2,10\n3,3,10\n4,4,10\n5,5,10\n6,6,10\n"
    uploaded = upload_dataset("pre.csv", csv_content)
    payload = {
        "dataset_id": uploaded["dataset_id"],
        "y_col": "y",
        "x_cols": ["x1", "x2"],
        "c_cols": [],
        "fit_options": {"intercept": True, "robust_se": "none"},
        "preprocess": {"log1p_cols": ["x1"], "zscore_cols": ["x1", "x2"]},
    }

    result = run_model(payload)
    assert "x1" in result["preprocess"]["log1p_cols_applied"]
    assert "x1" in result["preprocess"]["zscore_cols_applied"]
    assert "x2" not in result["preprocess"]["zscore_cols_applied"]
    assert any("z-score 跳过列 x2" in msg for msg in result["warnings"])
