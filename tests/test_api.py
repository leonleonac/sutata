from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import export_model_csv, run_model, upload_dataset


def test_upload_run_export_offline_flow():
    csv_content = b"y,x1,c1\n1,1,0\n2,2,1\n3,3,0\n4,4,1\n5,5,0\n6,6,1\n"
    uploaded = upload_dataset("sample.csv", csv_content)
    dataset_id = uploaded["dataset_id"]
    assert uploaded["schema"]["row_count"] == 6

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
