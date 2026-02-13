from pathlib import Path
import io
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import (
    close_run_group,
    create_run_group,
    export_groups_rtf,
    export_model_csv,
    export_model_markdown,
    list_run_groups,
    run_model,
    upload_dataset,
)


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
    stats_by_var = {row["variable"]: row for row in result["descriptive_stats"]}
    assert set(stats_by_var) == {"y", "x1", "c1"}
    assert stats_by_var["y"]["n"] == 6
    assert stats_by_var["y"]["mean"] == pytest.approx(3.5, abs=1e-9)
    assert stats_by_var["y"]["variance"] == pytest.approx(3.5, abs=1e-9)
    assert stats_by_var["y"]["median"] == pytest.approx(3.5, abs=1e-9)
    assert stats_by_var["y"]["min"] == pytest.approx(1.0, abs=1e-9)
    assert stats_by_var["y"]["max"] == pytest.approx(6.0, abs=1e-9)

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


def test_multiple_groups_list_and_close():
    csv_content = b"y,x1,x2\n1,1,2\n2,2,3\n3,3,4\n4,4,5\n5,5,6\n6,6,7\n"
    uploaded = upload_dataset("groups.csv", csv_content)
    dataset_id = uploaded["dataset_id"]

    g1 = create_run_group(
        {
            "dataset_id": dataset_id,
            "y_col": "y",
            "x_cols": ["x1"],
            "c_cols": [],
            "fit_options": {"intercept": True, "robust_se": "none"},
        }
    )
    g2 = create_run_group(
        {
            "dataset_id": dataset_id,
            "y_col": "y",
            "x_cols": ["x2"],
            "c_cols": [],
            "fit_options": {"intercept": True, "robust_se": "HC1"},
            "group_title": "Alt Spec",
        }
    )
    assert g1["group_id"] != g2["group_id"]

    groups = list_run_groups(dataset_id)
    assert len(groups) == 2
    assert set(g["title"] for g in groups) == {"Model 1", "Alt Spec"}

    close_result = close_run_group(g1["group_id"])
    assert close_result["group_id"] == g1["group_id"]
    assert close_result["already_closed"] is False

    groups_after_close = list_run_groups(dataset_id)
    assert len(groups_after_close) == 1
    assert groups_after_close[0]["group_id"] == g2["group_id"]


def test_upload_dta_with_labels():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6], "x1": [1, 2, 3, 4, 5, 6], "g": [0, 1, 0, 1, 0, 1]})
    buf = io.BytesIO()
    df.to_stata(
        buf,
        write_index=False,
        variable_labels={"y": "Outcome", "x1": "Predictor X1", "g": "Group Flag"},
        value_labels={"g": {0: "Control", 1: "Treatment"}},
    )
    uploaded = upload_dataset("with_labels.dta", buf.getvalue())
    schema = uploaded["schema"]
    columns = {c["name"]: c for c in schema["columns"]}
    assert columns["y"]["label"] == "Outcome"
    assert columns["x1"]["label"] == "Predictor X1"
    assert columns["g"]["label"] == "Group Flag"
    assert schema["stata_meta"]["variable_labels"]["g"] == "Group Flag"
    assert schema["stata_meta"]["value_labels"]["g"]["0"] == "Control"
    assert schema["stata_meta"]["value_labels"]["g"]["1"] == "Treatment"


def test_export_selected_groups_to_rtf():
    csv_content = b"y,x1,x2\n1,1,2\n2,2,3\n3,3,4\n4,4,5\n5,5,6\n6,6,7\n"
    uploaded = upload_dataset("rtf_groups.csv", csv_content)
    dataset_id = uploaded["dataset_id"]

    g1 = create_run_group(
        {
            "dataset_id": dataset_id,
            "y_col": "y",
            "x_cols": ["x1"],
            "c_cols": [],
            "fit_options": {"intercept": True, "robust_se": "none"},
            "group_title": "Baseline",
        }
    )
    g2 = create_run_group(
        {
            "dataset_id": dataset_id,
            "y_col": "y",
            "x_cols": ["x2"],
            "c_cols": [],
            "fit_options": {"intercept": True, "robust_se": "none"},
            "group_title": "Alt",
        }
    )

    rtf_text, filename = export_groups_rtf(dataset_id, [g1["group_id"]])
    assert filename.endswith(".rtf")
    assert rtf_text.startswith("{\\rtf1")
    assert "Group 1: Baseline" in rtf_text
    assert "Descriptive Statistics" in rtf_text
    assert "Regression Results" in rtf_text
    assert "Alt" not in rtf_text

    rtf_text_2, _ = export_groups_rtf(dataset_id, [g2["group_id"], g1["group_id"]])
    assert "Group 1: Alt" in rtf_text_2
    assert "Group 2: Baseline" in rtf_text_2
