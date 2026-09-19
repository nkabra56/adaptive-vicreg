import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from vicreg_tf import report_metrics as rm


def test_accuracy_column_matches_what_the_eval_scripts_write():
    linear_header = ["method", "dataset", "epochs", "batch", "image_size", "feat_dim", "lr", "l2", "opt", "test_acc"]
    knn_header = ["dataset", "method", "k", "temperature", "top1"]
    assert rm._pick_acc_col(pd.DataFrame(columns=linear_header)) == "test_acc"
    assert rm._pick_acc_col(pd.DataFrame(columns=knn_header)) == "top1"


def test_linear_table_reads_the_eval_linear_csv(tmp_path):
    csv = tmp_path / "linear_eval.csv"
    pd.DataFrame([{"method": "VICReg", "test_acc": 0.2174}]).to_csv(csv, index=False)
    table = rm.build_linear_table(str(tmp_path), {"run": [str(csv)]})
    assert table.loc[0, "method"] == "VICReg"
    assert table.loc[0, "top1"] == pytest.approx(0.2174)


def test_knn_table_takes_the_best_value_at_each_k(tmp_path):
    csv = tmp_path / "knn_eval.csv"
    pd.DataFrame(
        [
            {"method": "VICReg", "k": 20, "top1": 0.30},
            {"method": "VICReg", "k": 200, "top1": 0.35},
            {"method": "VICReg", "k": 200, "top1": 0.36},
        ]
    ).to_csv(csv, index=False)
    table = rm.build_knn_table(str(tmp_path), {"run": [str(csv)]}, {"run": []})
    assert table.loc[0, "top1_k20"] == pytest.approx(0.30)
    assert table.loc[0, "top1_k200"] == pytest.approx(0.36)


def test_parse_kv_spec():
    assert rm._parse_kv_spec("name=A, path = x.csv ,config=c.json") == {
        "name": "A",
        "path": "x.csv",
        "config": "c.json",
    }
    assert rm._parse_kv_spec("bare.csv") == {"path": "bare.csv"}


def test_unnamed_csv_attaches_only_when_there_is_a_single_run():
    one = {"a": rm.RunEntry(name="a", history="h")}
    rm._attach_file_to_named_run(one, ["p1.csv"], "linear_csvs")
    assert one["a"].linear_csvs == ["p1.csv"]

    two = {"a": rm.RunEntry(name="a", history="h"), "b": rm.RunEntry(name="b", history="h")}
    rm._attach_file_to_named_run(two, ["p1.csv", "name=b,path=p2.csv"], "linear_csvs")
    assert two["a"].linear_csvs == []
    assert two["b"].linear_csvs == ["p2.csv"]
