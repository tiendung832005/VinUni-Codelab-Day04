"""Chạy test trên bản sao dữ liệu để không thêm ticket vào dữ liệu bài lab."""

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter-code"))

import tools


@pytest.fixture(autouse=True)
def isolated_raw_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "raw-data"
    shutil.copytree(ROOT / "raw-data", data_dir)
    monkeypatch.setattr(tools, "RAW_DATA_DIR", str(data_dir))
    return data_dir
