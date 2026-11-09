"""Kiểm tra multi-tool, safeguards và lỗi dữ liệu ngoài tám test gốc."""

import json
from datetime import datetime

import pytest

import tools
from template import ToolCallingAgent


def test_combined_request_matches_sample():
    with open(tools.RAW_DATA_DIR + "/customer_queries.json", encoding="utf-8") as f:
        sample = json.load(f)[2]
    result = ToolCallingAgent().run(sample["query"])
    calls = [entry for entry in result["trace"] if "action" in entry]
    assert result["status"] == "completed"
    assert result["iterations"] == 2
    assert [entry["action"] for entry in calls] == sample["expected_tools"]
    assert calls[0]["arguments"] == sample["expected_args_catalog"]
    assert calls[1]["arguments"]["customer_name"] == "Phạm Thị Dung"
    assert "TK-" in result["answer"]
    assert "Nha Trang" in result["answer"]


def test_iteration_limit_does_not_submit_remaining_ticket(isolated_raw_data):
    path = isolated_raw_data / "support_tickets.json"
    before = path.read_bytes()
    result = ToolCallingAgent(max_iterations=1).run(
        "Tôi tên An, muốn xem xe điện dưới 600 triệu và ghi nhận xe bị lỗi."
    )
    assert result["status"] == "max_iterations_reached"
    assert result["iterations"] == 1
    assert "VF 3" in result["answer"]
    assert path.read_bytes() == before


def test_zero_iterations_never_calls_tool(monkeypatch):
    def unexpected_call(**kwargs):
        pytest.fail("Tool không được chạy khi max_iterations=0")
    monkeypatch.setitem(tools.TOOL_MAP, "search_product_catalog", unexpected_call)
    result = ToolCallingAgent(0).run("Xem xe điện VinFast")
    assert result["status"] == "max_iterations_reached"
    assert result["iterations"] == 0


def test_missing_name_asks_without_saving_ticket(isolated_raw_data):
    path = isolated_raw_data / "support_tickets.json"
    before = path.read_bytes()
    result = ToolCallingAgent().run("Xe VinFast bị lỗi, cần hỗ trợ gấp.")
    assert "cung cấp tên" in result["answer"]
    assert path.read_bytes() == before


def test_tickets_append_and_use_vietnam_timezone(isolated_raw_data):
    path = isolated_raw_data / "support_tickets.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    first = tools.submit_support_ticket("An", "Xe bị lỗi", "HIGH")
    second = tools.submit_support_ticket("Bình", "Phòng ẩm mốc")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[:len(original)] == original
    assert len(saved) == len(original) + 2
    assert first["ticket_id"] != second["ticket_id"]
    assert saved[-2]["priority"] == "high"
    assert datetime.fromisoformat(saved[-1]["created_at"]).utcoffset().total_seconds() == 7 * 3600


def test_missing_catalog_reports_error(isolated_raw_data):
    (isolated_raw_data / "product_catalog.json").unlink()
    result = ToolCallingAgent().run("Xem xe điện VinFast")
    assert "Không thể tra cứu" in result["answer"]
    assert "không tìm thấy" not in result["answer"]


def test_corrupt_tickets_are_not_overwritten(isolated_raw_data):
    path = isolated_raw_data / "support_tickets.json"
    path.write_text("invalid JSON", encoding="utf-8")
    assert "error" in tools.submit_support_ticket("An", "Xe bị lỗi")
    assert path.read_text(encoding="utf-8") == "invalid JSON"


@pytest.mark.parametrize("query,budget", [
    ("Xe điện giá dưới 1,2 tỷ", 1_200_000_000),
    ("Resort giá dưới 5.5 triệu", 5_500_000),
    ("Xe điện giá dưới 600.000.000 đồng", 600_000_000),
])
def test_price_units(query, budget):
    assert ToolCallingAgent._extract_max_price(query) == budget


def test_low_priority_and_trace_reset():
    agent = ToolCallingAgent()
    first = agent.run("Tôi tên An, xe VinFast bị lỗi nhưng không gấp.")
    assert first["trace"][0]["arguments"]["priority"] == "low"
    second = agent.run("Chính sách bảo hành VinFast?")
    assert all("action" not in entry for entry in second["trace"])
    assert any("action" in entry for entry in first["trace"])


def test_scope_and_empty_input():
    agent = ToolCallingAgent()
    assert "chỉ hỗ trợ" in agent.run("Viết bài văn về mùa thu")["answer"]
    assert agent.run("  ")["status"] == "invalid_input"
