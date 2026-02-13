import json
from pathlib import Path

import scripts.phase1_experiment_scope as scope


def test_scope_plan_includes_b0_in_gate2_and_b3_anchor():
    plan = scope.build_scope_plan()
    gate2 = next(g for g in plan["gates"] if g["name"] == "gate_2")
    assert "B0" in gate2["required_arms"]
    assert "B3" in gate2["optional_arms"]
    assert plan["fairness_note"]["secondary_anchor"].startswith("B3")


def test_scope_plan_marks_world_model_out_of_scope():
    plan = scope.build_scope_plan()
    phase = plan["phase1_scope"]
    assert phase["world_model_enabled"] is False
    assert "outer-loop world model capability" in phase["disallowed_claims"]


def test_scope_script_writes_json(tmp_path: Path):
    output = tmp_path / "scope.json"
    rc = scope.main(["--output", str(output)])
    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["phase1_scope"]["phase"] == "inner_loop_only"
