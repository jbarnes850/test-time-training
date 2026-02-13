from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.path_utils import repo_root


PHASE1_SCOPE = {
    "phase": "inner_loop_only",
    "world_model_enabled": False,
    "primary_business_claim": "adaptive curriculum is compute-optimal vs static/random curricula under matched compute",
    "allowed_claims": [
        "adaptive curriculum improves sample efficiency",
        "adaptive curriculum improves cross-level generalization",
    ],
    "disallowed_claims": [
        "outer-loop world model capability",
        "full possibility frontier modeling",
    ],
}


def build_scope_plan() -> dict[str, Any]:
    arms = {
        "B0": {
            "role": "matched_compute_baseline",
            "description": "Train from base model initialization (no RLVR sampler init).",
            "matched_compute": True,
        },
        "B1": {
            "role": "matched_compute_baseline",
            "description": "Random curriculum with uniform level pool.",
            "matched_compute": True,
        },
        "B2": {
            "role": "matched_compute_baseline",
            "description": "Static easy-to-hard curriculum with uniform level pool.",
            "matched_compute": True,
        },
        "B3": {
            "role": "status_quo_anchor",
            "description": "Existing fixed-distribution RLVR checkpoint with no additional training.",
            "matched_compute": False,
        },
        "C": {
            "role": "matched_compute_treatment",
            "description": "Adaptive teacher+mutator curriculum.",
            "matched_compute": True,
        },
    }

    gates = [
        {
            "name": "gate_1",
            "epochs": 3,
            "seeds": [42],
            "required_arms": ["B1", "B2", "C"],
            "optional_arms": ["B3"],
            "go_criteria": [
                "C_L2_fast_1 > max(B1_L2_fast_1, B2_L2_fast_1) + 0.05",
                "C_frontier_size_delta > 0",
            ],
            "stop_criteria": [
                "C_L2_fast_1 < B1_L2_fast_1",
            ],
        },
        {
            "name": "gate_2",
            "epochs": 10,
            "seeds": [42],
            "required_arms": ["B0", "B1", "B2", "C"],
            "optional_arms": ["B3"],
            "go_criteria": [
                "C_L2_fast_1 > max(B1_L2_fast_1, B2_L2_fast_1) + 0.10",
                "C_L3_fast_1 > 0.0",
            ],
            "stop_criteria": [
                "C_L2_fast_1 <= max(B1_L2_fast_1, B2_L2_fast_1)",
                "effective_tasks < 4 for >=2 consecutive epochs",
            ],
        },
        {
            "name": "gate_3",
            "epochs": 10,
            "seeds": [42, 123, 999],
            "required_arms": ["B0", "B1", "B2", "C"],
            "optional_arms": ["B3"],
            "publish_criteria": [
                "C > max(B1,B2) on L2 fast_1 with sign-test p < 0.05",
            ],
        },
    ]

    return {
        "phase1_scope": PHASE1_SCOPE,
        "arms": arms,
        "fairness_note": {
            "primary_comparison": "C vs B1/B2 under matched compute",
            "secondary_anchor": "B3 as status-quo anchor; not primary matched-compute comparator",
        },
        "gates": gates,
    }


def _resolve_output_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root() / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="artifacts/phase1_experiment_scope.json")
    args = parser.parse_args(argv)

    plan = build_scope_plan()
    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2) + "\n")

    print(f"Wrote Phase 1 scope plan: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
