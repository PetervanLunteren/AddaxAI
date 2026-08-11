"""Tests for the standalone GPU facts probe script.

The script's torch import lives inside main(), so build_facts is
testable with a fake torch module and no torch install.
"""

import json
from types import SimpleNamespace

from app.ml.inference.gpu_probe import build_facts


def _fake_torch(
    available: bool = True,
    arch_list: list[str] | None = None,
    capabilities: list[tuple[int, int]] | None = None,
    names: list[str] | None = None,
):
    caps = capabilities or []
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_arch_list=lambda: arch_list or [],
            device_count=lambda: len(caps),
            get_device_name=lambda i: (names or [f"GPU {i}" for i in range(len(caps))])[i],
            get_device_capability=lambda i: caps[i],
        )
    )


def test_build_facts_shape():
    torch = _fake_torch(
        available=True,
        arch_list=["sm_70", "sm_120", "compute_120"],
        capabilities=[(6, 1)],
        names=["Quadro P4000"],
    )
    assert build_facts(torch) == {
        "cuda_available": True,
        "arch_list": ["sm_70", "sm_120", "compute_120"],
        "devices": [
            {"index": 0, "name": "Quadro P4000", "capability": [6, 1]},
        ],
    }


def test_build_facts_is_json_serialisable():
    torch = _fake_torch(
        available=True,
        arch_list=["sm_70"],
        capabilities=[(6, 1), (8, 6)],
    )
    facts = build_facts(torch)
    assert json.loads(json.dumps(facts)) == facts


def test_no_devices_when_count_zero():
    torch = _fake_torch(available=False)
    facts = build_facts(torch)
    assert facts["cuda_available"] is False
    assert facts["devices"] == []
