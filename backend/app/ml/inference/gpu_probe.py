"""Standalone GPU facts probe — runs as subprocess in env-addaxai-base.

The bundled torch cu128 wheels only contain kernels for compute
capability 7.0 to 12.0. On an older GPU (e.g. Pascal, 6.1) torch still
reports ``cuda_available=True`` and only warns, after which every kernel
launch fails. The backend process has no torch, so this script reports
the raw CUDA facts and all decision logic lives in ``app/ml/gpu_guard.py``.

Contract: exactly one JSON line on stdout, exit 0. No try/except here: a
torch failure tracebacks to stderr and exits non-zero, and the guard
treats that as fail-open. torch's own capability warning goes to stderr,
so stdout stays a single JSON line.
"""

import json


def build_facts(torch_module) -> dict:
    """Raw CUDA facts from a torch module. No decision logic."""
    devices = []
    for i in range(torch_module.cuda.device_count()):
        major, minor = torch_module.cuda.get_device_capability(i)
        devices.append(
            {
                "index": i,
                "name": torch_module.cuda.get_device_name(i),
                "capability": [major, minor],
            }
        )
    return {
        "cuda_available": torch_module.cuda.is_available(),
        "arch_list": list(torch_module.cuda.get_arch_list()),
        "devices": devices,
    }


def main() -> None:
    import torch  # deferred so the module imports without torch (tests)

    print(json.dumps(build_facts(torch)))


if __name__ == "__main__":
    main()
