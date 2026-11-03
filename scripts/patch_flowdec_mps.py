# Copyright (c) 2026 AMLT course project (FlowDec review).
# Idempotent patch: make the official FlowDec backbone importable on Apple
# Silicon (MPS) and CPU, where the fused CUDA/C++ extensions cannot be built.
#
# The NCSN++ backbone loads two small C++/CUDA extensions at import time
# (`fused_bias_act` and `upfirdn2d`). They are only needed for the CUDA path:
# `upfirdn2d` already has a pure-PyTorch fallback (used for CPU) and the fused
# activation is never called by the config we use. We wrap the loads in
# try/except and route non-CUDA devices through the native implementation.
from __future__ import annotations

import sys
from pathlib import Path

from beartype import beartype

REPO_ROOT = Path(__file__).resolve().parents[1]
OP_DIR = REPO_ROOT / "code" / "FlowDec" / "flowdec" / "backbones" / "ncsnpp_utils" / "op"

MARK = "FLOWDEC_MPS_PATCH"

FUSED_OLD = """module_path = os.path.dirname(__file__)
fused = load(
    "fused",
    sources=[
        os.path.join(module_path, "fused_bias_act.cpp"),
        os.path.join(module_path, "fused_bias_act_kernel.cu"),
    ],
)"""

FUSED_NEW = f"""module_path = os.path.dirname(__file__)
# {MARK}: only build the CUDA extension when one is actually available.
try:
    fused = load(
        "fused",
        sources=[
            os.path.join(module_path, "fused_bias_act.cpp"),
            os.path.join(module_path, "fused_bias_act_kernel.cu"),
        ],
    )
except Exception:
    fused = None"""

UPFIRDN_OLD = """module_path = os.path.dirname(__file__)
upfirdn2d_op = load(
    "upfirdn2d",
    sources=[
        os.path.join(module_path, "upfirdn2d.cpp"),
        os.path.join(module_path, "upfirdn2d_kernel.cu"),
    ],
)"""

UPFIRDN_NEW = f"""module_path = os.path.dirname(__file__)
# {MARK}: only build the CUDA extension when one is actually available.
try:
    upfirdn2d_op = load(
        "upfirdn2d",
        sources=[
            os.path.join(module_path, "upfirdn2d.cpp"),
            os.path.join(module_path, "upfirdn2d_kernel.cu"),
        ],
    )
except Exception:
    upfirdn2d_op = None"""

UPFIRDN_FN_OLD = """def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if input.device.type == "cpu":"""

UPFIRDN_FN_NEW = f"""def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    # {MARK}: use the native implementation whenever the CUDA op is unavailable.
    if upfirdn2d_op is None or input.device.type != "cuda":"""


@beartype
def _patch(path: Path, old: str, new: str) -> str:
    text = path.read_text(encoding="utf-8")
    if MARK in text and new.splitlines()[1].strip() in text:
        return "already"
    if old not in text:
        return "pattern-missing"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    statuses = {
        "fused_act.py": _patch(OP_DIR / "fused_act.py", FUSED_OLD, FUSED_NEW),
        "upfirdn2d.py": _patch(OP_DIR / "upfirdn2d.py", UPFIRDN_OLD, UPFIRDN_NEW),
        "upfirdn2d.py:fn": _patch(OP_DIR / "upfirdn2d.py", UPFIRDN_FN_OLD, UPFIRDN_FN_NEW),
    }
    for name, status in statuses.items():
        print(f"{name}: {status}")
    if "pattern-missing" in statuses.values() and "already" not in statuses.values():
        sys.exit("patch failed: upstream file changed")


if __name__ == "__main__":
    main()