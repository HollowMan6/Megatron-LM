# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Subprocess-isolated CPU regression checks for CSA2 RoPE table sizing."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[4] / "tools/check_csa2_rope_memory.py"


def test_old_and_current_rope_are_bitwise_equal_on_cpu():
    subprocess.run([sys.executable, str(SCRIPT), "--numerics-only"], check=True)


def test_rope_table_uses_host_known_maximum_on_cpu():
    subprocess.run([sys.executable, str(SCRIPT)], check=True)
