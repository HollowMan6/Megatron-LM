# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Subprocess-isolated CPU checks against the pre-streaming CSA2 oracle."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[4] / "tools/check_csa2_candidate_streaming.py"


def test_candidate_streaming_matches_frozen_cpu_oracle():
    subprocess.run([sys.executable, str(SCRIPT), "--oracle-only"], check=True)


def test_fused_candidate_streaming_and_k1_match_frozen_cpu_oracle():
    subprocess.run([sys.executable, str(SCRIPT), "--fused-only"], check=True)
