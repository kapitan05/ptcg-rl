"""Relabel ≡ extract on GBDT-played replays (arrays identical, agreement 1.0)."""

import json
import sys
from pathlib import Path

import numpy as np

a, b = Path(sys.argv[1]), Path(sys.argv[2])
ma, mb = (
    json.loads((a / "manifest.json").read_text()),
    json.loads((b / "manifest.json").read_text()),
)
print("agreement", mb["student_teacher_agreement"])
for split in ma["splits"]:
    for sa, sb in zip(ma["splits"][split]["shards"], mb["splits"][split]["shards"], strict=True):
        with np.load(a / sa["path"]) as x, np.load(b / sb["path"]) as y:
            for k in x.files:
                eq = (
                    np.array_equal(x[k], y[k])
                    if k != "scores"
                    else np.allclose(x[k], y[k], atol=1e-10, rtol=0)
                )
                print(split, sa["path"], k, "OK" if eq else "DIFF")
                assert eq, k
print("RELABEL SELF-TEST PASS")
