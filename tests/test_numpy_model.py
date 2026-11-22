"""NumPy submission model ≡ PyTorch ObservationTransformer (random weights and tokens)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
torch = pytest.importorskip("torch")

from agents.mega_lucario.transformer.model import ObservationTransformer  # noqa: E402
from agents.mega_lucario.transformer.numpy_model import NumpyTransformer  # noqa: E402


def _tokens(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.normal(size=(n, 198)).astype(np.float32)
    x[:, 0] = (rng.random(n) > 0.1).astype(np.float32)
    x[0, 0] = 1
    x[:, 1] = rng.integers(0, 5, n)
    x[:, 2] = (rng.random(n) > 0.5).astype(np.float32)
    x[1, 2] = 1  # at least one legal valid token
    x[1, 0] = 1
    x[:, 3] = (rng.random(n) > 0.8).astype(np.float32)
    x[:, 4:6] = rng.integers(-1, 32, (n, 2))
    return x


@pytest.mark.parametrize("n", [20, 57, 120])
def test_numpy_matches_torch(n: int) -> None:
    """Logits and value agree within 1e-5 (the C++ gate is 2e-5)."""
    torch.manual_seed(n)
    model = ObservationTransformer().eval()
    with torch.no_grad():
        for p in model.value2.parameters():
            p.normal_(0, 0.1)
    state = {k: v.numpy() for k, v in model.state_dict().items()}
    ours = NumpyTransformer(state)
    x = _tokens(np.random.default_rng(n), n)
    with torch.no_grad():
        logits, value = model.forward_with_value(
            torch.from_numpy(x)[None], torch.ones(1, n, dtype=torch.bool)
        )
    got, got_value = ours.head(ours.encode(x), x)
    want = logits[0].double().numpy()
    finite = np.isfinite(want)
    assert (np.isfinite(got) == finite).all()
    assert np.abs(got[finite] - want[finite]).max() < 1e-5
    assert abs(got_value - float(value[0])) < 1e-5
