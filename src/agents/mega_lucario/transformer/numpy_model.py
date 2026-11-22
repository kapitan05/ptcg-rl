"""NumPy-only forward pass of `model.ObservationTransformer`, for the Kaggle submission.

Kaggle's agent sandbox is not guaranteed to have torch, so the submission ships weights as
`.npz` and runs this float64 port. Parity with PyTorch (and so with the audited C++ export)
is gated at 2e-5 by `scripts/submission_check.py`. The encoder is prefix-independent (it never
reads the legal/selected columns), so one decision encodes once and re-scores per selection.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

# Exact scalar erf, vectorized (n_tokens × 256 values per layer: cheap at this size).
_erf = np.frompyfunc(math.erf, 1, 1)


def _gelu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exact (erf) GELU, as torch.nn.functional.gelu's default."""
    return 0.5 * x * (1.0 + np.asarray(_erf(x / math.sqrt(2.0)), dtype=np.float64))


class NumpyTransformer:
    """Weights from a torch state_dict (`export_npz`), evaluated in float64."""

    def __init__(self, weights: dict[str, NDArray[np.float32]]) -> None:
        """Keep float64 copies of every parameter."""
        self.w = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
        self.support = np.linspace(-1.0, 1.0, 51)

    @classmethod
    def load(cls, path: Path) -> NumpyTransformer:
        """Load an `.npz` written by `export_npz`."""
        with np.load(path) as data:
            return cls({k: data[k] for k in data.files})

    def _lin(self, name: str, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x @ self.w[f"{name}.weight"].T + self.w[f"{name}.bias"]

    def _norm(self, name: str, x: NDArray[np.float64]) -> NDArray[np.float64]:
        mean = x.mean(-1, keepdims=True)
        var = ((x - mean) ** 2).mean(-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5) * self.w[f"{name}.weight"] + self.w[f"{name}.bias"]

    def _stem(self, name: str, x: NDArray[np.float64]) -> NDArray[np.float64]:
        h = self._lin(f"{name}.linear2", _gelu(self._lin(f"{name}.linear1", x)))
        return self._norm(f"{name}.norm", h)

    def encode(self, tokens: NDArray[np.float32]) -> NDArray[np.float64]:
        """Final-norm hidden states (n, 128) of one observation's tokens."""
        x = np.asarray(tokens, dtype=np.float64)
        n = len(x)
        if not 20 <= n <= 120 or x.shape[1] != 198:
            raise ValueError("Transformer supports 20–120 tokens of width 198")
        valid = x[:, 0] != 0
        kind = x[:, 1].astype(np.int64)
        table = self.w["deck_embedding.weight"]

        def embedding(slot: NDArray[np.float64]) -> NDArray[np.float64]:
            return table[np.maximum(slot, 0).astype(np.int64)] * (slot >= 0)[:, None]

        identity, extra = embedding(x[:, 4]), embedding(x[:, 5])
        counts, evolution = x[:, 102:134] @ table, x[:, 134:166] @ table
        generic, result = x[:, 6:102], x[:, 166:198]
        cat = np.concatenate
        h = self._stem("global_stem", cat([generic, identity, extra], 1)) * (kind == 0)[:, None]
        h = (
            h
            + self._stem("board_stem", cat([generic, identity, counts, evolution], 1))
            * (kind == 1)[:, None]
        )
        h = h + self._stem("zone_stem", cat([generic[:, :48], counts], 1)) * (kind == 2)[:, None]
        h = (
            h
            + (
                self._stem("option_stem", cat([generic, identity, result], 1))
                + self._lin("result_delta", result)
            )
            * (kind == 3)[:, None]
        )
        h = h + self.w["stop"] * (kind == 4)[:, None] + self.w["token_type.weight"][kind]
        for b in range(6):
            p = f"blocks.{b}"
            qkv = self._lin(f"{p}.qkv", self._norm(f"{p}.norm1", h)).reshape(n, 3, 4, 32)
            q, k, v = (qkv[:, i].transpose(1, 0, 2) for i in range(3))  # (4, n, 32)
            scores = q @ k.transpose(0, 2, 1) / math.sqrt(32.0)
            scores = np.where(valid[None, None, :], scores, -np.inf)
            scores = np.exp(scores - scores.max(-1, keepdims=True))
            attention = scores / scores.sum(-1, keepdims=True)
            h = h + self._lin(f"{p}.out", (attention @ v).transpose(1, 0, 2).reshape(n, 128))
            h = h + self._lin(f"{p}.ff2", _gelu(self._lin(f"{p}.ff1", self._norm(f"{p}.norm2", h))))
        return self._norm("norm", h)

    def head(
        self, h: NDArray[np.float64], tokens: NDArray[np.float32]
    ) -> tuple[NDArray[np.float64], float]:
        """Pointer logits (-inf where not legal) and expected Value for one selection prefix."""
        x = np.asarray(tokens, dtype=np.float64)
        valid = x[:, 0] != 0
        selected = x[:, 3] * valid
        summed = (h * selected[:, None]).sum(0)
        step = min(int(selected.sum()), 5)
        query = self._lin("policy_query", h[0] + summed + self.w["step_embedding.weight"][step])
        logits = (self._lin("policy_key", h) * query).sum(-1) / math.sqrt(128.0)
        logits = logits + self._lin("policy_bias", h)[:, 0]
        logits = np.where(valid & (x[:, 2] != 0), logits, -np.inf)
        z = self._lin("value2", _gelu(self._lin("value1", h[0])))
        z = np.exp(z - z.max())
        return logits, float((z / z.sum() * self.support).sum())


def export_npz(checkpoint: Path, output: Path) -> None:
    """Write a checkpoint's validated state_dict as float32 `.npz` (needs torch; build time)."""
    from agents.mega_lucario.gbdt.features import DECK_COUNTS
    from agents.mega_lucario.transformer.checkpoint import load_checkpoint
    from agents.mega_lucario.transformer.features import CONTRACT_ID, NAMES

    model, _ = load_checkpoint(checkpoint, CONTRACT_ID, DECK_COUNTS, NAMES)
    state = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    np.savez_compressed(output, **state)  # type: ignore[arg-type]


def decide(
    model: NumpyTransformer, obs: dict[str, Any], table: dict[int, dict[str, Any]]
) -> list[int]:
    """Greedy sequential selection, the same loop as C++ `TransformerPolicy::decide`.

    Per step: remaining legal options (+ STOP once minCount is met); a unique option needs no
    evaluation; otherwise pick the first option within the score tie tolerance of the best.
    """
    from agents.mega_lucario.gbdt.features import options_for
    from agents.mega_lucario.transformer.contract import SCORE_TIE_TOLERANCE
    from agents.mega_lucario.transformer.features import token_indices, tokenize, with_prefix

    action: list[int] = []
    tokens: NDArray[np.float32] | None = None
    hidden: NDArray[np.float64] | None = None
    while len(action) < obs["select"]["maxCount"]:
        options = options_for(obs, action)
        if not options:
            raise ValueError("No legal Transformer candidate")
        chosen = 0
        if len(options) > 1:
            if tokens is None or hidden is None:
                tokens = tokenize(obs, [], table)
                hidden = model.encode(tokens)
            logits, _ = model.head(hidden, with_prefix(tokens, obs, action))
            scores = logits[token_indices(obs, options)]
            best = scores.max()
            while scores[chosen] < best - SCORE_TIE_TOLERANCE:
                chosen += 1
        pick = options[chosen]
        if pick == -1:
            break
        action.append(pick)
    return action
