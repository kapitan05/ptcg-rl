# B/01: history features (design; implementation after the Sprint 3 chain)

**Question:** representation (owner B). Does the recent event history (`obs.logs`, the events since the agent's last decision) improve distillation and arena strength?

## What the data allows (checked 21 Nov)
- Replays keep `observation.logs`: 118 of 132 decisions in a sample game have events, up to 64 per decision.
- **Parity constraint:** the C++ engine exposes to agents only `type`, `attackId`, `fromArea`, `toArea`, plus `playerIndex` for TurnEnd/Attack/MoveCard/MoveCardReverse (`src/engine/cpp/official_observation.h`). Hidden card IDs are never passed. Python features must read only these fields, or C++ ≡ Python parity fails.

## Design (smallest change that tests the question)
- **History summary in the Global token:** it uses 76 of its 96 generic slots, so 20 free slots hold the history. Counts per log type, plus own/opponent attack and visible move counts, sign-log scaled like the other generic features.
- **No architecture change:** same stems, same 120 tokens, so the C++ model code and the PPO stack are unchanged. Full per-event tokens (a new token kind and stem) are B/02, only if B/01 shows signal.
- **New token contract** (`lucario-entity-transformer-v3`), **as a second agent** `mega_lucario_transformer_h` in the arena. Replacing v2 in place would invalidate every current checkpoint and the running Sprint 3 jobs.
- **Gates:** Python tokenize ≡ C++ tokenize (exact), then the usual `validate.py` (2e-5).

## Experiment
- Re-extract the A/02 teacher replays with v3 tokens (needs the 56 GB archive unpacked, so it runs after the Sprint 3 PPO chain frees the disk), distil with the A/02 seeds, and rate against A/02 in one roster.
- **Pre-registered success:** validation KL lower than A/02 on all 3 seeds, and paired head-to-head vs. A/02 with LCB > 0.5 or BT not lower.
