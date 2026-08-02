"""Rolling context for the orchestrator.

Feeds the SLOW timescale of the nested trust monitor: an injection that looks
unremarkable in a single cycle often shows up as a drift in what is being asked
for over an episode.
"""

from __future__ import annotations

from collections import deque

__all__ = ["RollingMemory"]


class RollingMemory:
    def __init__(self, window: int = 32):
        self.window = window
        self._entries: deque = deque(maxlen=window)

    def add(self, step: int, opcode: str, note: str | None, action: str, glucose: float) -> None:
        self._entries.append({"step": step, "opcode": opcode, "note": note,
                              "action": action, "glucose": glucose})

    def reset(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[dict]:
        return list(self._entries)

    def recent_notes(self, n: int = 8) -> list[str]:
        return [e["note"] for e in list(self._entries)[-n:] if e["note"]]

    def action_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self._entries:
            out[e["action"]] = out.get(e["action"], 0) + 1
        return out
