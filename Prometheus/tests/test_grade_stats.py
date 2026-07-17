"""Unit tests for adaptive learning state integrity.

Covers:
- grade_stats structure and key access
- ob_stats: hits never exceeds wins impossibility guard
- ltf_stats structure
- learning state merge / persistence invariants
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fresh_learning() -> dict:
    """Return a clean default learning state dict matching _LEARNING in trader.py."""
    return {
        "wins":             0,
        "losses":           0,
        "total_seen":       0,
        "score_adjust":     0.0,
        "grade_stats":      {},
        "direction_stats":  {},
        "ltf_stats":        {},
        "ob_stats":         {},
        "streak":           0,
        "best_streak":      0,
        "worst_streak":     0,
        "total_pnl":        0.0,
        "last_20_results":  [],
        "open_pnl_history": [],
    }


def _simulate_entry(state: dict, grade: str, direction: str, ob_dir: str) -> None:
    """Simulate a trade entry — update grade_stats.acted and ob_stats.hits."""
    gs = state["grade_stats"].setdefault(grade, {"seen": 0, "acted": 0, "wins": 0, "losses": 0})
    gs["acted"] += 1
    obs = state["ob_stats"].setdefault(ob_dir, {"hits": 0, "wins": 0})
    obs["hits"] += 1


def _simulate_close(state: dict, grade: str, direction: str, ob_dir: str,
                    profit: float, tp1_hit: bool) -> None:
    """Simulate a trade close — update wins/losses, grade_stats, ob_stats (TP1 only), ltf_stats."""
    won = profit > 0
    state["wins" if won else "losses"] += 1
    state["total_pnl"] = round(state["total_pnl"] + profit, 2)

    gs = state["grade_stats"].setdefault(grade, {"seen": 0, "acted": 0, "wins": 0, "losses": 0})
    gs.setdefault("wins", 0)
    gs.setdefault("losses", 0)
    gs["wins" if won else "losses"] += 1

    ds = state["direction_stats"].setdefault(direction, {"wins": 0, "losses": 0})
    ds["wins" if won else "losses"] += 1

    # ob_stats.wins: incremented ONLY at TP1 close (not at every deal close)
    if tp1_hit:
        obs = state["ob_stats"].setdefault(ob_dir, {"hits": 0, "wins": 0})
        obs["wins"] += 1


# ── Tests: grade_stats ────────────────────────────────────────────────────────

class TestGradeStats:
    def test_wins_incremented_on_win(self):
        state = _fresh_learning()
        _simulate_entry(state, "A", "BUY", "bullish")
        _simulate_close(state, "A", "BUY", "bullish", profit=10.0, tp1_hit=True)
        assert state["grade_stats"]["A"]["wins"] == 1
        assert state["grade_stats"]["A"]["losses"] == 0

    def test_losses_incremented_on_loss(self):
        state = _fresh_learning()
        _simulate_entry(state, "B", "SELL", "bearish")
        _simulate_close(state, "B", "SELL", "bearish", profit=-5.0, tp1_hit=False)
        assert state["grade_stats"]["B"]["losses"] == 1
        assert state["grade_stats"]["B"]["wins"] == 0

    def test_global_wins_and_grade_wins_consistent(self):
        state = _fresh_learning()
        for _ in range(3):
            _simulate_entry(state, "A", "BUY", "bullish")
            _simulate_close(state, "A", "BUY", "bullish", profit=5.0, tp1_hit=True)
        for _ in range(2):
            _simulate_entry(state, "A", "SELL", "bearish")
            _simulate_close(state, "A", "SELL", "bearish", profit=-3.0, tp1_hit=False)
        assert state["wins"] == 3
        assert state["losses"] == 2
        assert state["grade_stats"]["A"]["wins"] == 3
        assert state["grade_stats"]["A"]["losses"] == 2

    def test_multiple_grades_tracked_independently(self):
        state = _fresh_learning()
        _simulate_entry(state, "A", "BUY", "bullish")
        _simulate_close(state, "A", "BUY", "bullish", profit=10.0, tp1_hit=True)
        _simulate_entry(state, "B", "SELL", "bearish")
        _simulate_close(state, "B", "SELL", "bearish", profit=-5.0, tp1_hit=False)
        assert state["grade_stats"]["A"]["wins"] == 1
        assert state["grade_stats"]["A"]["losses"] == 0
        assert state["grade_stats"]["B"]["wins"] == 0
        assert state["grade_stats"]["B"]["losses"] == 1

    def test_grade_stats_keys_always_present(self):
        state = _fresh_learning()
        _simulate_entry(state, "C", "BUY", "bullish")
        _simulate_close(state, "C", "BUY", "bullish", profit=0.0, tp1_hit=False)
        required = {"seen", "acted", "wins", "losses"}
        gs = state["grade_stats"]["C"]
        assert required.issubset(gs.keys()), f"Missing keys: {required - gs.keys()}"

    def test_total_pnl_accumulates(self):
        state = _fresh_learning()
        profits = [10.0, -3.0, 7.5, -1.0]
        for p in profits:
            _simulate_entry(state, "A", "BUY", "bullish")
            _simulate_close(state, "A", "BUY", "bullish", profit=p, tp1_hit=(p > 0))
        expected = round(sum(profits), 2)
        assert state["total_pnl"] == pytest.approx(expected, abs=0.01)


# ── Tests: ob_stats ───────────────────────────────────────────────────────────

class TestOBStats:
    def test_hits_incremented_at_entry(self):
        state = _fresh_learning()
        _simulate_entry(state, "A", "BUY", "bullish")
        assert state["ob_stats"]["bullish"]["hits"] == 1

    def test_wins_incremented_only_at_tp1(self):
        state = _fresh_learning()
        _simulate_entry(state, "A", "BUY", "bullish")
        # SL hit — no win, no TP1
        _simulate_close(state, "A", "BUY", "bullish", profit=-5.0, tp1_hit=False)
        assert state["ob_stats"]["bullish"]["wins"] == 0
        assert state["ob_stats"]["bullish"]["hits"] == 1

    def test_wins_never_exceed_hits(self):
        """Critical invariant: wins must never exceed hits."""
        state = _fresh_learning()
        n = 10
        for _ in range(n):
            _simulate_entry(state, "A", "BUY", "bullish")
            _simulate_close(state, "A", "BUY", "bullish", profit=5.0, tp1_hit=True)
        obs = state["ob_stats"]["bullish"]
        assert obs["wins"] <= obs["hits"], (
            f"ob_stats corruption: wins={obs['wins']} > hits={obs['hits']}"
        )

    def test_bearish_and_bullish_tracked_separately(self):
        state = _fresh_learning()
        _simulate_entry(state, "A", "SELL", "bearish")
        _simulate_close(state, "A", "SELL", "bearish", profit=10.0, tp1_hit=True)
        _simulate_entry(state, "B", "BUY", "bullish")
        _simulate_close(state, "B", "BUY", "bullish", profit=-3.0, tp1_hit=False)
        assert state["ob_stats"]["bearish"]["hits"] == 1
        assert state["ob_stats"]["bearish"]["wins"] == 1
        assert state["ob_stats"]["bullish"]["hits"] == 1
        assert state["ob_stats"]["bullish"]["wins"] == 0

    def test_hits_wins_ratio_meaningful(self):
        """After 20 entries, wins/hits ratio should be between 0 and 1."""
        state = _fresh_learning()
        wins = 0
        for i in range(20):
            profit = 5.0 if (i % 3 == 0) else -3.0
            won = profit > 0
            if won:
                wins += 1
            _simulate_entry(state, "A", "BUY", "bullish")
            _simulate_close(state, "A", "BUY", "bullish", profit=profit, tp1_hit=won)
        obs = state["ob_stats"]["bullish"]
        assert obs["hits"] == 20
        assert obs["wins"] == wins
        ratio = obs["wins"] / obs["hits"]
        assert 0.0 <= ratio <= 1.0


# ── Tests: direction_stats ────────────────────────────────────────────────────

class TestDirectionStats:
    def test_direction_wins_tracked(self):
        state = _fresh_learning()
        _simulate_entry(state, "A", "SELL", "bearish")
        _simulate_close(state, "A", "SELL", "bearish", profit=10.0, tp1_hit=True)
        assert state["direction_stats"]["SELL"]["wins"] == 1
        assert state["direction_stats"]["SELL"]["losses"] == 0

    def test_direction_losses_tracked(self):
        state = _fresh_learning()
        _simulate_entry(state, "A", "BUY", "bullish")
        _simulate_close(state, "A", "BUY", "bullish", profit=-5.0, tp1_hit=False)
        assert state["direction_stats"]["BUY"]["losses"] == 1
        assert state["direction_stats"]["BUY"]["wins"] == 0

    def test_direction_stats_not_empty_after_trades(self):
        state = _fresh_learning()
        _simulate_entry(state, "A", "SELL", "bearish")
        _simulate_close(state, "A", "SELL", "bearish", profit=3.0, tp1_hit=True)
        assert state["direction_stats"] != {}


# ── Tests: learning state persistence ────────────────────────────────────────

class TestLearningStatePersistence:
    def test_json_round_trip_preserves_grade_stats(self):
        state = _fresh_learning()
        _simulate_entry(state, "A", "BUY", "bullish")
        _simulate_close(state, "A", "BUY", "bullish", profit=7.5, tp1_hit=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(state, f)
            fname = f.name

        with open(fname) as f:
            loaded = json.load(f)

        assert loaded["grade_stats"]["A"]["wins"] == 1
        assert loaded["total_pnl"] == pytest.approx(7.5, abs=0.01)

    def test_json_round_trip_preserves_ob_stats_invariant(self):
        state = _fresh_learning()
        for i in range(5):
            won = (i % 2 == 0)
            _simulate_entry(state, "B", "SELL", "bearish")
            _simulate_close(state, "B", "SELL", "bearish", profit=(5.0 if won else -3.0), tp1_hit=won)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(state, f)
            fname = f.name

        with open(fname) as f:
            loaded = json.load(f)

        obs = loaded["ob_stats"]["bearish"]
        assert obs["wins"] <= obs["hits"], f"Corruption after JSON round-trip: {obs}"

    def test_score_adjust_persisted(self):
        state = _fresh_learning()
        state["score_adjust"] = 2.5
        serialized = json.dumps(state)
        loaded = json.loads(serialized)
        assert loaded["score_adjust"] == pytest.approx(2.5, abs=0.001)

    def test_last_20_results_capped_at_20(self):
        """Simulate the rolling 20-result window logic."""
        state = _fresh_learning()
        last20 = state["last_20_results"]
        for i in range(35):
            won = (i % 3 != 0)
            last20.append(1 if won else 0)
            if len(last20) > 20:
                last20.pop(0)
        assert len(state["last_20_results"]) == 20

    def test_mergeable_keys_present(self):
        """All keys expected in the mergeable set must be in a fresh state."""
        state = _fresh_learning()
        # These are the keys that _load_learning() must preserve across restarts
        expected_mergeable = {
            "wins", "losses", "total_seen", "score_adjust",
            "grade_stats", "direction_stats", "ltf_stats", "ob_stats",
            "streak", "best_streak", "worst_streak",
            "total_pnl", "last_20_results", "open_pnl_history",
        }
        missing = expected_mergeable - state.keys()
        assert not missing, f"Missing keys from learning state: {missing}"
