"""
ML Pattern Learning Engine
============================
Learns from historical trade setups to improve probability estimates over time.

Architecture:
  SetupRecord → feature extraction → XGBoost / LightGBM classifier
               → probability score → update confidence weights

Feature vector per setup:
  - structure_type (encoded)
  - trend_strength
  - mtf_alignment_score
  - sr_confidence (nearest)
  - candlestick_score
  - chart_pattern_confidence
  - fib_level_proximity (is near key fib level)
  - smc_ob_present (binary)
  - smc_stop_hunt (binary)
  - confluence_score
  → Target: outcome (1 = win, 0 = loss)

The engine also tracks per-pattern win rates to adaptively score patterns.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional imports — gracefully degrade
try:
    import xgboost as xgb             # type: ignore
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost not installed — using fallback classifier")

try:
    import lightgbm as lgb            # type: ignore
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed — ML features disabled")


# ─────────────────────────────────────────────
# Pattern-type taxonomy
# ─────────────────────────────────────────────

# Maps lowercase substrings of pattern names → pattern_type_id
#   0 = unknown
#   1 = continuation_bull  (flag, pennant, cup-and-handle — continue prior uptrend)
#   2 = continuation_bear  (bear flag, bear pennant — continue prior downtrend)
#   3 = reversal_bull      (double bottom, inv H&S — reverse prior downtrend)
#   4 = reversal_bear      (head and shoulders, double top — reverse prior uptrend)
#   5 = breakout_neutral   (symmetrical triangle, wedge — direction-agnostic breakout)
#
# Rationale: TrendSpider Learning Centre (public educational content):
#   "Volume should CONTRACT during consolidation and EXPAND on breakout."
#   "A bull flag forms during an uptrend — always confirm it aligns with the bigger trend."
#   Pattern type is critical for regime-specific reliability.
PATTERN_TYPE_MAP: Dict[str, int] = {
    # continuation bull
    "bull flag":            1,
    "bull pennant":         1,
    "cup and handle":       1,
    "cup & handle":         1,
    # continuation bear
    "bear flag":            2,
    "bear pennant":         2,
    # reversal bull
    "double bottom":        3,
    "inv head":             3,
    "inverse head":         3,
    # reversal bear
    "head and shoulders":   4,
    "head & shoulders":     4,
    "double top":           4,
    # breakout neutral
    "symmetrical triangle": 5,
    "ascending triangle":   5,
    "descending triangle":  5,
    "wedge":                5,
    "pennant":              1,   # generic pennant → bull continuation by default (overridden if "bear" in name)
    "flag":                 1,   # generic flag → bull continuation by default
    "channel":              5,
    "triangle":             5,
}

# Soft Bayesian prior win-rates per pattern type — used ONLY in _statistical_fallback()
# (overridden by the trained model once ≥50 labeled samples exist)
# Values derived from established technical analysis theory, not copyrighted data sources.
PATTERN_CATEGORY_PRIORS: Dict[int, float] = {
    0: 0.50,   # unknown — no signal
    1: 0.60,   # continuation_bull — strong with trend + volume
    2: 0.58,   # continuation_bear — same family, slight asymmetry
    3: 0.55,   # reversal_bull — higher risk; requires exhaustion of prior trend
    4: 0.55,   # reversal_bear — same
    5: 0.52,   # breakout_neutral — direction ambiguous; higher false-breakout rate
}


def classify_pattern_type(pattern_name: str) -> int:
    """
    Map a pattern name string to a pattern_type_id.

    Checks for "bear" first to correctly classify bear flags/pennants before
    the generic "flag"/"pennant" fallback assigns type 1.

    Returns:
        int: pattern_type_id in range 0–5.
    """
    name_lower = pattern_name.lower().strip()
    if not name_lower:
        return 0
    # Bear subtypes must be checked before generic flag/pennant
    if "bear" in name_lower:
        for key, tid in PATTERN_TYPE_MAP.items():
            if key in name_lower:
                return tid
        return 2   # generic bear pattern → continuation_bear
    for key, tid in PATTERN_TYPE_MAP.items():
        if key in name_lower:
            return tid
    return 0


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class SetupRecord:
    """A single analysed setup with its eventual outcome."""
    setup_id:           str
    asset:              str
    timeframe:          str
    timestamp:          str

    # Features
    structure_type:     int   = 0     # 0=undef, 1=bull, 2=bear, 3=side
    trend_strength:     float = 0.0
    mtf_score:          float = 0.0
    sr_confidence:      float = 0.0
    candlestick_score:  float = 0.0
    pattern_confidence: float = 0.0
    fib_proximity:      int   = 0     # 1 = near key fib level
    ob_present:         int   = 0     # 1 = unmitigated OB present
    stop_hunt:          int   = 0     # 1 = stop hunt detected
    confluence_score:   float = 0.0

    # Volume & pattern-type features (added for breakout reliability scoring)
    volume_ratio:       float = 1.0   # last-bar vol / 20-bar avg vol; >1.5 = confirmed breakout
    pattern_type_id:    int   = 0     # see PATTERN_TYPE_MAP: 0=unk,1=cont_bull,...,5=neutral
    prior_trend_aligned: int  = 0     # 1 if pattern is contextually valid in prior trend

    # Outcome (filled in after trade resolves)
    outcome:            Optional[int]   = None   # 1 = win, 0 = loss
    rr_achieved:        Optional[float] = None   # e.g. 2.3

    # Metadata
    entry_price:        Optional[float] = None
    sl_price:           Optional[float] = None
    tp_price:           Optional[float] = None
    exit_price:         Optional[float] = None

    # Olympus lineage metadata (additive, backward compatible)
    source_system:      str = "hermes"
    model_version_used: str = "0"
    feature_version:    str = "v1"
    strategy_version:   str = "v1"
    execution_type:     str = "simulated"
    dataset_generation: str = "gen1"


@dataclass
class PatternStats:
    pattern:    str
    wins:       int   = 0
    losses:     int   = 0

    @property
    def total(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.5


@dataclass
class MLPrediction:
    win_probability:     float = 0.5
    quality_score:       float = 0.5   # normalised 0-1 setup quality
    confidence:          str   = "low"
    model_used:          str   = "statistical_fallback"
    features_used:       List[str] = field(default_factory=list)
    feature_importances: Optional[Dict[str, float]] = None
    model_version:       str   = "0"


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class PatternLearner:
    """
    Trains and infers a setup-quality classifier.

    Usage::

        learner = PatternLearner(model_dir="models")
        learner.add_setup(setup_record)
        learner.train()
        prediction = learner.predict(setup_record)
    """

    FEATURE_COLS = [
        "structure_type",
        "trend_strength",
        "mtf_score",
        "sr_confidence",
        "candlestick_score",
        "pattern_confidence",
        "fib_proximity",
        "ob_present",
        "stop_hunt",
        "confluence_score",
        # New features (v2): volume confirmation + pattern context
        "volume_ratio",        # >1.5 on breakout = strong confirmation
        "pattern_type_id",     # continuation vs reversal vs neutral
        "prior_trend_aligned", # 1 if pattern is structurally valid given prior trend
    ]

    def __init__(
        self,
        model_dir:          str   = "models",
        model_type:         str   = "xgboost",
        min_samples_train:  int   = 50,
        test_size:          float = 0.2,
        n_estimators:       int   = 200,
        max_depth:          int   = 6,
        learning_rate:      float = 0.05,
    ) -> None:
        self.model_dir     = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_type    = model_type
        self.min_samples   = min_samples_train
        self.test_size     = test_size
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.lr            = learning_rate

        self.model         = None
        self.scaler        = None
        self.records:       List[SetupRecord] = []
        self._record_ids:   set[str] = set()
        self.pattern_stats: Dict[str, PatternStats] = {}
        self.model_version  = 0

        self._db_path      = self.model_dir / "setups.json"
        self._model_path   = self.model_dir / "ml_model.pkl"
        self._stats_path   = self.model_dir / "pattern_stats.json"

        self._load_records()
        self._load_model()
        self._load_stats()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_setup(self, record: SetupRecord) -> bool:
        """Add a setup unless its stable identity is already present."""
        if record.setup_id in self._record_ids:
            logger.debug("Duplicate setup ignored: %s", record.setup_id)
            return False
        self.records.append(record)
        self._record_ids.add(record.setup_id)
        self._save_records()
        logger.debug("Setup added: %s", record.setup_id)
        return True

    def update_outcome(
        self,
        setup_id: str,
        outcome:  int,
        rr:       Optional[float] = None,
        exit_price: Optional[float] = None,
    ) -> None:
        """Label a stored setup with its trade outcome."""
        for rec in self.records:
            if rec.setup_id == setup_id:
                rec.outcome   = outcome
                rec.rr_achieved = rr
                rec.exit_price  = exit_price
                self._save_records()
                logger.info("Outcome recorded: %s -> %s (RR %.2f)", setup_id, outcome, rr or 0)
                # Retrain if enough new labeled samples
                labeled = sum(1 for r in self.records if r.outcome is not None)
                if labeled % 20 == 0 and labeled >= self.min_samples:
                    self.train()
                return
        logger.warning("Setup not found: %s", setup_id)

    def train(self) -> Dict[str, float]:
        """
        Train the ML model on all labeled setups.

        Returns:
            dict with accuracy and roc_auc metrics.
        """
        if not SKLEARN_AVAILABLE and not XGB_AVAILABLE:
            logger.error("No ML library available — training skipped")
            return {}

        labeled = [r for r in self.records if r.outcome is not None]
        if len(labeled) < self.min_samples:
            logger.warning("Need %d labeled samples, have %d", self.min_samples, len(labeled))
            return {}

        df = pd.DataFrame([asdict(r) for r in labeled])
        X  = df[self.FEATURE_COLS].fillna(0.0).values
        y  = df["outcome"].astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        model, scaler = self._build_model()

        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else 0.5

        self.model   = model
        self.scaler  = scaler
        self.model_version += 1
        self._save_model()

        logger.info(
            "Model v%d trained on %d samples | Acc %.3f | AUC %.3f",
            self.model_version, len(labeled), acc, auc,
        )
        return {"accuracy": acc, "roc_auc": auc, "train_samples": len(labeled)}

    def predict(self, record: SetupRecord) -> MLPrediction:
        """
        Predict win probability for an unseen setup.

        Falls back to statistical win rates or confluence score if no model.
        """
        if self.model is None:
            return self._statistical_fallback(record)

        features = np.array([[getattr(record, col, 0.0) for col in self.FEATURE_COLS]])

        if self.scaler:
            features = self.scaler.transform(features)

        proba = float(self.model.predict_proba(features)[0, 1])

        conf = "high" if abs(proba - 0.5) > 0.25 else "medium" if abs(proba - 0.5) > 0.10 else "low"

        fi = self.feature_importance() if hasattr(self, "feature_importance") else None

        return MLPrediction(
            win_probability=round(proba, 4),
            quality_score=round(proba, 4),
            confidence=conf,
            model_used=self.model_type,
            features_used=self.FEATURE_COLS,
            feature_importances=fi,
            model_version=str(self.model_version),
        )

    def update_pattern_stats(self, pattern_name: str, won: bool) -> None:
        """Update per-pattern win/loss counts."""
        if pattern_name not in self.pattern_stats:
            self.pattern_stats[pattern_name] = PatternStats(pattern=pattern_name)
        stats = self.pattern_stats[pattern_name]
        if won:
            stats.wins += 1
        else:
            stats.losses += 1
        self._save_stats()

    def get_pattern_win_rate(self, pattern_name: str) -> float:
        """Return historical win rate for a named pattern (default 0.5)."""
        stats = self.pattern_stats.get(pattern_name)
        return stats.win_rate if stats else 0.5

    def get_all_stats(self) -> List[PatternStats]:
        return sorted(
            self.pattern_stats.values(), key=lambda s: s.total, reverse=True
        )

    def feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importances if model supports it."""
        if self.model is None:
            return None
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.FEATURE_COLS, self.model.feature_importances_))
        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_model(self) -> Tuple[Any, Any]:
        """Instantiate the best available model."""
        scaler = None

        if self.model_type == "xgboost" and XGB_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.lr,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
        elif self.model_type == "lightgbm" and LGB_AVAILABLE:
            model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.lr,
                verbose=-1,
            )
        elif SKLEARN_AVAILABLE:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, max_depth=self.max_depth)
        else:
            raise RuntimeError("No ML library available")

        if SKLEARN_AVAILABLE and isinstance(model, __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression if False else object):
            from sklearn.preprocessing import StandardScaler as _SS
            scaler = _SS()

        return model, scaler

    def _statistical_fallback(self, record: SetupRecord) -> MLPrediction:
        """
        5-factor weighted heuristic used when no trained model is available.

        Weights:
          confluence_score   0.40  — primary signal quality gate
          pattern_type prior 0.20  — Bayesian prior win-rate by pattern category
          volume_ratio       0.20  — breakout volume confirmation (mapped 0.8→0.0, 1.5→1.0)
          prior_trend_aligned 0.15 — structural validity of pattern in prior trend context
          stop_hunt          0.05  — smart-money marker (stop-hunt before reversal)
        """
        confluence_factor = min(1.0, record.confluence_score / 100.0)

        type_prior = PATTERN_CATEGORY_PRIORS.get(record.pattern_type_id, 0.50)

        # Map volume_ratio linearly: 0.8 → 0.0, 1.5 → 1.0, clamped
        vol_factor = min(1.0, max(0.0, (record.volume_ratio - 0.8) / 0.7))

        score = (
            confluence_factor          * 0.40
            + type_prior               * 0.20
            + vol_factor               * 0.20
            + float(record.prior_trend_aligned) * 0.15
            + float(record.stop_hunt)  * 0.05
        )
        score = round(min(1.0, max(0.0, score)), 4)
        conf  = "high" if score > 0.7 else "medium" if score > 0.5 else "low"
        return MLPrediction(
            win_probability=score,
            quality_score=score,
            confidence=conf,
            model_used="statistical_fallback",
            features_used=[
                "confluence_score", "pattern_type_id",
                "volume_ratio", "prior_trend_aligned", "stop_hunt",
            ],
            model_version="statistical_fallback",
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_records(self) -> None:
        data = [asdict(r) for r in self.records]
        self._atomic_write_text(self._db_path, json.dumps(data, indent=2, default=str))

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        """Replace a JSON artifact atomically so interruption cannot truncate it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _load_records(self) -> None:
        if self._db_path.exists():
            try:
                data = json.loads(self._db_path.read_text())
                # Filter dict keys to known SetupRecord fields for backward compatibility
                # (records saved before v2 won't have volume_ratio / pattern_type_id / prior_trend_aligned)
                from dataclasses import fields as _dc_fields
                valid_fields = {f.name for f in _dc_fields(SetupRecord)}
                self.records = [
                    SetupRecord(**{k: v for k, v in d.items() if k in valid_fields})
                    for d in data
                ]
                self._record_ids = {record.setup_id for record in self.records}
                logger.info("Loaded %d setup records", len(self.records))
            except Exception as e:
                logger.warning("Could not load setup records: %s", e)

    def _save_model(self) -> None:
        with open(self._model_path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler, "version": self.model_version}, f)

    def _load_model(self) -> None:
        if self._model_path.exists():
            try:
                with open(self._model_path, "rb") as f:
                    data = pickle.load(f)
                self.model         = data.get("model")
                self.scaler        = data.get("scaler")
                self.model_version = data.get("version", 0)
                logger.info("Loaded ML model v%d", self.model_version)
            except Exception as e:
                logger.warning("Could not load ML model: %s", e)

    def _save_stats(self) -> None:
        data = {k: asdict(v) for k, v in self.pattern_stats.items()}
        self._atomic_write_text(self._stats_path, json.dumps(data, indent=2))

    def _load_stats(self) -> None:
        if self._stats_path.exists():
            try:
                data = json.loads(self._stats_path.read_text())
                self.pattern_stats = {k: PatternStats(**v) for k, v in data.items()}
            except Exception as e:
                logger.warning("Could not load pattern stats: %s", e)
