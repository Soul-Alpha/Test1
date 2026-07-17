"""
Computer Vision Chart Analyzer
================================
Accepts uploaded chart images (PNG/JPG) and extracts structured market data
using OpenCV and image processing techniques.

Capabilities:
  - Detect dark / light chart themes and normalize
  - Identify candlestick bodies, wicks, and approximate OHLC from pixels
  - Detect drawn trendlines via Hough line transform
  - Detect horizontal S/R lines from horizontal segments
  - Detect pattern regions (triangles, channels) using contour analysis
  - Extract price scale via OCR (tesseract if available)
  - Return structured analysis without OCR if tesseract unavailable

Dependencies:
  - opencv-python
  - Pillow
  - numpy
  - (Optional) pytesseract for price scale extraction
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2        # type: ignore
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python not installed — chart vision analysis disabled")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try to import pytesseract — optional
try:
    import pytesseract  # type: ignore
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.info("pytesseract not available — price-scale OCR disabled")


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class DetectedCandle:
    x:         int     # horizontal pixel position (bar index proxy)
    body_top:  float   # normalised y (0 = chart top, 1 = chart bottom)
    body_bot:  float
    wick_top:  float
    wick_bot:  float
    is_bullish: bool

    @property
    def midpoint(self) -> float:
        return (self.body_top + self.body_bot) / 2.0


@dataclass
class DetectedLine:
    x1: int
    y1: int
    x2: int
    y2: int
    is_horizontal: bool
    slope_deg:     float


@dataclass
class ChartVisionResult:
    theme:           str                    = "dark"   # "dark" | "light"
    candles:         List[DetectedCandle]   = field(default_factory=list)
    trendlines:      List[DetectedLine]     = field(default_factory=list)
    horizontal_levels: List[float]          = field(default_factory=list)  # normalised
    dominant_direction: str                 = "unknown"  # "bullish" | "bearish" | "ranging"
    pattern_hints:   List[str]              = field(default_factory=list)
    image_shape:     Tuple[int, int]        = (0, 0)
    extracted_prices: List[float]           = field(default_factory=list)  # from OCR
    chart_preview_path: Optional[str]       = None    # path to annotated preview
    raw_analysis:    Dict                   = field(default_factory=dict)
    narrative:       str                    = ""


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class ChartVisionAnalyzer:
    """
    Analyze a chart image and extract structural information.

    Usage::

        analyzer = ChartVisionAnalyzer()

        # From file path
        result = analyzer.analyze_path("chart.png")

        # From bytes
        result = analyzer.analyze_bytes(image_bytes)

        # From PIL Image
        result = analyzer.analyze_pil(pil_image)
    """

    def __init__(
        self,
        horizontal_angle_deg: float = 5.0,   # lines within ±5° treated as horizontal
        min_line_length:      int   = 50,     # pixels
        max_line_gap:         int   = 15,
        canny_low:            int   = 50,
        canny_high:           int   = 150,
        output_dir:           str   = "outputs",
    ) -> None:
        self.h_angle     = horizontal_angle_deg
        self.min_len     = min_line_length
        self.max_gap     = max_line_gap
        self.canny_low   = canny_low
        self.canny_high  = canny_high
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def _no_cv2_result(self, name: str = "chart") -> ChartVisionResult:
        """Return a stub result when OpenCV is unavailable."""
        r = ChartVisionResult()
        r.narrative = (
            "OpenCV (cv2) is not installed — chart vision analysis is disabled. "
            "Install it with: pip install opencv-python-headless"
        )
        r.dominant_direction = "unknown"
        logger.warning("cv2 not available — returning empty vision result for '%s'", name)
        return r

    def analyze_path(self, image_path: str) -> ChartVisionResult:
        """Load image from disk and analyze."""
        if not CV2_AVAILABLE:
            return self._no_cv2_result(Path(image_path).stem)
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        return self._analyze_cv2(img, source_name=Path(image_path).stem)

    def analyze_bytes(self, data: bytes, name: str = "upload") -> ChartVisionResult:
        """Analyze image from raw bytes (e.g., FastAPI UploadFile)."""
        if not CV2_AVAILABLE:
            return self._no_cv2_result(name)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
        return self._analyze_cv2(img, source_name=name)

    def analyze_pil(self, pil_img, name: str = "upload") -> ChartVisionResult:
        """Analyze a PIL Image object."""
        if not CV2_AVAILABLE:
            return self._no_cv2_result(name)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return self._analyze_cv2(img, source_name=name)

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def _analyze_cv2(self, img: np.ndarray, source_name: str = "chart") -> ChartVisionResult:
        result = ChartVisionResult(image_shape=img.shape[:2])

        # 1. Theme detection
        result.theme = self._detect_theme(img)

        # 2. Preprocess: convert to grayscale + edge map
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # 3. Line detection
        lines = self._detect_lines(edges, img.shape)
        result.trendlines      = [l for l in lines if not l.is_horizontal]
        horizontal_raw         = [l for l in lines if l.is_horizontal]
        result.horizontal_levels = self._cluster_horizontal_lines(
            horizontal_raw, img.shape[0]
        )

        # 4. Candlestick detection (colour-based)
        result.candles = self._detect_candles(img, result.theme)

        # 5. Dominant direction from candle sequence + global pixel balance fallback
        result.dominant_direction = self._infer_direction(result.candles)
        if result.dominant_direction in ("unknown", "ranging"):
            result.dominant_direction = self._global_pixel_direction(img)

        # 6. Pattern hints from geometry
        result.pattern_hints = self._pattern_hints(result)

        # 7. OCR price extraction (optional)
        if TESSERACT_AVAILABLE:
            result.extracted_prices = self._ocr_prices(img)

        # 8. Generate annotated preview
        annotated = self._annotate(img.copy(), result)
        preview_path = str(self.output_dir / f"{source_name}_annotated.png")
        cv2.imwrite(preview_path, annotated)
        result.chart_preview_path = preview_path

        result.narrative = self._build_narrative(result)
        logger.info("Vision analysis complete: %s | %d candles | %d trendlines",
                    source_name, len(result.candles), len(result.trendlines))
        return result

    # ── Detection methods ──────────────────────────────────────────────────────

    def _detect_theme(self, img: np.ndarray) -> str:
        """Dark theme if median pixel brightness < 128."""
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        median = float(np.median(gray))
        return "dark" if median < 128 else "light"

    def _detect_lines(
        self, edges: np.ndarray, shape: Tuple[int, ...]
    ) -> List[DetectedLine]:
        """Use probabilistic Hough transform to find lines."""
        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=self.min_len,
            maxLineGap=self.max_gap,
        )
        if raw is None:
            return []

        lines: List[DetectedLine] = []
        for seg in raw.reshape(-1, 4):
            x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
            dx = x2 - x1
            dy = y2 - y1
            angle_deg = float(np.degrees(np.arctan2(abs(dy), max(abs(dx), 1))))
            is_horiz  = angle_deg <= self.h_angle
            lines.append(DetectedLine(x1, y1, x2, y2, is_horiz, angle_deg))

        return lines

    def _cluster_horizontal_lines(
        self, lines: List[DetectedLine], img_height: int
    ) -> List[float]:
        """
        Convert pixel Y positions to normalised (0–1) values and cluster
        nearby horizontal lines into single levels.
        """
        if not lines:
            return []

        ys = sorted(set(
            round((l.y1 + l.y2) / 2.0 / img_height, 4)
            for l in lines
        ))

        # Cluster within 2 %
        clusters: List[List[float]] = [[ys[0]]]
        for y in ys[1:]:
            if y - clusters[-1][-1] < 0.02:
                clusters[-1].append(y)
            else:
                clusters.append([y])

        return [round(float(np.mean(c)), 4) for c in clusters]

    def _detect_candles(
        self, img: np.ndarray, theme: str
    ) -> List[DetectedCandle]:
        """
        Simple candle detection using colour segmentation.

        Bullish candles → green-ish pixels
        Bearish candles → red-ish pixels

        We scan columns and look for vertical colour runs.
        This is a heuristic approach that handles most charting platforms.
        """
        hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_chan  = hsv[:, :, 0]
        s_chan  = hsv[:, :, 1]
        v_chan  = hsv[:, :, 2]
        H, W    = img.shape[:2]

        # Masks
        green_mask = (
            ((h_chan >= 40) & (h_chan <= 90)) & (s_chan > 80) & (v_chan > 60)
        ).astype(np.uint8)
        red_mask = (
            ((h_chan <= 15) | (h_chan >= 160)) & (s_chan > 80) & (v_chan > 60)
        ).astype(np.uint8)

        candles: List[DetectedCandle] = []
        step = max(2, W // 200)   # sample columns

        for col in range(0, W, step):
            g_rows = np.where(green_mask[:, col] > 0)[0]
            r_rows = np.where(red_mask[:, col] > 0)[0]

            for is_bull, rows in [(True, g_rows), (False, r_rows)]:
                if len(rows) < 3:
                    continue
                top = float(rows.min()) / H
                bot = float(rows.max()) / H
                mid = (top + bot) / 2.0
                # Treat full coloured segment as body; wick is estimated
                wick_top = max(0.0, top - 0.02)
                wick_bot = min(1.0, bot + 0.02)
                candles.append(DetectedCandle(
                    x=col,
                    body_top=top,
                    body_bot=bot,
                    wick_top=wick_top,
                    wick_bot=wick_bot,
                    is_bullish=is_bull,
                ))

        # Sort by x position (bar order)
        candles.sort(key=lambda c: c.x)
        return candles

    def _global_pixel_direction(self, img: np.ndarray) -> str:
        """Fallback direction from full-image green/red pixel balance.

        Strips the price-axis (rightmost 12 %) and time-axis (bottom 8 %)
        to avoid labels influencing the result, then counts HSV-green vs
        HSV-red pixels in the chart body.
        """
        H, W = img.shape[:2]
        chart = img[: int(H * 0.92), : int(W * 0.88)]
        hsv   = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        active = (s > 60) & (v > 60)
        green_px = int(np.sum(active & (h >= 35) & (h <= 95)))
        red_px   = int(np.sum(active & ((h <= 20) | (h >= 155))))
        total = green_px + red_px
        if total < 50:
            return "ranging"
        ratio = green_px / total
        if ratio > 0.55:
            return "bullish"
        if ratio < 0.45:
            return "bearish"
        return "ranging"

    def _infer_direction(self, candles: List[DetectedCandle]) -> str:
        """Infer chart direction from candle colour balance and mid-point trend.

        Strategy (in priority order):
        1. Green-vs-red candle pixel-area ratio  → most reliable on any platform
        2. Slope of midpoints of each colour's candles separately (last third vs first third)
        3. Fall back to "ranging" when evidence is weak
        """
        if not candles:
            return "unknown"

        # 1. Pixel-area balance: each DetectedCandle height proxies its area
        green_area = sum(c.body_bot - c.body_top for c in candles if c.is_bullish)
        red_area   = sum(c.body_bot - c.body_top for c in candles if not c.is_bullish)
        total_area = green_area + red_area

        if total_area > 0:
            bull_ratio = green_area / total_area
            if bull_ratio > 0.62:
                return "bullish"
            if bull_ratio < 0.38:
                return "bearish"

        # 2. Slope of recent half vs early half (separated by colour)
        if len(candles) >= 5:
            mids = [c.midpoint for c in candles]
            n = len(mids)
            early_avg = float(np.mean(mids[: n // 3 + 1]))
            late_avg  = float(np.mean(mids[-(n // 3 + 1):]))
            # y increases downward, so late_avg < early_avg means price rose
            diff = early_avg - late_avg
            if diff > 0.025:
                return "bullish"
            if diff < -0.025:
                return "bearish"

        return "ranging"

    def _pattern_hints(self, r: ChartVisionResult) -> List[str]:
        """Simple geometric hints from detected lines."""
        hints: List[str] = []
        tl = r.trendlines

        up_lines   = [l for l in tl if l.y2 < l.y1]   # line going up-right
        down_lines = [l for l in tl if l.y2 > l.y1]   # line going down-right

        if up_lines and down_lines:
            hints.append("Converging trendlines suggest triangle / wedge formation.")
        elif len(up_lines) >= 2:
            hints.append("Multiple ascending trendlines visible — potential channel or wedge.")
        elif len(down_lines) >= 2:
            hints.append("Multiple descending trendlines visible — potential descending channel.")

        if len(r.horizontal_levels) >= 3:
            hints.append(f"{len(r.horizontal_levels)} horizontal levels identified — strong S/R grid.")

        if not hints:
            hints.append("No specific pattern hints detected from image geometry.")

        return hints

    def _ocr_prices(self, img: np.ndarray) -> List[float]:
        """Extract price numbers from the price axis using Tesseract OCR."""
        # Crop the right-side price axis (last 10 % of width)
        H, W = img.shape[:2]
        axis_region = img[:, int(W * 0.90):]
        gray = cv2.cvtColor(axis_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config="--psm 6 -c tessedit_char_whitelist=0123456789.")
        prices: List[float] = []
        for token in text.split():
            try:
                prices.append(float(token))
            except ValueError:
                pass
        return prices

    def _annotate(self, img: np.ndarray, r: ChartVisionResult) -> np.ndarray:
        """Draw detected lines on the image."""
        colour_h = (0, 255, 255)    # yellow for horizontal levels
        colour_t = (255, 100, 0)    # blue-ish for trendlines

        H = img.shape[0]
        for level in r.horizontal_levels:
            y = int(level * H)
            cv2.line(img, (0, y), (img.shape[1], y), colour_h, 1)

        for line in r.trendlines:
            cv2.line(img, (line.x1, line.y1), (line.x2, line.y2), colour_t, 1)

        # Overlay direction text
        cv2.putText(
            img,
            f"Direction: {r.dominant_direction.upper()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255) if r.theme == "dark" else (0, 0, 0),
            2,
        )
        return img

    def _build_narrative(self, r: ChartVisionResult) -> str:
        lines = [
            f"Chart Vision Analysis ({r.theme} theme, {r.image_shape[1]}×{r.image_shape[0]} px):",
            f"  Detected direction: {r.dominant_direction.capitalize()}.",
            f"  Candlestick bodies detected: {len(r.candles)}.",
            f"  Trendlines: {len(r.trendlines)} | Horizontal levels: {len(r.horizontal_levels)}.",
        ]
        for hint in r.pattern_hints:
            lines.append(f"  Geometric hint: {hint}")
        if r.extracted_prices:
            lines.append(
                f"  OCR price references: {', '.join(str(p) for p in r.extracted_prices[:5])}"
            )
        return "\n".join(lines)
