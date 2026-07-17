"""Vision package init — OpenCV is optional."""
try:
    from vision.chart_analyzer import ChartVisionAnalyzer, ChartVisionResult
    __all__ = ["ChartVisionAnalyzer", "ChartVisionResult"]
except ImportError:
    pass  # cv2 / PIL not installed; vision features unavailable

