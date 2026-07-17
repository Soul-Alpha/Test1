"""Quick smoke test for all core engines."""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data.sample_data import generate_xauusd_ohlcv
from engines.market_structure import MarketStructureEngine
from engines.support_resistance import SupportResistanceEngine
from engines.candlestick_engine import CandlestickEngine
from engines.fibonacci_engine import FibonacciEngine
from engines.liquidity_smc import LiquiditySMCEngine
from engines.chart_patterns import ChartPatternEngine
from engines.multi_timeframe import MultiTimeframeEngine
from analysis.confluence_scorer import ConfluenceScorer
from analysis.ai_reasoning import AIReasoningEngine

df = generate_xauusd_ohlcv(300)
print(f"Data: {len(df)} bars, price range {df['low'].min():.2f}-{df['high'].max():.2f}")

ms  = MarketStructureEngine().analyze(df)
print(f"MS:  {ms.structure_type.name} | strength={ms.trend_strength:.0%} | BOS={len(ms.bos_events)} | CHoCH={len(ms.choch_events)}")

sr  = SupportResistanceEngine().analyze(df)
print(f"SR:  {len(sr.support_zones)} support zones, {len(sr.resistance_zones)} resistance zones")

cs  = CandlestickEngine().analyze(df)
print(f"CS:  {len(cs.top_signals)} signals")

fib = FibonacciEngine().analyze(df)
print(f"Fib: {len(fib.levels)} levels")

smc = LiquiditySMCEngine().analyze(df)
print(f"SMC: OB={len(smc.order_blocks)} FVG={len(smc.fair_value_gaps)} pools={len(smc.liquidity_pools)}")

sh = ms.swing_highs; sl = ms.swing_lows
pat = ChartPatternEngine().analyze(df, sh, sl)
print(f"Pat: {len(pat.patterns)} patterns detected")

mtf = MultiTimeframeEngine().analyze({"4H": df})
print(f"MTF: bias={mtf.primary_bias} alignment={mtf.alignment_score:.2f}")

confluence = ConfluenceScorer().score(ms=ms, sr=sr, cs=cs, pat=pat, fib=fib, smc=smc, mtf=mtf)
print(f"Confluence: {confluence.total:.1f}/100  Grade={confluence.grade}  Dir={confluence.direction}")

current_price = float(df['close'].iloc[-1])
report = AIReasoningEngine().generate(
    asset="XAUUSD", timeframe="4H", current_price=current_price,
    ms=ms, sr=sr, cs=cs, pat=pat, fib=fib, smc=smc, mtf=mtf, confluence=confluence
)
print(f"Report: {len(report.full_text)} chars")
print("\n--- REPORT EXCERPT ---")
print(report.full_text[:500])
print("\n=== ALL ENGINES OK ===")
