"""
=============================================================
  SQUEEZE MOMENTUM INDICATOR [LazyBear]
  Portado exactamente desde Pine Script v4.
  Combina Bollinger Bands (BB) y Keltner Channels (KC)
  con un histograma de regresión lineal.

  Retorna:
    {
      "val"      : float   — Valor actual del histograma (impulso)
      "val_prev" : float   — Valor previo del histograma
      "sqz_on"   : bool    — True si BB está dentro de KC (Compresión)
      "sqz_off"  : bool    — True si BB rompe KC (Explosión)
      "no_sqz"   : bool    — Ninguno de los anteriores
      "bcolor"   : str     — Color del histograma (lime, green, red, maroon)
      "momentum_bullish" : bool — Histograma > 0
      "momentum_accel"   : bool — Histograma acelerando a favor
    }
=============================================================
"""

import numpy as np

def calcular_squeeze_momentum(ohlcv: list,
                               bb_length: int = 20,
                               bb_mult: float = 2.0,
                               kc_length: int = 20,
                               kc_mult: float = 1.5,
                               use_true_range: bool = True) -> dict:
    
    highs  = np.array([v[2] for v in ohlcv], dtype=float)
    lows   = np.array([v[3] for v in ohlcv], dtype=float)
    closes = np.array([v[4] for v in ohlcv], dtype=float)
    n      = len(closes)

    if n < max(bb_length, kc_length) + 5:
        return _sqz_neutral()

    # ── BOLLINGER BANDS ────────────────────────────────────────
    basis    = _sma(closes, bb_length)
    std_dev  = _stdev(closes, bb_length)
    upper_bb = basis + bb_mult * std_dev
    lower_bb = basis - bb_mult * std_dev

    # ── KELTNER CHANNELS ──────────────────────────────────────
    ma = _sma(closes, kc_length)
    
    if use_true_range:
        # True Range: max(H-L, |H-Cprev|, |L-Cprev|)
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i]  - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        range_data = tr
    else:
        range_data = highs - lows

    range_ma = _sma(range_data, kc_length)
    upper_kc = ma + range_ma * kc_mult
    lower_kc = ma - range_ma * kc_mult

    # ── ESTADO DEL SQUEEZE ────────────────────────────────────
    sqz_on  = (lower_bb[-1] > lower_kc[-1]) and (upper_bb[-1] < upper_kc[-1])
    sqz_off = (lower_bb[-1] < lower_kc[-1]) and (upper_bb[-1] > upper_kc[-1])
    no_sqz  = not sqz_on and not sqz_off

    # ── MOMENTUM (val) = LinReg del delta de precio ───────────
    # Midpoint = avg(avg(highest_high, lowest_low), sma_close)
    highest_h = np.zeros(n)
    lowest_l  = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - kc_length + 1)
        highest_h[i] = np.max(highs[start_idx:i+1])
        lowest_l[i]  = np.min(lows[start_idx:i+1])

    midpoint = (((highest_h + lowest_l) / 2.0) + ma) / 2.0
    delta    = closes - midpoint

    # linreg(source, length, 0)
    val      = _linreg(delta, kc_length)
    val_prev = _linreg(delta[:-1], kc_length) if n > kc_length + 1 else 0.0

    # ── COLOR DEL HISTOGRAMA ──────────────────────────────────
    if val > 0:
        bcolor = "lime" if val > val_prev else "green"
    else:
        bcolor = "red"  if val < val_prev else "maroon"

    return {
        "val"              : round(val, 6),
        "val_prev"         : round(val_prev, 6),
        "sqz_on"           : sqz_on,
        "sqz_off"          : sqz_off,
        "no_sqz"           : no_sqz,
        "bcolor"           : bcolor,
        "momentum_bullish" : val > 0,
        "momentum_accel"   : val > val_prev if val > 0 else val < val_prev,
    }


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = np.mean(arr[i-period+1:i+1])
    return result

def _stdev(arr: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = np.std(arr[i-period+1:i+1], ddof=0)
    return result

def _linreg(arr: np.ndarray, period: int) -> float:
    if len(arr) < period:
        return 0.0
    y = arr[-period:]
    x = np.arange(period)
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0] * (period - 1) + coeffs[1])

def _sqz_neutral() -> dict:
    return {
        "val": 0.0, "val_prev": 0.0,
        "sqz_on": False, "sqz_off": False, "no_sqz": True,
        "bcolor": "gray", "momentum_bullish": False, "momentum_accel": False,
    }
