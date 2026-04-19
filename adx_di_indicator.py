"""
=============================================================
  ADX + DI INDICATOR
  Portado exactamente desde Pine Script v4 de BeikabuOyaji.
  Usa el suavizado de Wilder (≠ EMA estándar de Pandas).

  Función principal:
    calcular_adx_di(df, length=14, umbral=20) → dict

  Retorna:
    {
      "adx"       : float   — Fuerza de tendencia (0-100)
      "di_plus"   : float   — Presión compradora
      "di_minus"  : float   — Presión vendedora
      "trend_ok"  : bool    — True si ADX >= umbral
      "cruce_long"  : bool  — DI+ cruzó sobre DI- (última vela)
      "cruce_short" : bool  — DI- cruzó sobre DI+ (última vela)
      "dominio"   : str     — "BULL" | "BEAR" | "NEUTRAL"
    }
=============================================================
"""

import numpy as np


def calcular_adx_di(ohlcv: list, length: int = 14, umbral: int = 20) -> dict:
    """
    Calcula ADX, DI+ y DI- con suavizado Wilder, idéntico a Pine Script v4.

    Parámetros:
        ohlcv   : lista de velas [[ts, open, high, low, close, vol], ...]
                  mínimo length*3 velas para estabilizar el cálculo
        length  : período de suavizado Wilder (default 14)
        umbral  : nivel ADX para considerar tendencia fuerte (default 20)

    Retorna:
        dict con adx, di_plus, di_minus, trend_ok, cruce_long, cruce_short, dominio
    """
    # ── Extraer OHLC ───────────────────────────────────────────
    highs  = np.array([v[2] for v in ohlcv], dtype=float)
    lows   = np.array([v[3] for v in ohlcv], dtype=float)
    closes = np.array([v[4] for v in ohlcv], dtype=float)
    n      = len(highs)

    if n < length * 3:
        # No hay suficientes velas para un resultado estable
        return _resultado_neutral()

    # ── Calcular TR, DM+, DM- barra a barra ────────────────────
    tr     = np.zeros(n)
    dm_pos = np.zeros(n)
    dm_neg = np.zeros(n)

    for i in range(1, n):
        hl       = highs[i]  - lows[i]
        hc       = abs(highs[i]  - closes[i-1])
        lc       = abs(lows[i]   - closes[i-1])
        tr[i]    = max(hl, hc, lc)

        up   = highs[i]  - highs[i-1]
        down = lows[i-1] - lows[i]

        if up > down:
            dm_pos[i] = max(up, 0)
            dm_neg[i] = 0
        elif down > up:
            dm_pos[i] = 0
            dm_neg[i] = max(down, 0)
        else:
            dm_pos[i] = 0
            dm_neg[i] = 0

    # ── Suavizado Wilder ────────────────────────────────────────
    # Formula Wilder: S[i] = S[i-1] - S[i-1]/length + Value[i]
    # Equivalente a: S[i] = S[i-1] * (1 - 1/length) + Value[i]
    str_s = np.zeros(n)
    sdm_p = np.zeros(n)
    sdm_n = np.zeros(n)

    # Semilla inicial
    str_s[1] = tr[1]
    sdm_p[1] = dm_pos[1]
    sdm_n[1] = dm_neg[1]

    for i in range(2, n):
        str_s[i] = str_s[i-1] - (str_s[i-1] / length) + tr[i]
        sdm_p[i] = sdm_p[i-1] - (sdm_p[i-1] / length) + dm_pos[i]
        sdm_n[i] = sdm_n[i-1] - (sdm_n[i-1] / length) + dm_neg[i]

    # ── Calcular DI+ y DI- ─────────────────────────────────────
    di_p = np.zeros(n)
    di_m = np.zeros(n)
    dx   = np.zeros(n)

    for i in range(1, n):
        if str_s[i] != 0:
            di_p[i] = (sdm_p[i] / str_s[i]) * 100
            di_m[i] = (sdm_n[i] / str_s[i]) * 100

        denom = di_p[i] + di_m[i]
        if denom != 0:
            dx[i] = (abs(di_p[i] - di_m[i]) / denom) * 100

    # ── ADX = SMA(DX, length) ───────────────────────────────────
    # Usar solo los valores válidos para calcular la SMA final
    adx_series = np.zeros(n)
    for i in range(length, n):
        adx_series[i] = np.mean(dx[i-length+1:i+1])

    # ── Valores finales (última y penúltima vela) ───────────────
    adx_actual   = float(adx_series[-1])
    di_p_actual  = float(di_p[-1])
    di_m_actual  = float(di_m[-1])
    di_p_prev    = float(di_p[-2]) if n >= 2 else di_p_actual
    di_m_prev    = float(di_m[-2]) if n >= 2 else di_m_actual

    # ── Detectar cruces en la última barra ─────────────────────
    cruce_long  = (di_p_prev < di_m_prev) and (di_p_actual > di_m_actual)
    cruce_short = (di_m_prev < di_p_prev) and (di_m_actual > di_p_actual)

    # ── Dominio actual (sin necesitar cruce) ────────────────────
    if di_p_actual > di_m_actual:
        dominio = "BULL"
    elif di_m_actual > di_p_actual:
        dominio = "BEAR"
    else:
        dominio = "NEUTRAL"

    return {
        "adx"         : round(adx_actual,  2),
        "di_plus"     : round(di_p_actual, 2),
        "di_minus"    : round(di_m_actual, 2),
        "trend_ok"    : adx_actual >= umbral,
        "cruce_long"  : cruce_long,
        "cruce_short" : cruce_short,
        "dominio"     : dominio,
    }


def _resultado_neutral() -> dict:
    """Retorna resultado vacío cuando no hay suficientes velas."""
    return {
        "adx"         : 0.0,
        "di_plus"     : 0.0,
        "di_minus"    : 0.0,
        "trend_ok"    : False,
        "cruce_long"  : False,
        "cruce_short" : False,
        "dominio"     : "NEUTRAL",
    }
