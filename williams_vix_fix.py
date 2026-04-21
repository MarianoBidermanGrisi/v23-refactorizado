import numpy as np
import pandas as pd

def calcular_wvf(ohlcv_confirmadas, pd_period=22, bbl=20, mult=2.0, lb=50, ph=0.85, pl=1.01):
    """
    Implementación del CM Williams Vix Fix en Python.
    
    Argumentos:
        ohlcv_confirmadas: Lista o array de velas [ts, o, h, l, c, v]
        pd_period: Periodo de Lookback para el cálculo base (Default 22)
        bbl: Longitud de las Bandas de Bollinger (Default 20)
        mult: Multiplicador de Desviación Estándar (Default 2.0)
        lb: Periodo de Lookback para Percentiles (Default 50)
        ph: Factor para Percentil Alto (Default 0.85)
        pl: Factor para Percentil Bajo (Default 1.01)
        
    Retorna:
        dict con los valores actuales y estados de señal.
    """
    if len(ohlcv_confirmadas) < max(pd_period, bbl, lb):
        return None

    # Extraer arrays
    highs = np.array([v[2] for v in ohlcv_confirmadas], dtype=float)
    lows = np.array([v[3] for v in ohlcv_confirmadas], dtype=float)
    closes = np.array([v[4] for v in ohlcv_confirmadas], dtype=float)

    # 1. Calcular WVF Base
    # wvf = ((highest(close, pd)-low)/(highest(close, pd)))*100
    wvf = []
    for i in range(len(closes)):
        if i < pd_period - 1:
            wvf.append(0.0)
            continue
        window_closes = closes[i - pd_period + 1 : i + 1]
        highest_close = max(1e-9, np.max(window_closes))
        val = ((highest_close - lows[i]) / highest_close) * 100
        wvf.append(val)
    
    wvf = np.array(wvf)

    # 2. Bandas de Bollinger sobre WVF
    wvf_pandas = pd.Series(wvf)
    mid_line = wvf_pandas.rolling(window=bbl).mean()
    s_dev = wvf_pandas.rolling(window=bbl).std()
    upper_band = mid_line + (s_dev * mult)
    
    # 3. Percentiles de Rango
    # rangeHigh = (highest(wvf, lb)) * ph
    # rangeLow = (lowest(wvf, lb)) * pl
    range_high = wvf_pandas.rolling(window=lb).max() * ph
    range_low = wvf_pandas.rolling(window=lb).min() * pl

    # Valores actuales (última vela confirmada)
    current_wvf = wvf[-1]
    current_upper = upper_band.iloc[-1]
    current_range_high = range_high.iloc[-1]
    current_range_low = range_low.iloc[-1]
    
    # Estados
    # col = wvf >= upperBand or wvf >= rangeHigh ? lime : gray
    is_bottom_signal = (current_wvf >= current_upper) or (current_wvf >= current_range_high)
    
    # Lógica de complacencia (Propuesta por el usuario)
    # WVF persistentemente bajo durante periodos extendidos
    # Usamos una ventana de los últimos 5 periodos para verificar complacencia
    recent_wvf = wvf[-5:]
    recent_range_lows = range_low.iloc[-5:].values
    is_complacency = np.all(recent_wvf <= (recent_range_lows * 1.5)) # 1.5 como margen de "cerca del suelo"

    return {
        "wvf": current_wvf,
        "upper_band": current_upper,
        "range_high": current_range_high,
        "range_low": current_range_low,
        "is_bottom_signal": is_bottom_signal,
        "is_complacency": is_complacency,
        "wvf_series": wvf # Por si se necesita para algo más
    }
