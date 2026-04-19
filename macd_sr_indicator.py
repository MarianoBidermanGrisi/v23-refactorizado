"""
=============================================================
  MACD SUPPORT AND RESISTANCE INDICATOR
  Conversión del script Pine Script "MACD S/R [ChartPrime]"
  
  Basado en cruces MACD vs Señal (Momentum) marcando soportes 
  y resistencias temporales exactos que se rompen (mitigan)
  con la acción del precio.
=============================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class MACDLevel:
    def __init__(self, is_support: bool, lvl: float, start_idx: int):
        self.is_support = is_support
        self.lvl = lvl
        self.start_idx = start_idx


class MacdSupportResistance:
    """
    Parámetros predeterminados del Pine Script:
      - fast_length=12, slow_length=26, signal_length=9
      - EMA es el defecto para el MACD original.
    """
    def __init__(self, fast_length: int = 12, slow_length: int = 26, signal_length: int = 9):
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.signal_length = signal_length

    def update(self, latest_ohlcv: np.ndarray) -> dict:
        """
        Calcula el estado histórico y actual de los Soportes/Resistencias MACD.
        
        Args:
            latest_ohlcv: Array numérico bidimensional con la estructura
                          [ts, open, high, low, close, volume]
        """
        N = len(latest_ohlcv)
        if N < self.slow_length + self.signal_length:
            return self._empty_result()
            
        highs = latest_ohlcv[:, 2]
        lows = latest_ohlcv[:, 3]
        closes = latest_ohlcv[:, 4]
        
        # 1. MACD Matemático clásico usando serie Pandas
        df_close = pd.Series(closes)
        fast_ma = df_close.ewm(span=self.fast_length, adjust=False).mean()
        slow_ma = df_close.ewm(span=self.slow_length, adjust=False).mean()
        macd = fast_ma - slow_ma
        signal = macd.ewm(span=self.signal_length, adjust=False).mean()
        
        macd_arr = macd.values
        sig_arr = signal.values
        
        s_r_levels: List[MACDLevel] = []
        
        # 2. Tracking cronológico de roturas
        # Iteramos desde la quinta vela para buscar extremos en barras anteriores (0 a 5)
        for i in range(5, N):
            prev_macd = macd_arr[i-1]
            prev_sig = sig_arr[i-1]
            curr_macd = macd_arr[i]
            curr_sig = sig_arr[i]
            
            # Detección algorítmica de puntos de inflexión
            crossunder = prev_macd >= prev_sig and curr_macd < curr_sig
            crossover = prev_macd <= prev_sig and curr_macd > curr_sig
            
            # Cruzar hacia ABAJO asume que recién rebotamos de un punto máximo -> nueva RESISTENCIA
            if crossunder:
                window_highs = highs[i-5 : i+1]  # últimas 6 barras
                s_r_levels.append(MACDLevel(is_support=False, lvl=float(np.max(window_highs)), start_idx=i))
                
            # Cruzar hacia ARRIBA asume fin de la caída, recién rebotamos -> nuevo SOPORTE
            if crossover:
                window_lows = lows[i-5 : i+1]
                s_r_levels.append(MACDLevel(is_support=True, lvl=float(np.min(window_lows)), start_idx=i))
                
            # Límite Pine Script: no retener más de 20 niveles vivos por array overflow
            if len(s_r_levels) > 20:
                s_r_levels.pop(0)
                
            # 3. Mitigación: Revisar si la barra iterada acaba de perforar algún nivel antiguo
            new_levels = []
            for sr in s_r_levels:
                # Si es soporte y el precio penetra (low < sr.lvl), el nivel se considera destruido/mitigado
                if sr.start_idx < i and sr.is_support and lows[i] < sr.lvl:
                    continue
                # Si es resistencia y el precio rompe al alza (high > sr.lvl), la resistencia colapsa
                if sr.start_idx < i and not sr.is_support and highs[i] > sr.lvl:
                    continue
                
                new_levels.append(sr)
                
            s_r_levels = new_levels
            
        active_supports = [sr.lvl for sr in s_r_levels if sr.is_support]
        active_resistances = [sr.lvl for sr in s_r_levels if not sr.is_support]
        
        # Estado actual extraído del último frame
        return {
            "macd": float(macd_arr[-1]),
            "signal": float(sig_arr[-1]),
            "is_bullish": float(macd_arr[-1]) > float(sig_arr[-1]),
            "active_supports": sorted(list(set(active_supports))),
            "active_resistances": sorted(list(set(active_resistances)))
        }

    def _empty_result(self) -> dict:
        return {
            "macd": 0.0,
            "signal": 0.0,
            "is_bullish": False,
            "active_supports": [],
            "active_resistances": []
        }
