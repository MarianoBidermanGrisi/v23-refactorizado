"""
=============================================================
  VP & PIVOT LEVELS INDICATOR (Python)
  Conversión del indicador Pine Script 'Volume Profile + Pivot Levels'
  
  Calcula el Volumen Perfilado en una ventana rolling, detecta el 
  Point of Control (PoC), el Delta global y niveles de Soportes/
  Resistencias basados en Pivotes de alto volumen.
=============================================================
"""

import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Pivot:
    def __init__(self, value: float, index: int, is_high: bool):
        self.value = value
        self.index = index
        self.is_high = is_high


class VolumeProfilePivots:
    """
    Réplica Python de la lógica Volume Profile + Pivot Levels.
    
    Parámetros:
        period       : Número de velas para la ventana del perfil de volumen (start en PS). (Default: 200)
        bins         : Resolución del Volume Profile (Default: 50)
        pivot_length : Longitud requerida antes y después para confirmar un pivote. (Default: 10)
        pivot_filter : Porcentaje (0-100) del volumen máximo (PoC) para validar un pivote. (Default: 20)
    """
    
    def __init__(self, period: int = 200, bins: int = 50, pivot_length: int = 10, pivot_filter: int = 20):
        self.period = period
        self.bins = bins
        self.pivot_length = pivot_length
        self.pivot_filter = pivot_filter

    def update(self, latest_ohlcv: np.ndarray) -> dict:
        """
        Calcula el estado actual del Volume Profile y los Pivotes Unmitigated.
        
        Args:
            latest_ohlcv: Array bidimensional Numpy shape (N, 6).
                          Requerido orden: [ts, open, high, low, close, volume]
                          N >= period + pivot_length
                          
        Returns:
            Dict con métricas de confluencia y niveles.
        """
        N = len(latest_ohlcv)
        if N < self.period:
            return self._empty_result()
            
        # Nos concentramos en la ventana de análisis principal ('start' en Pine Script)
        start_idx = N - self.period
        
        # Ojo: Para obtener pivotes justos al principio de la ventana, necesitamos contexto previo,
        # pero para mantener rigor con el PS original, limitamos la acción en torno al rango validado.
        
        opens = latest_ohlcv[start_idx:, 1]
        highs = latest_ohlcv[start_idx:, 2]
        lows = latest_ohlcv[start_idx:, 3]
        closes = latest_ohlcv[start_idx:, 4]
        volumes = latest_ohlcv[start_idx:, 5]
        
        # 1. Delta Global
        # PineScript: array.get(Delta, i) + (close > open ? volume : - volume)
        delta_arr = np.where(closes > opens, volumes, -volumes) if len(closes)>0 else np.array([0])
        total_delta = np.sum(delta_arr)
        
        # 2. Rango Mayor y Menor del Período
        H = np.max(highs)
        L = np.min(lows)
        
        # 3. Binning y Volume Profile
        bins_vol = np.zeros(self.bins)
        poc_price = L
        max_vol = 0
        bin_size = 0.0001
        
        if H > L:
            bin_size = (H - L) / self.bins
            for i in range(self.bins):
                bin_low = L + bin_size * i
                bin_high = bin_low + bin_size
                
                # Regla de asignación ChartPrime: close in [bin_low-bin_size, bin_high+bin_size)
                # Crea un suavizado 'smooth' artificial entre niveles
                cond = (closes >= bin_low - bin_size) & (closes < bin_high + bin_size)
                bins_vol[i] = np.sum(volumes[cond])
        
            max_vol = np.max(bins_vol)
            poc_bin_idx = np.argmax(bins_vol)
            poc_price = L + bin_size * poc_bin_idx + (bin_size / 2.0)

        # 4. Encontrar Pivotes Estructurales
        pivots: List[Pivot] = []
        full_highs = latest_ohlcv[:, 2]
        full_lows = latest_ohlcv[:, 3]
        
        # Escaneamos desde la base de la ventana hasta (N - pivot_length) para que el pivote
        # tenga las barras siguientes garantizadas. 
        # Rango seguro de escaneo es [max(pivot_length, start_idx), N - pivot_length]
        for i in range(max(self.pivot_length, start_idx), N - self.pivot_length):
            # Verificar Pivot High
            is_ph = True
            for j in range(1, self.pivot_length + 1):
                if full_highs[i] <= full_highs[i - j] or full_highs[i] <= full_highs[i + j]:
                    is_ph = False
                    break
            if is_ph:
                pivots.append(Pivot(full_highs[i], i, True))
                
            # Verificar Pivot Low
            is_pl = True
            for j in range(1, self.pivot_length + 1):
                if full_lows[i] >= full_lows[i - j] or full_lows[i] >= full_lows[i + j]:
                    is_pl = False
                    break
            if is_pl:
                pivots.append(Pivot(full_lows[i], i, False))
                
        # 5. Filamentación y Validación de Pivotes (Confluencia SV)
        active_supports = []
        active_resistances = []
        
        for p in pivots:
            if H > L:
                # Encuentra el bin medio correspondiente a la altura de precio del pivote
                bin_lows = L + bin_size * np.arange(self.bins)
                bin_mids = bin_lows + bin_size / 2.0
                distances = np.abs(bin_mids - p.value)
                closest_bin_idx = np.argmin(distances)
                
                # Verificación estricta (PineScript: math.abs(mid-val) <= bin_size and volPercent >= pivotFilter)
                if distances[closest_bin_idx] <= bin_size:
                    vol_percent = (bins_vol[closest_bin_idx] / max_vol) * 100 if max_vol > 0 else 0
                    if vol_percent >= self.pivot_filter:
                        
                        # 6. Agotamiento de Niveles ('Unmitigated levels check')
                        # Revisar desde el pivote hacia el presente si los precios lo han atravesado.
                        mitigated = False
                        # Solo analizamos velas *posteriores* al pivote
                        for k in range(p.index + 1, N):
                            # Rotura sucede si el nivel queda envuelto por una vela (high > nivel y low < nivel)
                            if full_highs[k] > p.value and full_lows[k] < p.value:
                                mitigated = True
                                break
                                
                        if not mitigated:
                            if p.is_high:
                                active_resistances.append(p.value)
                            else:
                                active_supports.append(p.value)
        
        # Eliminar posibles duplicidades cercanas generadas por el binning y ordenar.
        active_supports = sorted(list(set(active_supports)))
        active_resistances = sorted(list(set(active_resistances)))
        
        return {
            "total_delta": float(total_delta),
            "delta_positive": total_delta > 0,
            "poc_price": float(poc_price),
            "active_supports": active_supports,
            "active_resistances": active_resistances,
            "period_high": float(H),
            "period_low": float(L)
        }
        
    def _empty_result(self) -> dict:
        return {
            "total_delta": 0.0,
            "delta_positive": False,
            "poc_price": 0.0,
            "active_supports": [],
            "active_resistances": [],
            "period_high": 0.0,
            "period_low": 0.0
        }
