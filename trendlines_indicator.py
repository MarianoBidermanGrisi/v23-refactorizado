import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TrendlinesBreaks:
    """
    Traducción fiel del indicador 'Trendlines with Breaks [LuxAlgo]' de Pine Script a Python.
    Identifica líneas de tendencia dinámicas basadas en pivots y detecta rupturas con slope de ATR.
    """
    def __init__(self, length=10, mult=1.0):
        self.length = length
        self.mult = mult
        
        # Estado persistente
        self.upper = None  # None = sin pivots establecidos todavía
        self.lower = None  # None = sin pivots establecidos todavía
        self.slope_ph = 0.0
        self.slope_pl = 0.0
        self.upos = 0
        self.dnos = 0
        self.last_upos = 0
        self.last_dnos = 0

    def _calcular_atr(self, df, length):
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        return tr.rolling(window=length).mean()

    def update(self, ohlcv_confirmadas):
        """
        Calcula el estado actual de las líneas de tendencia.
        Retorna señales de ruptura y estado de tendencia.
        """
        try:
            df = pd.DataFrame(ohlcv_confirmadas, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            n = len(df)
            
            # Necesitamos al menos length * 2 velas para detectar pivots
            if n < self.length * 2 + 1:
                return {
                    "upper_break": False, 
                    "lower_break": False, 
                    "in_uptrend": False, 
                    "in_downtrend": False,
                    "upper_line": 0.0,
                    "lower_line": 0.0
                }

            # 1. Calcular Slope basado en ATR (Método LuxAlgo)
            atr_series = self._calcular_atr(df, self.length)
            atr_val = float(atr_series.iloc[-1])
            slope = (atr_val / self.length) * self.mult

            highs = df['high'].values
            lows = df['low'].values
            
            # 2. Detección de Pivots (Confirmados con 'length' velas de retraso)
            # Pine Script: ta.pivothigh(length, length)
            ph = None
            idx_ph = n - 1 - self.length
            if highs[idx_ph] == max(highs[idx_ph - self.length : idx_ph + self.length + 1]):
                ph = float(highs[idx_ph])

            pl = None
            idx_pl = n - 1 - self.length
            if lows[idx_pl] == min(lows[idx_pl - self.length : idx_pl + self.length + 1]):
                pl = float(lows[idx_pl])

            # 3. Actualizar Slopes y Líneas de Tendencia
            # Upper Trendline (Resistencia)
            if ph is not None:
                self.slope_ph = slope
                self.upper = ph
            elif self.upper is not None:
                self.upper -= self.slope_ph
            # else: upper sigue en None hasta que haya un primer pivot

            # Lower Trendline (Soporte)
            if pl is not None:
                self.slope_pl = slope
                self.lower = pl
            elif self.lower is not None:
                self.lower += self.slope_pl
            # else: lower sigue en None hasta que haya un primer pivot

            # 4. Detectar Rupturas (Breakouts)
            close_actual = float(df['close'].iloc[-1])
            
            # Guardar estado previo para detectar el cruce exacto
            self.last_upos = self.upos
            self.last_dnos = self.dnos

            # Si no hay pivots establecidos aún, retornar bloqueante (conservador)
            if self.upper is None or self.lower is None:
                return {
                    "upper_break": False, 
                    "lower_break": False, 
                    "in_uptrend": False,   # Bloqueante hasta tener pivots reales
                    "in_downtrend": False,
                    "upper_line": 0.0,
                    "lower_line": 0.0
                }

            # Lógica de tendencia: ¿estamos por encima de la línea de resistencia o por debajo del soporte?
            if ph is not None:
                self.upos = 0
            elif close_actual > self.upper:
                self.upos = 1
                
            if pl is not None:
                self.dnos = 0
            elif close_actual < self.lower:
                self.dnos = 1

            return {
                "upper_break": (self.upos > self.last_upos),
                "lower_break": (self.dnos > self.last_dnos),
                "in_uptrend": (close_actual > self.upper),
                "in_downtrend": (close_actual < self.lower),
                "upper_line": self.upper,
                "lower_line": self.lower
            }
        except Exception as e:
            logger.error(f"Error en TrendlinesBreaks: {e}")
            return {
                "upper_break": False, 
                "lower_break": False, 
                "in_uptrend": False, 
                "in_downtrend": False,
                "upper_line": 0.0,
                "lower_line": 0.0
            }
