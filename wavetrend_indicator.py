import pandas as pd
import numpy as np

class WaveTrendIndicator:
    """
    Implementación del indicador WaveTrend [LazyBear] de TradingView.
    Es un oscilador de momentum altamente efectivo para criptomonedas,
    superior al MACD para detectar cambios de ciclo sin tanto retraso.
    """
    def __init__(self, n1=10, n2=21):
        self.n1 = n1
        self.n2 = n2

    def calculate(self, df: pd.DataFrame) -> dict:
        if len(df) < max(self.n1, self.n2) * 2:
            return {
                'wt1': 0.0, 
                'wt2': 0.0, 
                'is_bullish': False, 
                'is_bearish': False,
                'cross_up': False,
                'cross_down': False
            }

        ap = (df['high'] + df['low'] + df['close']) / 3.0
        
        # esa = ema(ap, n1)
        esa = ap.ewm(span=self.n1, adjust=False).mean()
        
        # d = ema(abs(ap - esa), n1)
        d = (ap - esa).abs().ewm(span=self.n1, adjust=False).mean()
        
        # ci = (ap - esa) / (0.015 * d)
        ci = (ap - esa) / (0.015 * d).replace(0, np.nan)
        ci = ci.fillna(0)
        
        # tci = ema(ci, n2)
        wt1 = ci.ewm(span=self.n2, adjust=False).mean()
        
        # wt2 = sma(wt1, 4)
        wt2 = wt1.rolling(window=4).mean()
        
        current_wt1 = wt1.iloc[-1]
        current_wt2 = wt2.iloc[-1]
        prev_wt1 = wt1.iloc[-2]
        prev_wt2 = wt2.iloc[-2]
        
        # La tendencia a corto plazo es alcista si wt1 está por encima de wt2 (histograma verde)
        is_bullish = current_wt1 > current_wt2
        is_bearish = current_wt1 < current_wt2
        
        # Detección de cruces exactos (entradas precisas)
        cross_up = (prev_wt1 <= prev_wt2) and (current_wt1 > current_wt2)
        cross_down = (prev_wt1 >= prev_wt2) and (current_wt1 < current_wt2)
        
        return {
            'wt1': round(current_wt1, 2),
            'wt2': round(current_wt2, 2),
            'is_bullish': is_bullish,
            'is_bearish': is_bearish,
            'cross_up': cross_up,
            'cross_down': cross_down
        }
