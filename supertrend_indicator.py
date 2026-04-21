import numpy as np
import pandas as pd

class Supertrend:
    def __init__(self, period=10, multiplier=3.0, change_atr=True):
        self.period = period
        self.multiplier = multiplier
        self.change_atr = change_atr

    def calcular(self, ohlcv_list):
        if not ohlcv_list or len(ohlcv_list) < self.period:
            return {"trend": 1, "line": 0, "is_bullish": True, "st_series": []}
            
        df = pd.DataFrame(ohlcv_list, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        
        src = (df['high'] + df['low']) / 2
        
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        tr = ranges.max(axis=1)
        
        if self.change_atr:
            atr = tr.ewm(alpha=1/self.period, adjust=False).mean() # Pine's ATR (RMA)
        else:
            atr = tr.rolling(window=self.period).mean() # SMA logic
            
        # Support both new pandas bfill() and old fillna(method='bfill')
        atr = atr.bfill() if hasattr(atr, 'bfill') else atr.fillna(method='bfill')
            
        up_base = src - (self.multiplier * atr)
        dn_base = src + (self.multiplier * atr)
        
        up_base_arr = up_base.bfill().values if hasattr(up_base, 'bfill') else up_base.fillna(method='bfill').values
        dn_base_arr = dn_base.bfill().values if hasattr(dn_base, 'bfill') else dn_base.fillna(method='bfill').values
        
        up = np.copy(up_base_arr)
        dn = np.copy(dn_base_arr)
        close = df['close'].values
        trend = np.ones(len(df))
        
        # original pinescript limit: only 1..len
        for i in range(1, len(df)):
            if close[i-1] > up[i-1]:
                up[i] = max(up_base_arr[i], up[i-1])
            else:
                up[i] = up_base_arr[i]
                
            if close[i-1] < dn[i-1]:
                dn[i] = min(dn_base_arr[i], dn[i-1])
            else:
                dn[i] = dn_base_arr[i]
                
            if trend[i-1] == -1 and close[i] > dn[i-1]:
                trend[i] = 1
            elif trend[i-1] == 1 and close[i] < up[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
                
        st_series = np.where(trend == 1, up, dn)
                
        return {
            "trend": int(trend[-1]),
            "line": float(st_series[-1]),
            "is_bullish": trend[-1] == 1,
            "st_series": st_series.tolist()
        }
