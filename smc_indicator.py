import numpy as np
import pandas as pd

class SmartMoneyConcepts:
    def __init__(self, pivot_length=10):
        self.pivot_length = pivot_length

    def calcular(self, ohlcv_list):
        if not ohlcv_list or len(ohlcv_list) < self.pivot_length * 2 + 1:
            return {"is_bullish_structure": True, "bullish_obs": [], "bearish_obs": []}
            
        df = pd.DataFrame(ohlcv_list, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values
        
        last_swing_high = float('inf')
        last_swing_low = float('-inf')
        last_sh_idx = 0
        last_sl_idx = 0
        
        structure_bullish = True
        
        bullish_obs = []
        bearish_obs = []
        
        pl = self.pivot_length
        
        for i in range(pl * 2, len(df)):
            idx = i - pl
            
            is_swing_high = True
            is_swing_low = True
            for j in range(idx - pl, i + 1):
                if highs[j] > highs[idx]: is_swing_high = False
                if lows[j] < lows[idx]: is_swing_low = False
                
            if is_swing_high:
                last_swing_high = highs[idx]
                last_sh_idx = idx
            if is_swing_low:
                last_swing_low = lows[idx]
                last_sl_idx = idx
                
            # Break of Structure / CHoCH
            if not structure_bullish and closes[i] > last_swing_high:
                structure_bullish = True
                last_swing_high = float('inf') 
                
                # OB: Find the last bearish candle before this bullish push
                search_start = max(0, last_sl_idx - 5)
                ob_bottom = float('inf')
                ob_top = float('-inf')
                found = False
                for k in range(i, search_start - 1, -1):
                    if closes[k] < opens[k]: # Bearish candle
                        ob_bottom = min(ob_bottom, lows[k])
                        ob_top = max(ob_top, highs[k])
                        found = True
                        break
                if found:
                    bullish_obs.append({'top': ob_top, 'bottom': ob_bottom})
                    
            elif structure_bullish and closes[i] < last_swing_low:
                structure_bullish = False
                last_swing_low = float('-inf')
                
                # OB: Find the last bullish candle before this bearish push
                search_start = max(0, last_sh_idx - 5)
                ob_bottom = float('inf')
                ob_top = float('-inf')
                found = False
                for k in range(i, search_start - 1, -1):
                    if closes[k] > opens[k]: # Bullish candle
                        ob_bottom = min(ob_bottom, lows[k])
                        ob_top = max(ob_top, highs[k])
                        found = True
                        break
                if found:
                    bearish_obs.append({'top': ob_top, 'bottom': ob_bottom})
                    
            # Mitigation - Purge blocks that were crossed
            current_low = lows[i]
            current_high = highs[i]
            
            valid_bullish_obs = []
            for ob in bullish_obs:
                if current_low > ob['bottom']: 
                    valid_bullish_obs.append(ob)
            bullish_obs = valid_bullish_obs[-5:] # Keep last 5
            
            valid_bearish_obs = []
            for ob in bearish_obs:
                if current_high < ob['top']: 
                    valid_bearish_obs.append(ob)
            bearish_obs = valid_bearish_obs[-5:] # Keep last 5

        return {
            "is_bullish_structure": structure_bullish,
            "bullish_obs": bullish_obs,
            "bearish_obs": bearish_obs
        }
