"""
Trendlines with Breaks [LuxAlgo]
Traducción fiel del indicador Pine Script a Python stateful (con estado por símbolo).

Lógica corregida:
  - La pendiente se calcula entre DOS pivots consecutivos (no entre pivot y close).
  - La línea se extiende desde el segundo pivot hasta la barra actual con esa pendiente.
  - in_uptrend  = close > línea superior extendida (rompió la resistencia → alcista)
  - in_downtrend = close < línea inferior extendida (rompió el soporte → bajista)

Parámetros:
    length  : Mitad de la ventana para detectar pivots (e.g. 10 → busca ±10 velas)
    mult    : (reservado, para compatibilidad futura con bandas ATR)

Salida (dict):
    upper_break   : True si el precio acaba de romper la línea de resistencia
    lower_break   : True si el precio acaba de romper la línea de soporte
    in_uptrend    : True si close > línea de resistencia extendida
    in_downtrend  : True si close < línea de soporte extendida
    upper_line    : Valor numérico de la línea de resistencia en la barra actual
    lower_line    : Valor numérico de la línea de soporte en la barra actual
"""

import logging

logger = logging.getLogger(__name__)


class TrendlinesBreaks:
    """
    Trendlines with Breaks [LuxAlgo] — implementación Python corregida.
    Mantiene estado entre llamadas para simular el comportamiento bar-by-bar de TradingView.
    """

    def __init__(self, length: int = 14, mult: float = 1.0):
        self.length = length
        self.mult   = mult

        # Dos pivots de resistencia (para calcular pendiente real)
        self._ph1_val = None   # pivot más antiguo
        self._ph1_idx = None
        self._ph2_val = None   # pivot más reciente
        self._ph2_idx = None

        # Dos pivots de soporte
        self._pl1_val = None
        self._pl1_idx = None
        self._pl2_val = None
        self._pl2_idx = None

        # Contadores de ruptura (evento puntual = cambio de 0→1)
        self.upos      = 0
        self.dnos      = 0
        self.last_upos = 0
        self.last_dnos = 0

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _find_pivot_high(self, highs: list, idx: int) -> float | None:
        """Retorna el valor si idx es un máximo pivot, de lo contrario None."""
        left  = highs[max(0, idx - self.length): idx]
        right = highs[idx + 1: min(len(highs), idx + 1 + self.length)]
        val   = highs[idx]
        if left and right and val >= max(left) and val >= max(right):
            return val
        return None

    def _find_pivot_low(self, lows: list, idx: int) -> float | None:
        """Retorna el valor si idx es un mínimo pivot, de lo contrario None."""
        left  = lows[max(0, idx - self.length): idx]
        right = lows[idx + 1: min(len(lows), idx + 1 + self.length)]
        val   = lows[idx]
        if left and right and val <= min(left) and val <= min(right):
            return val
        return None

    def _line_at(self, val1: float, idx1: int, val2: float, idx2: int, current_idx: int) -> float:
        """Extiende la línea definida por (idx1,val1)→(idx2,val2) hasta current_idx."""
        if idx2 == idx1:
            return val2
        slope = (val2 - val1) / (idx2 - idx1)
        return val2 + slope * (current_idx - idx2)

    # ──────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────

    def update(self, ohlcv: list) -> dict:
        """
        Recibe la lista COMPLETA de velas OHLCV confirmadas y recalcula las líneas.
        Cada elemento: [timestamp, open, high, low, close, volume]
        """
        try:
            min_bars = self.length * 2 + 2
            if not ohlcv or len(ohlcv) < min_bars:
                return self._neutral()

            highs  = [c[2] for c in ohlcv]
            lows   = [c[3] for c in ohlcv]
            closes = [c[4] for c in ohlcv]
            n      = len(ohlcv)
            last   = n - 1

            # ── Rastrear los dos pivots de resistencia más recientes ──
            found_ph = 0
            for i in range(last - 1, self.length - 1, -1):
                if i + self.length >= n:
                    continue
                v = self._find_pivot_high(highs, i)
                if v is not None:
                    if found_ph == 0:
                        self._ph2_val, self._ph2_idx = v, i
                        found_ph = 1
                    elif found_ph == 1:
                        self._ph1_val, self._ph1_idx = v, i
                        break

            # ── Rastrear los dos pivots de soporte más recientes ──
            found_pl = 0
            for i in range(last - 1, self.length - 1, -1):
                if i + self.length >= n:
                    continue
                v = self._find_pivot_low(lows, i)
                if v is not None:
                    if found_pl == 0:
                        self._pl2_val, self._pl2_idx = v, i
                        found_pl = 1
                    elif found_pl == 1:
                        self._pl1_val, self._pl1_idx = v, i
                        break

            close_actual = closes[last]

            # ── Calcular líneas extendidas al bar actual ──
            if self._ph2_val is not None and self._ph1_val is not None:
                upper = self._line_at(
                    self._ph1_val, self._ph1_idx,
                    self._ph2_val, self._ph2_idx,
                    last
                )
            elif self._ph2_val is not None:
                # Solo un pivot: línea horizontal desde él
                upper = self._ph2_val
            else:
                upper = max(highs[-self.length:])

            if self._pl2_val is not None and self._pl1_val is not None:
                lower = self._line_at(
                    self._pl1_val, self._pl1_idx,
                    self._pl2_val, self._pl2_idx,
                    last
                )
            elif self._pl2_val is not None:
                lower = self._pl2_val
            else:
                lower = min(lows[-self.length:])

            # ── Detección de ruptura (evento puntual) ──
            self.last_upos = self.upos
            self.last_dnos = self.dnos

            self.upos = 1 if close_actual > upper else 0
            self.dnos = 1 if close_actual < lower  else 0

            return {
                "upper_break":  (self.upos == 1 and self.last_upos == 0),
                "lower_break":  (self.dnos == 1 and self.last_dnos == 0),
                "in_uptrend":   (close_actual > upper),
                "in_downtrend": (close_actual < lower),
                "upper_line":   upper,
                "lower_line":   lower,
            }

        except Exception as e:
            logger.error(f"Error en TrendlinesBreaks: {e}")
            return self._neutral()

    def _neutral(self) -> dict:
        return {
            "upper_break":  False,
            "lower_break":  False,
            "in_uptrend":   False,
            "in_downtrend": False,
            "upper_line":   0.0,
            "lower_line":   0.0,
        }
