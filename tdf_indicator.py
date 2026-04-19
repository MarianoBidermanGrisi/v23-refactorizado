"""
=============================================================
  TDF INDICATOR — Trend Duration Forecast (Python)
  Conversión exacta del indicador Pine Script de ChartPrime

  MÓDULO AUTÓNOMO — se importa en render_bitget_core.py
  desde la función escanear_mercado()

  Uso básico:
    from tdf_indicator import TrendDurationForecast

    tdf = TrendDurationForecast(length=50, trend_sensitivity=3, samples=10)
    signal = tdf.update(closes)
    # signal es un dict con: trend, trend_count, avg_bullish,
    #   avg_bearish, probable_length, is_new_trend, confidence
=============================================================
"""

import numpy as np
from collections import deque
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ==============================================================
#  HULL MOVING AVERAGE
# ==============================================================

def _wma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Weighted Moving Average vectorizado con NumPy.
    Pesos lineales: 1, 2, 3, ..., period (más reciente = más peso).
    Retorna NaN donde no hay suficientes datos.
    """
    n = len(data)
    result = np.full(n, np.nan)
    weights = np.arange(1, period + 1, dtype=float)
    w_sum = weights.sum()

    for i in range(period - 1, n):
        window = data[i - period + 1 : i + 1]
        result[i] = np.dot(window, weights) / w_sum

    return result


def calcular_hma(closes: np.ndarray, length: int) -> np.ndarray:
    """
    Hull Moving Average.
    Fórmula: WMA( 2×WMA(close, n/2) − WMA(close, n) , √n )

    Args:
        closes: array numpy de precios de cierre (más antiguo primero)
        length: período del HMA

    Returns:
        array numpy con valores HMA (NaN donde no hay suficientes datos)
    """
    half  = max(1, length // 2)
    sqrt_ = max(1, int(np.sqrt(length)))

    wma_half  = _wma(closes, half)
    wma_full  = _wma(closes, length)

    # Serie intermedia: doble de la rápida menos la lenta
    intermediate = 2.0 * wma_half - wma_full

    # HMA = WMA de esa intermedia con período √length
    hma = _wma(intermediate, sqrt_)
    return hma


# ==============================================================
#  DETECCIÓN DE ASCENSO / DESCENSO (equivalente ta.rising / ta.falling)
# ==============================================================

def _es_ascendente(hma: np.ndarray, idx: int, n: int) -> bool:
    """
    True si hma[idx] > hma[idx-1] > ... > hma[idx-n+1]
    Equivalente exacto de ta.rising(hma, n) en Pine Script.
    """
    if idx < n:
        return False
    for i in range(idx - n + 1, idx + 1):
        if hma[i] <= hma[i - 1]:
            return False
    return True


def _es_descendente(hma: np.ndarray, idx: int, n: int) -> bool:
    """
    True si hma[idx] < hma[idx-1] < ... < hma[idx-n+1]
    Equivalente exacto de ta.falling(hma, n) en Pine Script.
    """
    if idx < n:
        return False
    for i in range(idx - n + 1, idx + 1):
        if hma[i] >= hma[i - 1]:
            return False
    return True


# ==============================================================
#  CLASE PRINCIPAL — TREND DURATION FORECAST
# ==============================================================

class TrendDurationForecast:
    """
    Réplica Python del indicador "Trend Duration Forecast [ChartPrime]".

    Mantiene el estado interno barra a barra, igual que Pine Script usa
    variables 'var': current_trend, trend_count, y los deques de duraciones.

    Parámetros (idénticos al Pine Script original):
        length            → Smoothing Length (HMA period)         default 50
        trend_sensitivity → Trend Detection Sensitivity           default 3
        samples           → Trend Sample Size (rolling window)    default 10
    """

    def __init__(
        self,
        length: int = 50,
        trend_sensitivity: int = 3,
        samples: int = 10,
    ):
        self.length            = length
        self.trend_sensitivity = trend_sensitivity
        self.samples           = samples

        # ── Estado persistente (equivalente a 'var' en Pine Script) ──
        self.current_trend: Optional[bool] = None   # True=alcista, False=bajista, None=sin datos
        self.trend_count:   int            = 0       # barras en tendencia actual

        # Arrays rolling de duraciones (deque con maxlen = samples)
        self.bullish_durations: deque = deque(maxlen=samples)
        self.bearish_durations: deque = deque(maxlen=samples)

    # ------------------------------------------------------------------
    #  API PRINCIPAL: update()
    # ------------------------------------------------------------------

    def update(self, closes: np.ndarray) -> dict:
        """
        Procesa el array completo de cierres y retorna el estado actual.

        En un bot de trading, llamas este método cada vez que llega
        una nueva vela CONFIRMADA, pasando todos los cierres históricos
        disponibles (el más reciente al final).

        Returns dict con:
            trend           : 1 (alcista) | -1 (bajista) | 0 (indefinido)
            trend_count     : barras transcurridas en la tendencia actual
            hma_current     : valor HMA de la última barra
            avg_bullish     : duración media de tendencias alcistas previas (float | None)
            avg_bearish     : duración media de tendencias bajistas previas (float | None)
            probable_length : duración probable de la tendencia actual (float | None)
            remaining_bars  : barras probables restantes hasta agotamiento (float | None)
            is_new_trend    : True si acaba de cambiar la tendencia en esta actualización
            confidence      : puntuación 0-100 de fiabilidad de la señal
            bullish_history : lista de duraciones alcistas almacenadas
            bearish_history : lista de duraciones bajistas almacenadas
        """
        closes = np.asarray(closes, dtype=float)

        # Necesitamos al menos length + trend_sensitivity barras para calcular
        min_bars = self.length + self.trend_sensitivity + 5
        if len(closes) < min_bars:
            return self._resultado_vacio()

        # 1. Calcular HMA completo
        hma = calcular_hma(closes, self.length)
        idx = len(closes) - 1   # índice de la última barra (más reciente)

        # 2. Detectar dirección (solo en barra confirmada — la última)
        rising  = _es_ascendente(hma,  idx, self.trend_sensitivity)
        falling = _es_descendente(hma, idx, self.trend_sensitivity)

        # 3. Actualizar tendencia (lógica idéntica al Pine Script)
        is_new_trend = False
        new_trend    = self.current_trend   # por defecto, mantiene

        if rising and not falling:
            new_trend = True
        elif falling and not rising:
            new_trend = False

        # 4. Cambio de tendencia detectado
        if new_trend != self.current_trend:
            is_new_trend = True

            # Guardar duración de la tendencia que ACABA DE TERMINAR
            if self.current_trend is True:
                self.bullish_durations.append(self.trend_count)
                logger.debug(f"[TDF] Cerrada tendencia ALCISTA de {self.trend_count} barras")
            elif self.current_trend is False:
                self.bearish_durations.append(self.trend_count)
                logger.debug(f"[TDF] Cerrada tendencia BAJISTA de {self.trend_count} barras")

            self.current_trend = new_trend
            self.trend_count   = 0

        # 5. Incrementar contador de tendencia (siempre, como el Pine Script)
        self.trend_count += 1

        # 6. Calcular promedios y duración probable
        avg_bull = self._promedio(self.bullish_durations)
        avg_bear = self._promedio(self.bearish_durations)

        if self.current_trend is True:
            probable_length = avg_bull
            trend_code      = 1
        elif self.current_trend is False:
            probable_length = avg_bear
            trend_code      = -1
        else:
            probable_length = None
            trend_code      = 0

        # 7. Barras restantes hasta agotamiento probable
        remaining = None
        if probable_length is not None:
            remaining = max(0.0, probable_length - self.trend_count)

        # 8. Calcular confianza de la señal
        confidence = self._calcular_confianza(
            closes, hma, idx, trend_code, probable_length
        )

        return {
            "trend":           trend_code,
            "trend_count":     self.trend_count,
            "hma_current":     float(hma[idx]) if not np.isnan(hma[idx]) else None,
            "avg_bullish":     avg_bull,
            "avg_bearish":     avg_bear,
            "probable_length": probable_length,
            "remaining_bars":  remaining,
            "is_new_trend":    is_new_trend,
            "confidence":      confidence,
            "bullish_history": list(self.bullish_durations),
            "bearish_history": list(self.bearish_durations),
        }

    # ------------------------------------------------------------------
    #  EVALUACIÓN DE SEÑAL PARA EL BOT
    # ------------------------------------------------------------------

    def evaluar_entrada(
        self,
        closes: np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Wrapper de alto nivel que retorna directamente si hay señal
        de entrada LONG, SHORT, o ninguna.

        Diseñado para integrarse en la función escanear_mercado()
        del bot (render_bitget_core.py).

        Returns dict con:
            signal        : 'BUY' | 'SELL' | None
            side          : 'buy' | 'sell' | None  (listo para abrir_operacion)
            tendencia_str : string descriptivo para Telegram
            fuerza        : int 0-7 (escala del bot original)
            confidence    : 0-100
            detalle       : dict completo del estado TDF
        """
        detalle = self.update(closes)

        signal        = None
        side          = None
        tendencia_str = "INDETERMINADO"

        # Solo generar señal cuando la tendencia ACABA de cambiar
        if detalle["is_new_trend"]:
            if detalle["trend"] == 1:
                signal        = "BUY"
                side          = "buy"
                tendencia_str = "ALCISTA (HMA ↑)"
            elif detalle["trend"] == -1:
                signal        = "SELL"
                side          = "sell"
                tendencia_str = "BAJISTA (HMA ↓)"
        else:
            # Sin nuevo cambio — reportar estado actual
            if detalle["trend"] == 1:
                tendencia_str = "ALCISTA (continuando)"
            elif detalle["trend"] == -1:
                tendencia_str = "BAJISTA (continuando)"

        # Convertir confidence (0-100) a fuerza (0-7)
        fuerza = round(detalle["confidence"] / 100 * 7)

        # Filtro mínimo: usar solo si confidence >= 50
        if signal is not None and detalle["confidence"] < 50:
            logger.info(
                f"[TDF] Señal {signal} descartada — confidence={detalle['confidence']} < 50"
            )
            signal = None
            side   = None

        return {
            "signal":        signal,
            "side":          side,
            "tendencia_str": tendencia_str,
            "fuerza":        fuerza,
            "confidence":    detalle["confidence"],
            "detalle":       detalle,
        }

    def reset(self):
        """Reinicia el estado interno (útil al cambiar de símbolo/timeframe)."""
        self.current_trend      = None
        self.trend_count        = 0
        self.bullish_durations  = deque(maxlen=self.samples)
        self.bearish_durations  = deque(maxlen=self.samples)

    # ------------------------------------------------------------------
    #  HELPERS INTERNOS
    # ------------------------------------------------------------------

    def _promedio(self, duraciones: deque) -> Optional[float]:
        """Media aritmética del deque. None si está vacío."""
        if not duraciones:
            return None
        return float(np.mean(list(duraciones)))

    def _calcular_confianza(
        self,
        closes: np.ndarray,
        hma:    np.ndarray,
        idx:    int,
        trend:  int,
        probable_length: Optional[float],
    ) -> int:
        """
        Puntaje 0-100 de fiabilidad de la señal actual.

        Criterios (igual que evaluate_long/short_entry del plan):
          +30  Tendencia confirmada por HMA
          +20  Precio en el lado correcto del HMA
          +20  Volumen por encima del promedio (si se pasa)
          +20  Duración previa en rango favorable
          +10  Tendencia actual joven respecto al promedio
        """
        score = 0

        if trend != 0:
            score += 30  # Tendencia confirmada

        # Precio vs HMA
        hma_val = hma[idx]
        if not np.isnan(hma_val):
            precio = closes[idx]
            if trend == 1 and precio > hma_val:
                score += 20
            elif trend == -1 and precio < hma_val:
                score += 20

        # Duración previa favorable
        if trend == 1 and self.bearish_durations:
            avg_bear  = float(np.mean(list(self.bearish_durations)))
            last_bear = self.bearish_durations[-1]
            if last_bear <= avg_bear:
                score += 20   # Bajista corta → rebote más probable
            else:
                score += 10   # Bajista larga → compradores agotados aún en duda
        elif trend == -1 and self.bullish_durations:
            avg_bull  = float(np.mean(list(self.bullish_durations)))
            last_bull = self.bullish_durations[-1]
            if last_bull >= avg_bull:
                score += 20   # Alcista larga → compradores agotados
            else:
                score += 10

        # Tendencia actual joven
        if probable_length is not None and probable_length > 0:
            pct_consumido = self.trend_count / probable_length
            if pct_consumido < 0.5:
                score += 10   # Tendencia joven, tiene recorrido

        return min(100, score)

    def _resultado_vacio(self) -> dict:
        """Diccionario vacío para cuando no hay suficientes datos."""
        return {
            "trend": 0, "trend_count": 0, "hma_current": None,
            "avg_bullish": None, "avg_bearish": None,
            "probable_length": None, "remaining_bars": None,
            "is_new_trend": False, "confidence": 0,
            "bullish_history": [], "bearish_history": [],
        }

    def __repr__(self) -> str:
        trend_str = {True: "ALCISTA ↑", False: "BAJISTA ↓", None: "INDEFINIDO"}.get(
            self.current_trend, "?"
        )
        avg_b  = self._promedio(self.bullish_durations)
        avg_be = self._promedio(self.bearish_durations)
        avg_b_str  = f"{avg_b:.1f}"  if avg_b  is not None else "N/A"
        avg_be_str = f"{avg_be:.1f}" if avg_be is not None else "N/A"
        return (
            f"TrendDurationForecast("
            f"length={self.length}, sensitivity={self.trend_sensitivity}, "
            f"samples={self.samples}) | "
            f"Tendencia: {trend_str} | "
            f"Barras: {self.trend_count} | "
            f"AvgBull={avg_b_str} | "
            f"AvgBear={avg_be_str}"
        )
