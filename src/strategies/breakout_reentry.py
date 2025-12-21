"""
Estrategias de trading con logs detallados.
Contiene toda la lógica de análisis técnico y señales de trading.
MEJORADO CON LOGS EXTENSIVOS PARA MAYOR VISIBILIDAD
CORRECCIÓN: Lógica de breakout corregida para que coincida con la estrategia original
"""
import numpy as np
import math
import statistics
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from ..config.settings import config, Constants

@dataclass
class CanalInfo:
    """Información del canal de regresión"""
    resistencia: float
    soporte: float
    resistencia_media: float
    soporte_media: float
    linea_tendencia: float
    pendiente_tendencia: float
    precio_actual: float
    ancho_canal: float
    ancho_canal_porcentual: float
    angulo_tendencia: float
    coeficiente_pearson: float
    fuerza_texto: str
    nivel_fuerza: int
    direccion: str
    r2_score: float
    pendiente_resistencia: float
    pendiente_soporte: float
    stoch_k: float
    stoch_d: float
    timeframe: str
    num_velas: int

@dataclass
class DatosMercado:
    """Datos del mercado organizados"""
    maximos: List[float]
    minimos: List[float]
    cierres: List[float]
    tiempos: List[int]
    precio_actual: float
    timeframe: str
    num_velas: int

class EstrategiaBreakoutReentry:
    """Estrategia de trading Breakout + Reentry"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Contadores para estadísticas de logs
        self.stats = {
            'datos_obtenidos': 0,
            'regresiones_calculadas': 0,
            'canales_calculados': 0,
            'breakouts_detectados': 0,
            'reentries_detectados': 0,
            'operaciones_calculadas': 0
        }

    def obtener_datos_mercado(self, symbol: str, timeframe: str, num_velas: int) -> Optional[DatosMercado]:
        """
        Obtiene y organiza datos del mercado
        Args:
            symbol: Símbolo de trading
            timeframe: Intervalo de tiempo
            num_velas: Número de velas
        Returns:
            DatosMercado o None si hay error
        """
        try:
            from ..api.clients import binance_client
            timestamp_inicio = datetime.now()
            self.logger.debug(f"🔍 [DATOS] Iniciando obtención de datos para {symbol} {timeframe} - {num_velas} velas")

            # Obtener datos de Binance
            datos_raw = binance_client.obtener_datos_klines(symbol, timeframe, num_velas + 14)
            if not datos_raw:
                self.logger.warning(f"⚠️ [DATOS] No se obtuvieron datos de Binance para {symbol} {timeframe}")
                return None

            # Organizar datos
            maximos = [float(vela[2]) for vela in datos_raw]
            minimos = [float(vela[3]) for vela in datos_raw]
            cierres = [float(vela[4]) for vela in datos_raw]
            tiempos = list(range(len(datos_raw)))

            # Logs detallados de los datos obtenidos
            precio_actual = cierres[-1] if cierres else 0
            timestamp_fin = datetime.now()
            tiempo_procesamiento = (timestamp_fin - timestamp_inicio).total_seconds()
            self.logger.info(f"📊 [DATOS] {symbol} {timeframe}:")
            self.logger.info(f" • Velas obtenidas: {len(datos_raw)} (solicitadas: {num_velas + 14})")
            self.logger.info(f" • Precio actual: {precio_actual:.8f}")
            self.logger.info(f" • Rango de precios: {min(minimos):.8f} - {max(maximos):.8f}")
            self.logger.info(f" • Timestamp datos más recientes: {datetime.fromtimestamp(datos_raw[-1][0]/1000).isoformat()}")
            self.logger.info(f" • Tiempo de procesamiento: {tiempo_procesamiento:.3f}s")

            self.stats['datos_obtenidos'] += 1

            return DatosMercado(
                maximos=maximos,
                minimos=minimos,
                cierres=cierres,
                tiempos=tiempos,
                precio_actual=precio_actual,
                timeframe=timeframe,
                num_velas=num_velas
            )
        except Exception as e:
            self.logger.error(f"❌ [DATOS] Error obteniendo datos de {symbol}: {e}")
            return None

    def calcular_regresion_lineal(self, x: List[int], y: List[float]) -> Optional[Tuple[float, float]]:
        """Calcula regresión lineal con logs detallados"""
        try:
            if len(x) != len(y) or len(x) == 0:
                self.logger.debug(f"🔍 [REGRESION] Datos inválidos: len(x)={len(x)}, len(y)={len(y)}")
                return None

            x_array = np.array(x)
            y_array = np.array(y)
            n = len(x)

            # Logs de validación inicial
            self.logger.debug(f"📈 [REGRESION] Calculando regresión para {n} puntos")
            self.logger.debug(f" • Rango X: {min(x)} a {max(x)}")
            self.logger.debug(f" • Rango Y: {min(y):.8f} a {max(y):.8f}")

            sum_x = np.sum(x_array)
            sum_y = np.sum(y_array)
            sum_xy = np.sum(x_array * y_array)
            sum_x2 = np.sum(x_array * x_array)

            denom = (n * sum_x2 - sum_x * sum_x)
            if denom == 0:
                pendiente = 0
                self.logger.debug("⚠️ [REGRESION] Denominador cero, pendiente = 0")
            else:
                pendiente = (n * sum_xy - sum_x * sum_y) / denom

            intercepto = (sum_y - pendiente * sum_x) / n if n else 0

            # Log del resultado
            self.logger.debug(f"✅ [REGRESION] Resultado: pendiente={pendiente:.8f}, intercepto={intercepto:.8f}")
            return pendiente, intercepto

        except Exception as e:
            self.logger.error(f"❌ [REGRESION] Error en regresión lineal: {e}")
            return None

    def calcular_pearson_y_angulo(self, x: List[int], y: List[float]) -> Tuple[float, float]:
        """Calcula coeficiente de Pearson y ángulo de tendencia con logs detallados"""
        try:
            if len(x) != len(y) or len(x) < 2:
                self.logger.debug(f"🔍 [PEARSON] Datos insuficientes: {len(x)} puntos")
                return 0, 0

            x_array = np.array(x)
            y_array = np.array(y)
            n = len(x)

            self.logger.debug(f"📊 [PEARSON] Analizando correlación para {n} puntos")

            sum_x = np.sum(x_array)
            sum_y = np.sum(y_array)
            sum_xy = np.sum(x_array * y_array)
            sum_x2 = np.sum(x_array * x_array)
            sum_y2 = np.sum(y_array * y_array)

            numerator = n * sum_xy - sum_x * sum_y
            denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))

            if denominator == 0:
                self.logger.debug("⚠️ [PEARSON] Denominador cero en correlación")
                return 0, 0

            pearson = numerator / denominator

            denom_pend = (n * sum_x2 - sum_x * sum_x)
            pendiente = (n * sum_xy - sum_x * sum_y) / denom_pend if denom_pend != 0 else 0

            # Evitar división por cero para el ángulo
            y_range = max(y) - min(y)
            angulo_radianes = math.atan(pendiente * len(x) / y_range) if y_range != 0 else 0
            angulo_grados = math.degrees(angulo_radianes)

            # Interpretación de la correlación
            if abs(pearson) >= 0.8:
                correlacion_texto = "MUY FUERTE"
            elif abs(pearson) >= 0.6:
                correlacion_texto = "FUERTE"
            elif abs(pearson) >= 0.4:
                correlacion_texto = "MODERADA"
            elif abs(pearson) >= 0.2:
                correlacion_texto = "DÉBIL"
            else:
                correlacion_texto = "MUY DÉBIL"

            self.logger.debug(f"✅ [PEARSON] Resultados:")
            self.logger.debug(f" • Coeficiente Pearson: {pearson:.4f} ({correlacion_texto})")
            self.logger.debug(f" • Ángulo tendencia: {angulo_grados:.2f}°")
            self.logger.debug(f" • Pendiente: {pendiente:.8f}")

            return pearson, angulo_grados

        except Exception as e:
            self.logger.error(f"❌ [PEARSON] Error calculando Pearson y ángulo: {e}")
            return 0, 0

    def calcular_r2(self, y_real: List[float], x: List[int], pendiente: float, intercepto: float) -> float:
        """Calcula R² score con logs detallados"""
        try:
            if len(y_real) != len(x):
                self.logger.debug(f"🔍 [R2] Datos inconsistentes: len(y_real)={len(y_real)}, len(x)={len(x)}")
                return 0

            y_real_array = np.array(y_real)
            y_pred = pendiente * np.array(x) + intercepto

            ss_res = np.sum((y_real_array - y_pred) ** 2)
            ss_tot = np.sum((y_real_array - np.mean(y_real_array)) ** 2)

            if ss_tot == 0:
                self.logger.debug("⚠️ [R2] Varianza total cero")
                return 0

            r2_score = 1 - (ss_res / ss_tot)

            # Interpretación del R²
            if r2_score >= 0.8:
                ajuste_texto = "EXCELENTE"
            elif r2_score >= 0.6:
                ajuste_texto = "BUENO"
            elif r2_score >= 0.4:
                ajuste_texto = "REGULAR"
            else:
                ajuste_texto = "POBRE"

            self.logger.debug(f"✅ [R2] Score: {r2_score:.4f} (Ajuste: {ajuste_texto})")
            self.logger.debug(f" • Varianza explicada: {ss_res:.4f}")
            self.logger.debug(f" • Varianza total: {ss_tot:.4f}")

            return r2_score

        except Exception as e:
            self.logger.error(f"❌ [R2] Error calculando R²: {e}")
            return 0

    def calcular_stochastic(self, datos: DatosMercado, period: int = None, k_period: int = None, d_period: int = None) -> Tuple[float, float]:
        """Calcula Stochastic K y D con logs detallados"""
        try:
            # Usar valores de Constants si no se proporcionan
            if period is None:
                period = Constants.PERIOD_STOCHASTIC
            if k_period is None:
                k_period = Constants.K_PERIOD
            if d_period is None:
                d_period = Constants.D_PERIOD

            if len(datos.cierres) < period:
                self.logger.debug(f"🔍 [STOCH] Datos insuficientes: {len(datos.cierres)} < {period}")
                return 50, 50

            cierres = datos.cierres
            maximos = datos.maximos
            minimos = datos.minimos

            self.logger.debug(f"📊 [STOCH] Calculando Stochastic (period={period}, k={k_period}, d={d_period})")

            k_values = []
            for i in range(period - 1, len(cierres)):
                highest_high = max(maximos[i - period + 1:i + 1])
                lowest_low = min(minimos[i - period + 1:i + 1])

                if highest_high == lowest_low:
                    k = 50
                else:
                    k = 100 * (cierres[i] - lowest_low) / (highest_high - lowest_low)
                k_values.append(k)

            if len(k_values) >= k_period:
                k_smoothed = []
                for i in range(k_period - 1, len(k_values)):
                    k_avg = sum(k_values[i - k_period + 1:i + 1]) / k_period
                    k_smoothed.append(k_avg)

                if len(k_smoothed) >= d_period:
                    d = sum(k_smoothed[-d_period:]) / d_period
                    k_final = k_smoothed[-1]

                    # Interpretación de Stochastic
                    if k_final >= 80:
                        stoch_texto_k = "SOBRECOMPRADO"
                    elif k_final <= 20:
                        stoch_texto_k = "SOBREVENDIDO"
                    else:
                        stoch_texto_k = "NEUTRAL"

                    if d >= 80:
                        stoch_texto_d = "SOBRECOMPRADO"
                    elif d <= 20:
                        stoch_texto_d = "SOBREVENDIDO"
                    else:
                        stoch_texto_d = "NEUTRAL"

                    self.logger.debug(f"✅ [STOCH] Resultados:")
                    self.logger.debug(f" • Stochastic K: {k_final:.2f} ({stoch_texto_k})")
                    self.logger.debug(f" • Stochastic D: {d:.2f} ({stoch_texto_d})")
                    return k_final, d

            self.logger.debug(f"⚠️ [STOCH] No se pudieron calcular valores válidos, retornando 50,50")
            return 50, 50

        except Exception as e:
            self.logger.error(f"❌ [STOCH] Error calculando Stochastic: {e}")
            return 50, 50

    def clasificar_fuerza_tendencia(self, angulo_grados: float) -> Tuple[str, int]:
        """Clasifica la fuerza de la tendencia con logs detallados"""
        try:
            angulo_abs = abs(angulo_grados)
            self.logger.debug(f"💪 [FUERZA] Clasificando tendencia: ángulo={angulo_grados:.2f}° (abs={angulo_abs:.2f}°)")

            if angulo_abs < 3:
                fuerza_texto = "💔 Muy Débil"
                nivel = 1
            elif angulo_abs < 13:
                fuerza_texto = "❤️‍🩹 Débil"
                nivel = 2
            elif angulo_abs < 27:
                fuerza_texto = "💛 Moderada"
                nivel = 3
            elif angulo_abs < 45:
                fuerza_texto = "💚 Fuerte"
                nivel = 4
            else:
                fuerza_texto = "💙 Muy Fuerte"
                nivel = 5

            self.logger.debug(f"✅ [FUERZA] Clasificación: {fuerza_texto} (nivel {nivel})")
            return fuerza_texto, nivel

        except Exception as e:
            self.logger.error(f"❌ [FUERZA] Error clasificando fuerza: {e}")
            return "💔 Muy Débil", 1

    def determinar_direccion_tendencia(self, angulo_grados: float, umbral_minimo: float = 1) -> str:
        """Determina la dirección de la tendencia con logs detallados"""
        try:
            angulo_abs = abs(angulo_grados)
            self.logger.debug(f"🧭 [DIRECCION] Analizando dirección: ángulo={angulo_grados:.2f}°, umbral={umbral_minimo}°")

            if angulo_abs < umbral_minimo:
                direccion = Constants.DIRECCION_RANGO
                direccion_emoji = "📊"
            elif angulo_grados > 0:
                direccion = Constants.DIRECCION_ALCISTA
                direccion_emoji = "📈"
            else:
                direccion = Constants.DIRECCION_BAJISTA
                direccion_emoji = "📉"

            self.logger.debug(f"✅ [DIRECCION] Resultado: {direccion_emoji} {direccion}")
            return direccion

        except Exception as e:
            self.logger.error(f"❌ [DIRECCION] Error determinando dirección: {e}")
            return Constants.DIRECCION_RANGO

    def calcular_canal_regresion(self, datos_mercado: DatosMercado, candle_period: int) -> Optional[CanalInfo]:
        """
        Calcula el canal de regresión con análisis completo y logs extensivos
        Args:
            datos_mercado: Datos del mercado
            candle_period: Período de velas para análisis
        Returns:
            CanalInfo o None si hay error
        """
        try:
            timestamp_inicio = datetime.now()
            if not datos_mercado or len(datos_mercado.maximos) < candle_period:
                self.logger.debug(f"🔍 [CANAL] Datos insuficientes: {len(datos_mercado.maximos) if datos_mercado else 0} < {candle_period}")
                return None

            self.logger.info(f"🏗️ [CANAL] Calculando canal de regresión ({candle_period} velas)")
            self.logger.info(f" • Timeframe: {datos_mercado.timeframe}")
            self.logger.info(f" • Precio actual: {datos_mercado.precio_actual:.8f}")

            # Extraer datos del período
            start_idx = -candle_period
            tiempos = datos_mercado.tiempos[start_idx:]
            maximos = datos_mercado.maximos[start_idx:]
            minimos = datos_mercado.minimos[start_idx:]
            cierres = datos_mercado.cierres[start_idx:]
            tiempos_reg = list(range(len(tiempos)))

            self.logger.debug(f"📊 [CANAL] Datos extraídos del período:")
            self.logger.debug(f" • Velas analizadas: {len(tiempos)}")
            self.logger.debug(f" • Rango de precios: {min(minimos):.8f} - {max(maximos):.8f}")

            # Calcular regresiones
            reg_max = self.calcular_regresion_lineal(tiempos_reg, maximos)
            reg_min = self.calcular_regresion_lineal(tiempos_reg, minimos)
            reg_close = self.calcular_regresion_lineal(tiempos_reg, cierres)

            if not all([reg_max, reg_min, reg_close]):
                self.logger.warning(f"⚠️ [CANAL] No se pudieron calcular todas las regresiones")
                return None

            pendiente_max, intercepto_max = reg_max
            pendiente_min, intercepto_min = reg_min
            pendiente_cierre, intercepto_cierre = reg_close

            tiempo_actual = tiempos_reg[-1]

            resistencia_media = pendiente_max * tiempo_actual + intercepto_max
            soporte_media = pendiente_min * tiempo_actual + intercepto_min

            # Calcular desviaciones
            diferencias_max = [maximos[i] - (pendiente_max * tiempos_reg[i] + intercepto_max) for i in range(len(tiempos_reg))]
            diferencias_min = [minimos[i] - (pendiente_min * tiempos_reg[i] + intercepto_min) for i in range(len(tiempos_reg))]

            desviacion_max = np.std(diferencias_max) if diferencias_max else 0
            desviacion_min = np.std(diferencias_min) if diferencias_min else 0

            # Líneas del canal
            resistencia_superior = resistencia_media + desviacion_max
            soporte_inferior = soporte_media - desviacion_min

            precio_actual = datos_mercado.precio_actual

            # Análisis técnico con logs
            self.logger.debug(f"📈 [CANAL] Iniciando análisis técnico...")
            pearson, angulo_tendencia = self.calcular_pearson_y_angulo(tiempos_reg, cierres)
            fuerza_texto, nivel_fuerza = self.clasificar_fuerza_tendencia(angulo_tendencia)
            direccion = self.determinar_direccion_tendencia(angulo_tendencia, 1)
            stoch_k, stoch_d = self.calcular_stochastic(datos_mercado)

            # Cálculos adicionales
            precio_medio = (resistencia_superior + soporte_inferior) / 2
            ancho_canal_absoluto = resistencia_superior - soporte_inferior
            ancho_canal_porcentual = (ancho_canal_absoluto / precio_medio) * 100
            r2_score = self.calcular_r2(cierres, tiempos_reg, pendiente_cierre, intercepto_cierre)

            # Log completo del canal calculado
            self.logger.info(f"✅ [CANAL] Canal calculado exitosamente:")
            self.logger.info(f" • Resistencia: {resistencia_superior:.8f}")
            self.logger.info(f" • Soporte: {soporte_inferior:.8f}")
            self.logger.info(f" • Ancho canal: {ancho_canal_absoluto:.8f} ({ancho_canal_porcentual:.2f}%)")
            self.logger.info(f" • Dirección: {direccion}")
            self.logger.info(f" • Fuerza: {fuerza_texto} (nivel {nivel_fuerza})")
            self.logger.info(f" • Pearson: {pearson:.4f}")
            self.logger.info(f" • R²: {r2_score:.4f}")
            self.logger.info(f" • Stochastic: K={stoch_k:.2f}, D={stoch_d:.2f}")

            timestamp_fin = datetime.now()
            tiempo_calculo = (timestamp_fin - timestamp_inicio).total_seconds()
            self.logger.debug(f"⏱️ [CANAL] Tiempo de cálculo: {tiempo_calculo:.3f}s")

            self.stats['canales_calculados'] += 1

            return CanalInfo(
                resistencia=resistencia_superior,
                soporte=soporte_inferior,
                resistencia_media=resistencia_media,
                soporte_media=soporte_media,
                linea_tendencia=pendiente_cierre * tiempo_actual + intercepto_cierre,
                pendiente_tendencia=pendiente_cierre,
                precio_actual=precio_actual,
                ancho_canal=ancho_canal_absoluto,
                ancho_canal_porcentual=ancho_canal_porcentual,
                angulo_tendencia=angulo_tendencia,
                coeficiente_pearson=pearson,
                fuerza_texto=fuerza_texto,
                nivel_fuerza=nivel_fuerza,
                direccion=direccion,
                r2_score=r2_score,
                pendiente_resistencia=pendiente_max,
                pendiente_soporte=pendiente_min,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                timeframe=datos_mercado.timeframe,
                num_velas=candle_period
            )

        except Exception as e:
            self.logger.error(f"❌ [CANAL] Error calculando canal de regresión: {e}")
            return None

    def detectar_breakout(self, symbol: str, canal_info: CanalInfo, datos_mercado: DatosMercado) -> Optional[str]:
        """
        Detecta si el precio ha roto el canal con logs extensivos
        CORRECCIÓN: Lógica de breakout corregida para coincidir con la estrategia original
        Args:
            symbol: Símbolo de trading
            canal_info: Información del canal
            datos_mercado: Datos del mercado
        Returns:
            Tipo de breakout o None
        """
        try:
            timestamp_inicio = datetime.now()
            if not canal_info:
                self.logger.debug(f"🔍 [BREAKOUT] {symbol}: No hay información de canal")
                return None

            self.logger.debug(f"🔍 [BREAKOUT] {symbol}: Iniciando detección...")
            self.logger.debug(f" • Precio actual: {datos_mercado.precio_actual:.8f}")
            self.logger.debug(f" • Resistencia: {canal_info.resistencia:.8f}")
            self.logger.debug(f" • Soporte: {canal_info.soporte:.8f}")
            self.logger.debug(f" • Ancho canal: {canal_info.ancho_canal_porcentual:.2f}%")

            # Filtro 1: Ancho mínimo del canal
            if canal_info.ancho_canal_porcentual < config.min_channel_width_percent:
                self.logger.debug(f"❌ [BREAKOUT] {symbol}: Canal muy estrecho ({canal_info.ancho_canal_porcentual:.2f}% < {config.min_channel_width_percent}%)")
                return None

            precio_cierre = datos_mercado.precio_actual
            resistencia = canal_info.resistencia
            soporte = canal_info.soporte
            angulo = canal_info.angulo_tendencia
            direccion = canal_info.direccion
            nivel_fuerza = canal_info.nivel_fuerza
            r2 = canal_info.r2_score
            pearson = canal_info.coeficiente_pearson

            # Filtro 2: Fuerza mínima de tendencia
            if abs(angulo) < config.min_trend_strength_degrees:
                self.logger.debug(f"❌ [BREAKOUT] {symbol}: Tendencia muy débil ({abs(angulo):.1f}° < {config.min_trend_strength_degrees}°)")
                return None

            # Filtro 3: Calidad de correlación
            if abs(pearson) < Constants.MIN_PEARSON or r2 < Constants.MIN_R2_SCORE:
                self.logger.debug(f"❌ [BREAKOUT] {symbol}: Calidad insuficiente (Pearson: {pearson:.3f}, R²: {r2:.3f})")
                return None

            # Todos los filtros pasaron - evaluar breakout
            self.logger.debug(f"✅ [BREAKOUT] {symbol}: Todos los filtros pasaron")
            self.logger.debug(f" • Dirección: {direccion}")
            self.logger.debug(f" • Nivel fuerza: {nivel_fuerza} (min: {Constants.MIN_NIVEL_FUERZA})")

            # CORRECCIÓN: Lógica de breakout corregida según la estrategia original
            # BREAKOUT_LONG: Ruptura de SOPORTE hacia arriba (salida por abajo del canal)
            # BREAKOUT_SHORT: Ruptura de RESISTENCIA hacia abajo (salida por arriba del canal)
            if direccion == Constants.DIRECCION_ALCISTA and nivel_fuerza >= Constants.MIN_NIVEL_FUERZA:
                if precio_cierre < soporte:  # CORREGIDO: Ruptura de SOPORTE hacia arriba
                    timestamp_fin = datetime.now()
                    tiempo_deteccion = (timestamp_fin - timestamp_inicio).total_seconds()
                    self.logger.info(f"🚀 [BREAKOUT] {symbol} - BREAKOUT LONG DETECTADO:")
                    self.logger.info(f" • Precio cierre: {precio_cierre:.8f}")
                    self.logger.info(f" • Soporte: {soporte:.8f}")
                    self.logger.info(f" • Diferencia: {(soporte - precio_cierre):.8f} (+{((soporte/precio_cierre - 1) * 100):.3f}%)")
                    self.logger.info(f" • Dirección: {direccion}")
                    self.logger.info(f" • Tiempo detección: {tiempo_deteccion:.3f}s")
                    self.stats['breakouts_detectados'] += 1
                    return Constants.BREAKOUT_LONG
            elif direccion == Constants.DIRECCION_BAJISTA and nivel_fuerza >= Constants.MIN_NIVEL_FUERZA:
                if precio_cierre > resistencia:  # CORREGIDO: Ruptura de RESISTENCIA hacia abajo
                    timestamp_fin = datetime.now()
                    tiempo_deteccion = (timestamp_fin - timestamp_inicio).total_seconds()
                    self.logger.info(f"📉 [BREAKOUT] {symbol} - BREAKOUT SHORT DETECTADO:")
                    self.logger.info(f" • Precio cierre: {precio_cierre:.8f}")
                    self.logger.info(f" • Resistencia: {resistencia:.8f}")
                    self.logger.info(f" • Diferencia: {(precio_cierre - resistencia):.8f} (+{((precio_cierre/resistencia - 1) * 100):.3f}%)")
                    self.logger.info(f" • Dirección: {direccion}")
                    self.logger.info(f" • Tiempo detección: {tiempo_deteccion:.3f}s")
                    self.stats['breakouts_detectados'] += 1
                    return Constants.BREAKOUT_SHORT

            self.logger.debug(f"💤 [BREAKOUT] {symbol}: No hay breakout activo")
            return None

        except Exception as e:
            self.logger.error(f"❌ [BREAKOUT] Error detectando breakout en {symbol}: {e}")
            return None

    def detectar_reentry(self, symbol: str, canal_info: CanalInfo, datos_mercado: DatosMercado,
                         breakout_info: Dict[str, Any]) -> Optional[str]:
        """
        Detecta si el precio ha reingresado al canal con logs extensivos
        Args:
            symbol: Símbolo de trading
            canal_info: Información del canal
            datos_mercado: Datos del mercado
            breakout_info: Información del breakout anterior
        Returns:
            Tipo de operación o None
        """
        try:
            timestamp_inicio = datetime.now()
            if not breakout_info:
                self.logger.debug(f"🔍 [REENTRY] {symbol}: No hay información de breakout previo")
                return None

            tipo_breakout = breakout_info['tipo']
            timestamp_breakout = breakout_info['timestamp']

            # Verificar timeout
            tiempo_desde_breakout = (datetime.now() - timestamp_breakout).total_seconds() / 60
            self.logger.debug(f"⏰ [REENTRY] {symbol}: Tiempo desde breakout: {tiempo_desde_breakout:.1f} min")
            if tiempo_desde_breakout > Constants.TIMEOUT_REENTRY_MINUTOS:
                self.logger.info(f"⏰ [REENTRY] {symbol}: Timeout de reentry ({tiempo_desde_breakout:.1f} > {Constants.TIMEOUT_REENTRY_MINUTOS} min), cancelando espera")
                return None

            precio_actual = datos_mercado.precio_actual
            resistencia = canal_info.resistencia
            soporte = canal_info.soporte
            stoch_k = canal_info.stoch_k
            stoch_d = canal_info.stoch_d
            tolerancia = 0.001 * precio_actual # 0.1% de tolerancia para el reingreso

            self.logger.debug(f"🔍 [REENTRY] {symbol}: Analizando reentry...")
            self.logger.debug(f" • Tipo breakout: {tipo_breakout}")
            self.logger.debug(f" • Precio actual: {precio_actual:.8f}")
            self.logger.debug(f" • Rango canal: {soporte:.8f} - {resistencia:.8f}")
            self.logger.debug(f" • Stochastic K: {stoch_k:.2f}")
            self.logger.debug(f" • Tolerancia: {tolerancia:.8f}")

            if tipo_breakout == Constants.BREAKOUT_LONG:
                # Verificar reentry desde soporte hacia arriba
                if soporte <= precio_actual <= resistencia:
                    distancia_soporte = abs(precio_actual - soporte)
                    self.logger.debug(f"📊 [REENTRY] {symbol}: Precio dentro del canal")
                    self.logger.debug(f" • Distancia a soporte: {distancia_soporte:.8f}")
                    if distancia_soporte <= tolerancia and stoch_k <= Constants.STOCH_OVERSOLD:
                        timestamp_fin = datetime.now()
                        tiempo_deteccion = (timestamp_fin - timestamp_inicio).total_seconds()
                        self.logger.info(f"✅ [REENTRY] {symbol} - REENTRY LONG CONFIRMADO:")
                        self.logger.info(f" • Precio entrada: {precio_actual:.8f}")
                        self.logger.info(f" • Soporte: {soporte:.8f}")
                        self.logger.info(f" • Stochastic K: {stoch_k:.2f} (<= {Constants.STOCH_OVERSOLD})")
                        self.logger.info(f" • Distancia soporte: {distancia_soporte:.8f}")
                        self.logger.info(f" • Tiempo detección: {tiempo_deteccion:.3f}s")
                        self.stats['reentries_detectados'] += 1
                        return Constants.OPERACION_LONG
                    else:
                        if distancia_soporte > tolerancia:
                            self.logger.debug(f"❌ [REENTRY] {symbol}: Lejos del soporte ({distancia_soporte:.8f} > {tolerancia:.8f})")
                        if stoch_k > Constants.STOCH_OVERSOLD:
                            self.logger.debug(f"❌ [REENTRY] {symbol}: Stochastic K muy alto ({stoch_k:.2f} > {Constants.STOCH_OVERSOLD})")
            elif tipo_breakout == Constants.BREAKOUT_SHORT:
                # Verificar reentry desde resistencia hacia abajo
                if soporte <= precio_actual <= resistencia:
                    distancia_resistencia = abs(precio_actual - resistencia)
                    self.logger.debug(f"📊 [REENTRY] {symbol}: Precio dentro del canal")
                    self.logger.debug(f" • Distancia a resistencia: {distancia_resistencia:.8f}")
                    if distancia_resistencia <= tolerancia and stoch_k >= Constants.STOCH_OVERBOUGHT:
                        timestamp_fin = datetime.now()
                        tiempo_deteccion = (timestamp_fin - timestamp_inicio).total_seconds()
                        self.logger.info(f"✅ [REENTRY] {symbol} - REENTRY SHORT CONFIRMADO:")
                        self.logger.info(f" • Precio entrada: {precio_actual:.8f}")
                        self.logger.info(f" • Resistencia: {resistencia:.8f}")
                        self.logger.info(f" • Stochastic K: {stoch_k:.2f} (>= {Constants.STOCH_OVERBOUGHT})")
                        self.logger.info(f" • Distancia resistencia: {distancia_resistencia:.8f}")
                        self.logger.info(f" • Tiempo detección: {tiempo_deteccion:.3f}s")
                        self.stats['reentries_detectados'] += 1
                        return Constants.OPERACION_SHORT
                    else:
                        if distancia_resistencia > tolerancia:
                            self.logger.debug(f"❌ [REENTRY] {symbol}: Lejos de la resistencia ({distancia_resistencia:.8f} > {tolerancia:.8f})")
                        if stoch_k < Constants.STOCH_OVERBOUGHT:
                            self.logger.debug(f"❌ [REENTRY] {symbol}: Stochastic K muy bajo ({stoch_k:.2f} < {Constants.STOCH_OVERBOUGHT})")

            self.logger.debug(f"💤 [REENTRY] {symbol}: No hay reentry confirmado")
            return None

        except Exception as e:
            self.logger.error(f"❌ [REENTRY] Error detectando reentry en {symbol}: {e}")
            return None

    def calcular_niveles_entrada(self, tipo_operacion: str, canal_info: CanalInfo, precio_actual: float) -> Tuple[float, float, float]:
        """
        Calcula niveles de entrada, take profit y stop loss con logs detallados
        Args:
            tipo_operacion: Tipo de operación (LONG/SHORT)
            canal_info: Información del canal
            precio_actual: Precio actual
        Returns:
            Tupla con (entrada, take_profit, stop_loss)
        """
        try:
            timestamp_inicio = datetime.now()
            if not canal_info:
                self.logger.debug(f"🔍 [NIVELES] No hay información de canal")
                return None, None, None

            self.logger.debug(f"💰 [NIVELES] Calculando niveles para operación {tipo_operacion}")
            self.logger.debug(f" • Precio actual: {precio_actual:.8f}")
            self.logger.debug(f" • Canal: {canal_info.soporte:.8f} - {canal_info.resistencia:.8f}")

            resistencia = canal_info.resistencia
            soporte = canal_info.soporte
            ancho_canal = resistencia - soporte
            sl_porcentaje = 0.02 # Ejemplo, se podría parametrizar

            if tipo_operacion == Constants.OPERACION_LONG:
                precio_entrada = precio_actual
                stop_loss = precio_entrada * (1 - sl_porcentaje)
                take_profit = precio_entrada + ancho_canal
            else: # SHORT
                precio_entrada = precio_actual
                stop_loss = resistencia * (1 + sl_porcentaje)
                take_profit = precio_entrada - ancho_canal

            # Calcular ratio riesgo/beneficio
            riesgo = abs(precio_entrada - stop_loss)
            beneficio = abs(take_profit - precio_entrada)
            ratio_rr = beneficio / riesgo if riesgo > 0 else 0

            self.logger.debug(f"📊 [NIVELES] Cálculo inicial:")
            self.logger.debug(f" • Entrada: {precio_entrada:.8f}")
            self.logger.debug(f" • Take Profit: {take_profit:.8f}")
            self.logger.debug(f" • Stop Loss: {stop_loss:.8f}")
            self.logger.debug(f" • Riesgo: {riesgo:.8f}")
            self.logger.debug(f" • Beneficio: {beneficio:.8f}")
            self.logger.debug(f" • Ratio R/R: {ratio_rr:.2f}")

            # Ajustar take profit si el ratio es muy bajo
            if ratio_rr < config.min_rr_ratio:
                ratio_original = ratio_rr
                if tipo_operacion == Constants.OPERACION_LONG:
                    take_profit = precio_entrada + (riesgo * config.min_rr_ratio)
                else:
                    take_profit = precio_entrada - (riesgo * config.min_rr_ratio)

                # Recalcular ratio
                beneficio_ajustado = abs(take_profit - precio_entrada)
                ratio_rr_ajustado = beneficio_ajustado / riesgo if riesgo > 0 else 0

                self.logger.debug(f"⚠️ [NIVELES] Ratio R/R ajustado:")
                self.logger.debug(f" • Ratio original: {ratio_original:.2f} < {config.min_rr_ratio}")
                self.logger.debug(f" • Take Profit ajustado: {take_profit:.8f}")
                self.logger.debug(f" • Nuevo ratio R/R: {ratio_rr_ajustado:.2f}")

            timestamp_fin = datetime.now()
            tiempo_calculo = (timestamp_fin - timestamp_inicio).total_seconds()
            self.logger.info(f"✅ [NIVELES] {tipo_operacion} calculados:")
            self.logger.info(f" • Entrada: {precio_entrada:.8f}")
            self.logger.info(f" • Take Profit: {take_profit:.8f}")
            self.logger.info(f" • Stop Loss: {stop_loss:.8f}")
            self.logger.info(f" • Ratio R/R: {ratio_rr:.2f}")
            self.logger.info(f" • Tiempo cálculo: {tiempo_calculo:.3f}s")

            self.stats['operaciones_calculadas'] += 1
            return precio_entrada, take_profit, stop_loss

        except Exception as e:
            self.logger.error(f"❌ [NIVELES] Error calculando niveles de entrada: {e}")
            return None, None, None

    def obtener_estadisticas_logs(self) -> Dict[str, int]:
        """Obtiene estadísticas de los logs generados"""
        return self.stats.copy()

    def reset_estadisticas_logs(self):
        """Reinicia las estadísticas de logs"""
        self.stats = {
            'datos_obtenidos': 0,
            'regresiones_calculadas': 0,
            'canales_calculados': 0,
            'breakouts_detectados': 0,
            'reentries_detectados': 0,
            'operaciones_calculadas': 0
        }
        self.logger.info("🔄 [STATS] Estadísticas de logs reiniciadas")

# Instancia global de la estrategia
estrategia = EstrategiaBreakoutReentry()
print("📈 Estrategia Breakout + Reentry con logs detallados cargada correctamente")
