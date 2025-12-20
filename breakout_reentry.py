"""
Estrategias de trading.
Contiene toda la lógica de análisis técnico y señales de trading.
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
            
            # Obtener datos de Binance
            datos_raw = binance_client.obtener_datos_klines(symbol, timeframe, num_velas + 14)
            if not datos_raw:
                return None
                
            # Organizar datos
            maximos = [float(vela[2]) for vela in datos_raw]
            minimos = [float(vela[3]) for vela in datos_raw]
            cierres = [float(vela[4]) for vela in datos_raw]
            tiempos = list(range(len(datos_raw)))
            
            return DatosMercado(
                maximos=maximos,
                minimos=minimos,
                cierres=cierres,
                tiempos=tiempos,
                precio_actual=cierres[-1] if cierres else 0,
                timeframe=timeframe,
                num_velas=num_velas
            )
            
        except Exception as e:
            self.logger.error(f"Error obteniendo datos de {symbol}: {e}")
            return None
    
    def calcular_regresion_lineal(self, x: List[int], y: List[float]) -> Optional[Tuple[float, float]]:
        """Calcula regresión lineal"""
        try:
            if len(x) != len(y) or len(x) == 0:
                return None
                
            x_array = np.array(x)
            y_array = np.array(y)
            n = len(x)
            
            sum_x = np.sum(x_array)
            sum_y = np.sum(y_array)
            sum_xy = np.sum(x_array * y_array)
            sum_x2 = np.sum(x_array * x_array)
            
            denom = (n * sum_x2 - sum_x * sum_x)
            if denom == 0:
                pendiente = 0
            else:
                pendiente = (n * sum_xy - sum_x * sum_y) / denom
                
            intercepto = (sum_y - pendiente * sum_x) / n if n else 0
            
            return pendiente, intercepto
            
        except Exception as e:
            self.logger.error(f"Error en regresión lineal: {e}")
            return None
    
    def calcular_pearson_y_angulo(self, x: List[int], y: List[float]) -> Tuple[float, float]:
        """Calcula coeficiente de Pearson y ángulo de tendencia"""
        try:
            if len(x) != len(y) or len(x) < 2:
                return 0, 0
                
            x_array = np.array(x)
            y_array = np.array(y)
            n = len(x)
            
            sum_x = np.sum(x_array)
            sum_y = np.sum(y_array)
            sum_xy = np.sum(x_array * y_array)
            sum_x2 = np.sum(x_array * x_array)
            sum_y2 = np.sum(y_array * y_array)
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
            
            if denominator == 0:
                return 0, 0
                
            pearson = numerator / denominator
            
            denom_pend = (n * sum_x2 - sum_x * sum_x)
            pendiente = (n * sum_xy - sum_x * sum_y) / denom_pend if denom_pend != 0 else 0
            
            angulo_radianes = math.atan(pendiente * len(x) / (max(y) - min(y)) if (max(y) - min(y)) != 0 else 0)
            angulo_grados = math.degrees(angulo_radianes)
            
            return pearson, angulo_grados
            
        except Exception as e:
            self.logger.error(f"Error calculando Pearson y ángulo: {e}")
            return 0, 0
    
    def calcular_r2(self, y_real: List[float], x: List[int], pendiente: float, intercepto: float) -> float:
        """Calcula R² score"""
        try:
            if len(y_real) != len(x):
                return 0
                
            y_real_array = np.array(y_real)
            y_pred = pendiente * np.array(x) + intercepto
            
            ss_res = np.sum((y_real_array - y_pred) ** 2)
            ss_tot = np.sum((y_real_array - np.mean(y_real_array)) ** 2)
            
            if ss_tot == 0:
                return 0
                
            return 1 - (ss_res / ss_tot)
            
        except Exception as e:
            self.logger.error(f"Error calculando R²: {e}")
            return 0
    
    def calcular_stochastic(self, datos: DatosMercado, period: int = Constants.PERIOD_STOCHASTIC, 
                          k_period: int = Constants.K_PERIOD, d_period: int = Constants.D_PERIOD) -> Tuple[float, float]:
        """Calcula Stochastic K y D"""
        try:
            if len(datos.cierres) < period:
                return 50, 50
                
            cierres = datos.cierres
            maximos = datos.maximos
            minimos = datos.minimos
            
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
                    return k_final, d
                    
            return 50, 50
            
        except Exception as e:
            self.logger.error(f"Error calculando Stochastic: {e}")
            return 50, 50
    
    def clasificar_fuerza_tendencia(self, angulo_grados: float) -> Tuple[str, int]:
        """Clasifica la fuerza de la tendencia"""
        try:
            angulo_abs = abs(angulo_grados)
            
            if angulo_abs < 3:
                return "💔 Muy Débil", 1
            elif angulo_abs < 13:
                return "❤️‍🩹 Débil", 2
            elif angulo_abs < 27:
                return "💛 Moderada", 3
            elif angulo_abs < 45:
                return "💚 Fuerte", 4
            else:
                return "💙 Muy Fuerte", 5
                
        except Exception as e:
            self.logger.error(f"Error clasificando fuerza: {e}")
            return "💔 Muy Débil", 1
    
    def determinar_direccion_tendencia(self, angulo_grados: float, umbral_minimo: float = 1) -> str:
        """Determina la dirección de la tendencia"""
        try:
            if abs(angulo_grados) < umbral_minimo:
                return Constants.DIRECCION_RANGO
            elif angulo_grados > 0:
                return Constants.DIRECCION_ALCISTA
            else:
                return Constants.DIRECCION_BAJISTA
                
        except Exception as e:
            self.logger.error(f"Error determinando dirección: {e}")
            return Constants.DIRECCION_RANGO
    
    def calcular_canal_regresion(self, datos_mercado: DatosMercado, candle_period: int) -> Optional[CanalInfo]:
        """
        Calcula el canal de regresión con análisis completo
        
        Args:
            datos_mercado: Datos del mercado
            candle_period: Período de velas para análisis
            
        Returns:
            CanalInfo o None si hay error
        """
        try:
            if not datos_mercado or len(datos_mercado.maximos) < candle_period:
                return None
                
            # Extraer datos del período
            start_idx = -candle_period
            tiempos = datos_mercado.tiempos[start_idx:]
            maximos = datos_mercado.maximos[start_idx:]
            minimos = datos_mercado.minimos[start_idx:]
            cierres = datos_mercado.cierres[start_idx:]
            
            tiempos_reg = list(range(len(tiempos)))
            
            # Calcular regresiones
            reg_max = self.calcular_regresion_lineal(tiempos_reg, maximos)
            reg_min = self.calcular_regresion_lineal(tiempos_reg, minimos)
            reg_close = self.calcular_regresion_lineal(tiempos_reg, cierres)
            
            if not all([reg_max, reg_min, reg_close]):
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
            
            # Análisis técnico
            pearson, angulo_tendencia = self.calcular_pearson_y_angulo(tiempos_reg, cierres)
            fuerza_texto, nivel_fuerza = self.clasificar_fuerza_tendencia(angulo_tendencia)
            direccion = self.determinar_direccion_tendencia(angulo_tendencia, 1)
            
            stoch_k, stoch_d = self.calcular_stochastic(datos_mercado)
            
            # Cálculos adicionales
            precio_medio = (resistencia_superior + soporte_inferior) / 2
            ancho_canal_absoluto = resistencia_superior - soporte_inferior
            ancho_canal_porcentual = (ancho_canal_absoluto / precio_medio) * 100
            
            r2_score = self.calcular_r2(cierres, tiempos_reg, pendiente_cierre, intercepto_cierre)
            
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
            self.logger.error(f"Error calculando canal de regresión: {e}")
            return None
    
    def detectar_breakout(self, symbol: str, canal_info: CanalInfo, datos_mercado: DatosMercado) -> Optional[str]:
        """
        Detecta si el precio ha roto el canal
        
        Args:
            symbol: Símbolo de trading
            canal_info: Información del canal
            datos_mercado: Datos del mercado
            
        Returns:
            Tipo de breakout o None
        """
        try:
            if not canal_info:
                return None
                
            if canal_info.ancho_canal_porcentual < config.min_channel_width_percent:
                return None
                
            precio_cierre = datos_mercado.precio_actual
            resistencia = canal_info.resistencia
            soporte = canal_info.soporte
            angulo = canal_info.angulo_tendencia
            direccion = canal_info.direccion
            nivel_fuerza = canal_info.nivel_fuerza
            r2 = canal_info.r2_score
            pearson = canal_info.coeficiente_pearson
            
            # Filtros de calidad
            if abs(angulo) < config.min_trend_strength_degrees:
                return None
                
            if abs(pearson) < Constants.MIN_PEARSON or r2 < Constants.MIN_R2_SCORE:
                return None
                
            # Detección de breakout
            if direccion == Constants.DIRECCION_ALCISTA and nivel_fuerza >= Constants.MIN_NIVEL_FUERZA:
                if precio_cierre > resistencia:
                    self.logger.info(f"🚀 {symbol} - BREAKOUT LONG: {precio_cierre:.8f} > Resistencia: {resistencia:.8f}")
                    return Constants.BREAKOUT_LONG
                    
            elif direccion == Constants.DIRECCION_BAJISTA and nivel_fuerza >= Constants.MIN_NIVEL_FUERZA:
                if precio_cierre < soporte:
                    self.logger.info(f"📉 {symbol} - BREAKOUT SHORT: {precio_cierre:.8f} < Soporte: {soporte:.8f}")
                    return Constants.BREAKOUT_SHORT
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error detectando breakout en {symbol}: {e}")
            return None
    
    def detectar_reentry(self, symbol: str, canal_info: CanalInfo, datos_mercado: DatosMercado, 
                        breakout_info: Dict[str, Any]) -> Optional[str]:
        """
        Detecta si el precio ha reingresado al canal
        
        Args:
            symbol: Símbolo de trading
            canal_info: Información del canal
            datos_mercado: Datos del mercado
            breakout_info: Información del breakout anterior
            
        Returns:
            Tipo de operación o None
        """
        try:
            if not breakout_info:
                return None
                
            tipo_breakout = breakout_info['tipo']
            timestamp_breakout = breakout_info['timestamp']
            
            # Verificar timeout
            tiempo_desde_breakout = (datetime.now() - timestamp_breakout).total_seconds() / 60
            if tiempo_desde_breakout > Constants.TIMEOUT_REENTRY_MINUTOS:
                self.logger.info(f"⏰ {symbol} - Timeout de reentry (>120 min), cancelando espera")
                return None
                
            precio_actual = datos_mercado.precio_actual
            resistencia = canal_info.resistencia
            soporte = canal_info.soporte
            stoch_k = canal_info.stoch_k
            stoch_d = canal_info.stoch_d
            
            tolerancia = 0.001 * precio_actual
            
            if tipo_breakout == Constants.BREAKOUT_LONG:
                # Verificar reentry desde soporte hacia arriba
                if soporte <= precio_actual <= resistencia:
                    distancia_soporte = abs(precio_actual - soporte)
                    if distancia_soporte <= tolerancia and stoch_k <= Constants.STOCH_OVERSOLD:
                        self.logger.info(f"✅ {symbol} - REENTRY LONG confirmado!")
                        return Constants.OPERACION_LONG
                        
            elif tipo_breakout == Constants.BREAKOUT_SHORT:
                # Verificar reentry desde resistencia hacia abajo
                if soporte <= precio_actual <= resistencia:
                    distancia_resistencia = abs(precio_actual - resistencia)
                    if distancia_resistencia <= tolerancia and stoch_k >= Constants.STOCH_OVERBOUGHT:
                        self.logger.info(f"✅ {symbol} - REENTRY SHORT confirmado!")
                        return Constants.OPERACION_SHORT
                        
            return None
            
        except Exception as e:
            self.logger.error(f"Error detectando reentry en {symbol}: {e}")
            return None
    
    def calcular_niveles_entrada(self, tipo_operacion: str, canal_info: CanalInfo, precio_actual: float) -> Tuple[float, float, float]:
        """
        Calcula niveles de entrada, take profit y stop loss
        
        Args:
            tipo_operacion: Tipo de operación (LONG/SHORT)
            canal_info: Información del canal
            precio_actual: Precio actual
            
        Returns:
            Tupla con (entrada, take_profit, stop_loss)
        """
        try:
            if not canal_info:
                return None, None, None
                
            resistencia = canal_info.resistencia
            soporte = canal_info.soporte
            ancho_canal = resistencia - soporte
            sl_porcentaje = 0.02
            
            if tipo_operacion == Constants.OPERACION_LONG:
                precio_entrada = precio_actual
                stop_loss = precio_entrada * (1 - sl_porcentaje)
                take_profit = precio_entrada + ancho_canal
                
            else:  # SHORT
                precio_entrada = precio_actual
                stop_loss = resistencia * (1 + sl_porcentaje)
                take_profit = precio_entrada - ancho_canal
                
            # Calcular ratio riesgo/beneficio
            riesgo = abs(precio_entrada - stop_loss)
            beneficio = abs(take_profit - precio_entrada)
            ratio_rr = beneficio / riesgo if riesgo > 0 else 0
            
            # Ajustar take profit si el ratio es muy bajo
            if ratio_rr < config.min_rr_ratio:
                if tipo_operacion == Constants.OPERACION_LONG:
                    take_profit = precio_entrada + (riesgo * config.min_rr_ratio)
                else:
                    take_profit = precio_entrada - (riesgo * config.min_rr_ratio)
                    
            return precio_entrada, take_profit, stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculando niveles de entrada: {e}")
            return None, None, None

# Instancia global de la estrategia
estrategia = EstrategiaBreakoutReentry()

print("📈 Estrategia Breakout + Reentry cargada correctamente")
