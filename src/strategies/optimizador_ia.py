"""
Optimizador de IA con logs detallados.
Encuentra los mejores parámetros de trading analizando datos históricos.
MEJORADO CON LOGS EXTENSIVOS PARA MAYOR VISIBILIDAD
"""
import os
import json
import logging
import statistics
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from ..config.settings import config, Constants

class OptimizadorIA:
    """Optimizador de IA con logs extensivos para encontrar mejores parámetros de trading"""

    def __init__(self, log_path: str = None, min_samples: int = Constants.MIN_MUESTRAS_OPTIMIZACION):
        """
        Inicializa el optimizador de IA
        Args:
            log_path: Ruta del archivo CSV con operaciones históricas
            min_samples: Mínimo de muestras requeridas para optimización
        """
        self.logger = logging.getLogger(__name__)
        self.log_path = log_path or config.log_path
        self.min_samples = min_samples
        self.datos = []
        self.stats_optimizacion = {
            'optimizaciones_realizadas': 0,
            'parametros_evaluados': 0,
            'mejores_configuraciones_encontradas': 0,
            'tiempo_total_optimizacion': 0.0,
            'datos_cargados': 0,
            'filtros_aplicados': 0
        }

        self.logger.info("🧠 [OPTIMIZADOR] Inicializando Optimizador de IA")
        self.logger.info(f"   • Archivo de datos: {self.log_path}")
        self.logger.info(f"   • Mínimo muestras: {self.min_samples}")
        
        # Cargar datos históricos
        self.cargar_datos()
        
        self.logger.info("✅ [OPTIMIZADOR] Optimizador de IA inicializado correctamente")

    def cargar_datos(self) -> List[Dict[str, Any]]:
        """
        Carga datos históricos de operaciones con logs detallados
        Returns:
            Lista de operaciones históricas
        """
        try:
            timestamp_inicio = datetime.now()
            
            self.logger.info("📊 [DATOS] Cargando datos históricos de operaciones...")
            self.logger.debug(f"   • Archivo: {self.log_path}")
            
            if not os.path.exists(self.log_path):
                self.logger.warning(f"⚠️ [DATOS] Archivo de log no existe: {self.log_path}")
                return []
            
            self.datos = []
            lineas_procesadas = 0
            lineas_validas = 0
            lineas_invalidas = 0
            
            with open(self.log_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                self.logger.debug(f"   • Headers encontrados: {headers}")
                
                for linea_num, row in enumerate(reader, 1):
                    lineas_procesadas += 1
                    
                    try:
                        # Mapear campos del CSV a nuestro formato
                        operacion = {
                            'pnl_percent': float(row.get('pnl_percent', 0)),
                            'angulo_tendencia': float(row.get('angulo_tendencia', 0)),
                            'pearson': float(row.get('pearson', 0)),
                            'r2_score': float(row.get('r2_score', 0)),
                            'ancho_canal_relativo': float(row.get('ancho_canal_relativo', 0)),
                            'nivel_fuerza': int(row.get('nivel_fuerza', 1)),
                            'timestamp': row.get('timestamp', ''),
                            'symbol': row.get('symbol', ''),
                            'tipo': row.get('tipo', ''),
                            'resultado': row.get('resultado', '')
                        }
                        
                        # Validar datos críticos
                        if (operacion['pnl_percent'] != 0 and 
                            operacion['angulo_tendencia'] != 0 and 
                            operacion['pearson'] != 0 and
                            operacion['r2_score'] != 0):
                            
                            self.datos.append(operacion)
                            lineas_validas += 1
                        else:
                            lineas_invalidas += 1
                            self.logger.debug(f"   • Línea {linea_num}: Datos incompletos, omitida")
                            
                    except (ValueError, TypeError) as e:
                        lineas_invalidas += 1
                        self.logger.debug(f"   • Línea {linea_num}: Error parseando datos: {e}")
                        continue
            
            timestamp_fin = datetime.now()
            tiempo_carga = (timestamp_fin - timestamp_inicio).total_seconds()
            
            self.stats_optimizacion['datos_cargados'] = len(self.datos)
            
            self.logger.info("✅ [DATOS] Datos históricos cargados:")
            self.logger.info(f"   • Total líneas procesadas: {lineas_procesadas}")
            self.logger.info(f"   • Operaciones válidas: {lineas_validas}")
            self.logger.info(f"   • Operaciones inválidas: {lineas_invalidas}")
            self.logger.info(f"   • Período cubierto: {self._obtener_periodo_datos()}")
            self.logger.info(f"   • Tiempo de carga: {tiempo_carga:.3f}s")
            
            # Estadísticas de los datos
            if self.datos:
                self._log_estadisticas_datos()
            
            return self.datos
            
        except Exception as e:
            self.logger.error(f"❌ [DATOS] Error cargando datos: {e}")
            return []

    def _obtener_periodo_datos(self) -> str:
        """Obtiene el período cubierto por los datos"""
        try:
            if not self.datos:
                return "Sin datos"
            
            # Ordenar por timestamp para obtener el rango
            timestamps = []
            for op in self.datos:
                if op['timestamp']:
                    try:
                        ts = datetime.fromisoformat(op['timestamp'])
                        timestamps.append(ts)
                    except:
                        continue
            
            if len(timestamps) < 2:
                return "Datos insuficientes"
            
            timestamps.sort()
            fecha_inicio = timestamps[0]
            fecha_fin = timestamps[-1]
            diferencia = fecha_fin - fecha_inicio
            
            return f"{fecha_inicio.strftime('%Y-%m-%d')} a {fecha_fin.strftime('%Y-%m-%d')} ({diferencia.days} días)"
            
        except Exception as e:
            self.logger.debug(f"Error calculando período: {e}")
            return "Error calculando período"

    def _log_estadisticas_datos(self):
        """Log de estadísticas detalladas de los datos cargados"""
        try:
            pnl_values = [op['pnl_percent'] for op in self.datos]
            win_count = sum(1 for pnl in pnl_values if pnl > 0)
            loss_count = sum(1 for pnl in pnl_values if pnl < 0)
            win_rate = win_count / len(self.datos) * 100 if self.datos else 0
            
            # Distribución por símbolo
            symbols = {}
            for op in self.datos:
                symbol = op['symbol']
                symbols[symbol] = symbols.get(symbol, 0) + 1
            
            # Distribución por tipo de operación
            tipos = {}
            for op in self.datos:
                tipo = op['tipo']
                tipos[tipo] = tipos.get(tipo, 0) + 1
            
            # Distribución por resultado
            resultados = {}
            for op in self.datos:
                resultado = op['resultado']
                resultados[resultado] = resultados.get(resultado, 0) + 1
            
            self.logger.info("📈 [ESTADISTICAS] Distribución de datos:")
            self.logger.info(f"   • Total operaciones: {len(self.datos)}")
            self.logger.info(f"   • Operaciones ganadoras: {win_count} ({win_rate:.1f}%)")
            self.logger.info(f"   • Operaciones perdedoras: {loss_count}")
            self.logger.info(f"   • PnL promedio: {statistics.mean(pnl_values):.3f}%")
            self.logger.info(f"   • PnL máximo: {max(pnl_values):.3f}%")
            self.logger.info(f"   • PnL mínimo: {min(pnl_values):.3f}%")
            
            # Top símbolos
            symbols_sorted = sorted(symbols.items(), key=lambda x: x[1], reverse=True)[:5]
            self.logger.info(f"   • Top 5 símbolos: {', '.join([f'{s}({c})' for s, c in symbols_sorted])}")
            
            # Distribución por tipo
            self.logger.info(f"   • Distribución tipos: {dict(tipos)}")
            
            # Distribución por resultado
            self.logger.info(f"   • Distribución resultados: {dict(resultados)}")
            
        except Exception as e:
            self.logger.error(f"Error calculando estadísticas: {e}")

    def evaluar_configuracion(self, trend_threshold: float, min_strength: float, entry_margin: float) -> float:
        """
        Evalúa una configuración específica con logs detallados
        Args:
            trend_threshold: Umbral de tendencia en grados
            min_strength: Fuerza mínima de tendencia
            entry_margin: Margen de entrada
        Returns:
            Score de la configuración
        """
        try:
            timestamp_inicio = datetime.now()
            
            self.logger.debug(f"🧪 [EVALUACION] Evaluando configuración:")
            self.logger.debug(f"   • Trend threshold: {trend_threshold}°")
            self.logger.debug(f"   • Min strength: {min_strength}°")
            self.logger.debug(f"   • Entry margin: {entry_margin}")
            
            # Filtrar operaciones que cumplen los criterios
            operaciones_filtradas = []
            filtros_aplicados = {
                'angulo_minimo': 0,
                'fuerza_minima': 0,
                'pearson_minimo': 0,
                'r2_minimo': 0,
                'nivel_fuerza_minimo': 0
            }
            
            for operacion in self.datos:
                angulo = abs(operacion['angulo_tendencia'])
                fuerza = abs(operacion['angulo_tendencia'])  # Usar ángulo como medida de fuerza
                pearson = abs(operacion['pearson'])
                r2 = operacion['r2_score']
                nivel_fuerza = operacion.get('nivel_fuerza', 1)
                
                # Aplicar filtros
                if angulo >= trend_threshold:
                    filtros_aplicados['angulo_minimo'] += 1
                    
                    if fuerza >= min_strength:
                        filtros_aplicados['fuerza_minima'] += 1
                        
                        if pearson >= Constants.MIN_PEARSON:
                            filtros_aplicados['pearson_minimo'] += 1
                            
                            if r2 >= Constants.MIN_R2_SCORE:
                                filtros_aplicados['r2_minimo'] += 1
                                
                                if nivel_fuerza >= 2:
                                    filtros_aplicados['nivel_fuerza_minimo'] += 1
                                    operaciones_filtradas.append(operacion)
            
            # Verificar si hay suficientes muestras
            min_requeridas = max(8, int(0.15 * len(self.datos)))
            
            if len(operaciones_filtradas) < min_requeridas:
                self.logger.debug(f"❌ [EVALUACION] Muestras insuficientes: {len(operaciones_filtradas)} < {min_requeridas}")
                self.stats_optimizacion['filtros_aplicados'] += 1
                return -1000  # Score muy bajo por muestras insuficientes
            
            # Calcular métricas
            pnl_values = [op['pnl_percent'] for op in operaciones_filtradas]
            win_count = sum(1 for pnl in pnl_values if pnl > 0)
            win_rate = win_count / len(pnl_values)
            pnl_mean = statistics.mean(pnl_values)
            pnl_std = statistics.stdev(pnl_values) if len(pnl_values) > 1 else 0
            n = len(pnl_values)
            
            # Calcular score principal
            score_base = (pnl_mean - 0.5 * pnl_std) * win_rate * statistics.sqrt(n)
            
            # Bonus por operaciones de alta calidad
            operaciones_calidad = [op for op in operaciones_filtradas 
                                 if op['r2_score'] >= 0.6 and op.get('nivel_fuerza', 1) >= 3]
            
            if operaciones_calidad:
                bonus_calidad = 1.2
                self.logger.debug(f"   • Operaciones alta calidad: {len(operaciones_calidad)} ({bonus_calidad}x)")
            else:
                bonus_calidad = 1.0
            
            score_final = score_base * bonus_calidad
            
            timestamp_fin = datetime.now()
            tiempo_evaluacion = (timestamp_fin - timestamp_inicio).total_seconds()
            
            self.logger.debug(f"✅ [EVALUACION] Resultados:")
            self.logger.debug(f"   • Operaciones que pasan filtros: {len(operaciones_filtradas)}/{len(self.datos)}")
            self.logger.debug(f"   • Win rate: {win_rate:.3f} ({win_count}/{n})")
            self.logger.debug(f"   • PnL promedio: {pnl_mean:.3f}%")
            self.logger.debug(f"   • PnL desviación: {pnl_std:.3f}%")
            self.logger.debug(f"   • Score base: {score_base:.3f}")
            self.logger.debug(f"   • Score final: {score_final:.3f}")
            self.logger.debug(f"   • Tiempo evaluación: {tiempo_evaluacion:.3f}s")
            
            self.stats_optimizacion['parametros_evaluados'] += 1
            
            return score_final
            
        except Exception as e:
            self.logger.error(f"❌ [EVALUACION] Error evaluando configuración: {e}")
            return -1000

    def buscar_mejores_parametros(self) -> Optional[Dict[str, Any]]:
        """
        Busca los mejores parámetros con logs extensivos
        Returns:
            Diccionario con mejores parámetros o None
        """
        try:
            timestamp_inicio = datetime.now()
            
            self.logger.info("🔍 [BUSQUEDA] Iniciando búsqueda de mejores parámetros...")
            
            # Validar datos suficientes
            if not self.validar_datos_suficientes():
                self.logger.warning("⚠️ [BUSQUEDA] Datos insuficientes para optimización")
                return None
            
            # Definir rangos de parámetros
            trend_values = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40]  # grados
            strength_values = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30]  # grados
            margin_values = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.008, 0.01]
            
            total_combinaciones = len(trend_values) * len(strength_values) * len(margin_values)
            
            self.logger.info(f"📊 [BUSQUEDA] Parámetros de búsqueda:")
            self.logger.info(f"   • Valores trend: {trend_values}")
            self.logger.info(f"   • Valores strength: {strength_values}")
            self.logger.info(f"   • Valores margin: {margin_values}")
            self.logger.info(f"   • Total combinaciones: {total_combinaciones}")
            
            mejor_score = -999999
            mejor_config = None
            configuraciones_evaluadas = 0
            top_configs = []
            
            # Búsqueda exhaustiva
            for i, trend_threshold in enumerate(trend_values):
                for j, min_strength in enumerate(strength_values):
                    for k, entry_margin in enumerate(margin_values):
                        configuraciones_evaluadas += 1
                        
                        # Log de progreso cada 50 configuraciones
                        if configuraciones_evaluadas % 50 == 0:
                            progreso = (configuraciones_evaluadas / total_combinaciones) * 100
                            self.logger.info(f"⏳ [BUSQUEDA] Progreso: {configuraciones_evaluadas}/{total_combinaciones} ({progreso:.1f}%)")
                        
                        score = self.evaluar_configuracion(trend_threshold, min_strength, entry_margin)
                        
                        # Guardar top configuraciones
                        config_info = {
                            'trend_threshold_degrees': trend_threshold,
                            'min_trend_strength_degrees': min_strength,
                            'entry_margin': entry_margin,
                            'score': score
                        }
                        
                        if len(top_configs) < 10:
                            top_configs.append(config_info)
                            top_configs.sort(key=lambda x: x['score'], reverse=True)
                        elif score > top_configs[-1]['score']:
                            top_configs[-1] = config_info
                            top_configs.sort(key=lambda x: x['score'], reverse=True)
                        
                        # Actualizar mejor configuración
                        if score > mejor_score:
                            mejor_score = score
                            mejor_config = config_info.copy()
                            
                            self.logger.debug(f"🏆 [BUSQUEDA] Nueva mejor configuración encontrada:")
                            self.logger.debug(f"   • Score: {score:.3f}")
                            self.logger.debug(f"   • Trend: {trend_threshold}°")
                            self.logger.debug(f"   • Strength: {min_strength}°")
                            self.logger.debug(f"   • Margin: {entry_margin}")
            
            timestamp_fin = datetime.now()
            tiempo_busqueda = (timestamp_fin - timestamp_inicio).total_seconds()
            
            self.stats_optimizacion['optimizaciones_realizadas'] += 1
            self.stats_optimizacion['tiempo_total_optimizacion'] += tiempo_busqueda
            self.stats_optimizacion['mejores_configuraciones_encontradas'] += 1
            
            if mejor_config:
                # Guardar mejores parámetros
                try:
                    mejores_parametros_file = config.mejores_parametros_file
                    with open(mejores_parametros_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'mejor_configuracion': mejor_config,
                            'top_10_configuraciones': top_configs,
                            'timestamp_optimizacion': datetime.now().isoformat(),
                            'estadisticas': self.stats_optimizacion.copy()
                        }, f, indent=2, ensure_ascii=False)
                    
                    self.logger.info(f"💾 [BUSQUEDA] Mejores parámetros guardados en: {mejores_parametros_file}")
                except Exception as e:
                    self.logger.error(f"❌ [BUSQUEDA] Error guardando parámetros: {e}")
                
                # Log de resultados finales
                self.logger.info("✅ [BUSQUEDA] Optimización completada:")
                self.logger.info(f"   • Configuraciones evaluadas: {configuraciones_evaluadas}")
                self.logger.info(f"   • Mejor score: {mejor_score:.3f}")
                self.logger.info(f"   • Tiempo total: {tiempo_busqueda:.3f}s")
                self.logger.info(f"   • Tiempo promedio por config: {tiempo_busqueda/configuraciones_evaluadas:.3f}s")
                
                self.logger.info("🏆 [RESULTADO] Mejor configuración encontrada:")
                self.logger.info(f"   • Trend threshold: {mejor_config['trend_threshold_degrees']}°")
                self.logger.info(f"   • Min trend strength: {mejor_config['min_trend_strength_degrees']}°")
                self.logger.info(f"   • Entry margin: {mejor_config['entry_margin']}")
                self.logger.info(f"   • Score: {mejor_config['score']:.3f}")
                
                # Log del top 5
                self.logger.info("📊 [TOP 5] Mejores configuraciones:")
                for i, config in enumerate(top_configs[:5], 1):
                    self.logger.info(f"   {i}. Score: {config['score']:.3f} | "
                                   f"Trend: {config['trend_threshold_degrees']}° | "
                                   f"Strength: {config['min_trend_strength_degrees']}° | "
                                   f"Margin: {config['entry_margin']}")
                
                return mejor_config
            else:
                self.logger.warning("⚠️ [BUSQUEDA] No se encontró configuración válida")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ [BUSQUEDA] Error en búsqueda de mejores parámetros: {e}")
            return None

    def validar_datos_suficientes(self) -> bool:
        """
        Valida que hay suficientes datos para optimización con logs detallados
        Returns:
            True si hay suficientes datos
        """
        try:
            self.logger.debug("🔍 [VALIDACION] Validando suficiencia de datos...")
            
            if not self.datos:
                self.logger.debug("   ❌ No hay datos cargados")
                return False
            
            # Contar tipos de resultados
            tp_count = sum(1 for op in self.datos if op.get('resultado') == 'TP')
            sl_count = sum(1 for op in self.datos if op.get('resultado') == 'SL')
            total_ops = len(self.datos)
            
            self.logger.debug(f"📊 [VALIDACION] Estadísticas de datos:")
            self.logger.debug(f"   • Total operaciones: {total_ops}")
            self.logger.debug(f"   • Operaciones TP: {tp_count}")
            self.logger.debug(f"   • Operaciones SL: {sl_count}")
            self.logger.debug(f"   • Win rate actual: {(tp_count/total_ops*100):.1f}%" if total_ops > 0 else "   • Win rate actual: N/A")
            
            # Validaciones
            validaciones = {
                'muestras_minimas': total_ops >= self.min_samples,
                'distribucion_tp_sl': tp_count >= 3 and sl_count >= 3,
                'variedad_simbolos': len(set(op.get('symbol', '') for op in self.datos)) >= 2
            }
            
            self.logger.debug(f"🔍 [VALIDACION] Resultados de validación:")
            for validacion, resultado in validaciones.items():
                status = "✅" if resultado else "❌"
                self.logger.debug(f"   • {validacion}: {status}")
            
            todas_pasan = all(validaciones.values())
            
            if todas_pasan:
                self.logger.debug("✅ [VALIDACION] Datos suficientes para optimización")
            else:
                self.logger.debug("❌ [VALIDACION] Datos insuficientes para optimización")
            
            return todas_pasan
            
        except Exception as e:
            self.logger.error(f"❌ [VALIDACION] Error validando datos: {e}")
            return False

    def generar_reporte_optimizacion(self, parametros: Dict[str, Any]) -> str:
        """
        Genera reporte de optimización con logs detallados
        Args:
            parametros: Parámetros optimizados
        Returns:
            String con reporte formateado
        """
        try:
            self.logger.debug("📋 [REPORTE] Generando reporte de optimización...")
            
            if not self.datos:
                return "❌ No hay datos para generar reporte"
            
            # Obtener estadísticas de los datos
            pnl_values = [op['pnl_percent'] for op in self.datos]
            win_count = sum(1 for pnl in pnl_values if pnl > 0)
            win_rate = win_count / len(self.datos) * 100
            
            reporte = f"""
📊 REPORTE DE OPTIMIZACIÓN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🧠 PARÁMETROS OPTIMIZADOS:
• Trend Threshold: {parametros.get('trend_threshold_degrees', 'N/A')}°
• Min Trend Strength: {parametros.get('min_trend_strength_degrees', 'N/A')}°
• Entry Margin: {parametros.get('entry_margin', 'N/A')}

📈 ESTADÍSTICAS DE DATOS:
• Total operaciones analizadas: {len(self.datos)}
• Operaciones ganadoras: {win_count}
• Win rate: {win_rate:.1f}%
• PnL promedio: {statistics.mean(pnl_values):.3f}%
• PnL máximo: {max(pnl_values):.3f}%
• PnL mínimo: {min(pnl_values):.3f}%

⏱️ ESTADÍSTICAS DE OPTIMIZACIÓN:
• Optimizaciones realizadas: {self.stats_optimizacion['optimizaciones_realizadas']}
• Parámetros evaluados: {self.stats_optimizacion['parametros_evaluados']}
• Tiempo total optimizando: {self.stats_optimizacion['tiempo_total_optimizacion']:.3f}s
• Datos cargados: {self.stats_optimizacion['datos_cargados']}

💡 RECOMENDACIONES:
• Usar estos parámetros para mejorar el rendimiento del bot
• Monitorear resultados y reoptimizar periódicamente
• Considerar reoptimizar con más datos históricos
"""
            
            self.logger.debug("✅ [REPORTE] Reporte generado correctamente")
            return reporte
            
        except Exception as e:
            self.logger.error(f"❌ [REPORTE] Error generando reporte: {e}")
            return f"❌ Error generando reporte: {e}"

    def obtener_estadisticas_optimizacion(self) -> Dict[str, Any]:
        """Obtiene estadísticas de optimización"""
        stats = self.stats_optimizacion.copy()
        stats.update({
            'total_datos': len(self.datos),
            'datos_validos': len(self.datos),
            'archivo_datos': self.log_path
        })
        return stats

# Instancia global del optimizador
optimizador_ia = OptimizadorIA()
print("🧠 Optimizador de IA con logs detallados cargado correctamente")
