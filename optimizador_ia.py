"""
Optimizador IA para parámetros de trading.
Analiza datos históricos para encontrar la mejor configuración.
"""

import csv
import json
import itertools
import statistics
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..config.settings import config

class OptimizadorIA:
    """Optimizador de inteligencia artificial para parámetros de trading"""
    
    def __init__(self, log_path: str = None, min_samples: int = Constants.MIN_MUESTRAS_OPTIMIZACION):
        """
        Inicializa el optimizador IA
        
        Args:
            log_path: Ruta del archivo de log de operaciones
            min_samples: Mínimo de muestras para optimizar
        """
        self.log_path = log_path or config.log_path
        self.min_samples = min_samples
        self.logger = logging.getLogger(__name__)
        self.datos = self.cargar_datos()
        
    def cargar_datos(self) -> List[Dict[str, Any]]:
        """
        Carga datos históricos desde el archivo de log
        
        Returns:
            Lista de diccionarios con datos de operaciones
        """
        try:
            datos = []
            
            if not hasattr(self, 'log_path') or not self.log_path:
                self.logger.warning("Ruta de log no especificada")
                return datos
                
            with open(self.log_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Extraer y validar datos
                        pnl = float(row.get('pnl_percent', 0))
                        angulo = float(row.get('angulo_tendencia', 0))
                        pearson = float(row.get('pearson', 0))
                        r2 = float(row.get('r2_score', 0))
                        ancho_relativo = float(row.get('ancho_canal_relativo', 0))
                        nivel_fuerza = int(row.get('nivel_fuerza', 1))
                        
                        datos.append({
                            'pnl': pnl,
                            'angulo': angulo,
                            'pearson': pearson,
                            'r2': r2,
                            'ancho_relativo': ancho_relativo,
                            'nivel_fuerza': nivel_fuerza,
                            'timestamp': row.get('timestamp', ''),
                            'symbol': row.get('symbol', ''),
                            'tipo': row.get('tipo', ''),
                            'resultado': row.get('resultado', '')
                        })
                        
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Ignorando fila inválida: {e}")
                        continue
                        
        except FileNotFoundError:
            self.logger.warning(f"Archivo de log no encontrado: {self.log_path}")
        except Exception as e:
            self.logger.error(f"Error cargando datos: {e}")
            
        self.logger.info(f"✅ Datos cargados: {len(datos)} operaciones")
        return datos
    
    def evaluar_configuracion(self, trend_threshold: float, min_strength: float, entry_margin: float) -> float:
        """
        Evalúa una configuración específica
        
        Args:
            trend_threshold: Umbral de ángulo de tendencia
            min_strength: Fuerza mínima de tendencia
            entry_margin: Margen de entrada
            
        Returns:
            Score de la configuración
        """
        try:
            if not self.datos:
                self.logger.warning("No hay datos para evaluar")
                return -99999
                
            # Filtrar operaciones según criterios
            filtradas = [
                op for op in self.datos
                if abs(op['angulo']) >= trend_threshold
                and abs(op['angulo']) >= min_strength
                and abs(op['pearson']) >= 0.4
                and op.get('nivel_fuerza', 1) >= 2
                and op.get('r2', 0) >= 0.4
            ]
            
            n = len(filtradas)
            
            # Verificar cantidad mínima de muestras
            min_requerido = max(8, int(0.15 * len(self.datos)))
            if n < min_requerido:
                score = -10000 - n
                self.logger.debug(f"Pocos datos: {n} < {min_requerido}, score: {score}")
                return score
                
            # Calcular métricas de rendimiento
            pnls = [op['pnl'] for op in filtradas]
            pnl_mean = statistics.mean(pnls) if filtradas else 0
            pnl_std = statistics.stdev(pnls) if len(pnls) > 1 else 0
            
            # Calcular winrate
            wins = sum(1 for op in filtradas if op['pnl'] > 0)
            winrate = wins / n if n > 0 else 0
            
            # Score principal: media ponderada por winrate y consistencia
            score = (pnl_mean - 0.5 * pnl_std) * winrate * statistics.sqrt(n)
            
            # Bonus por operaciones de alta calidad
            ops_calidad = [
                op for op in filtradas 
                if op.get('r2', 0) >= 0.6 and op.get('nivel_fuerza', 1) >= 3
            ]
            
            if ops_calidad:
                score *= 1.2
                self.logger.debug(f"Bonus por calidad: {len(ops_calidad)} operaciones de alta calidad")
                
            self.logger.debug(f"Evaluación: {trend_threshold}, {min_strength}, {entry_margin} -> Score: {score:.4f}")
            return score
            
        except Exception as e:
            self.logger.error(f"Error evaluando configuración: {e}")
            return -99999
    
    def buscar_mejores_parametros(self) -> Optional[Dict[str, Any]]:
        """
        Busca los mejores parámetros mediante búsqueda exhaustiva
        
        Returns:
            Diccionario con mejores parámetros o None
        """
        try:
            if not self.datos or len(self.datos) < self.min_samples:
                self.logger.warning(f"Insuficientes datos para optimizar (requeridos: {self.min_samples}, disponibles: {len(self.datos)})")
                return None
                
            mejor_score = -1e9
            mejores_param = None
            
            # Rangos de parámetros a probar
            trend_values = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40]
            strength_values = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30]
            margin_values = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.008, 0.01]
            
            # Generar todas las combinaciones
            combos = list(itertools.product(trend_values, strength_values, margin_values))
            total = len(combos)
            
            self.logger.info(f"🔎 Optimizador: probando {total} combinaciones...")
            
            # Evaluar cada combinación
            for idx, (trend_thresh, min_strength, entry_marg) in enumerate(combos, start=1):
                try:
                    score = self.evaluar_configuracion(trend_thresh, min_strength, entry_marg)
                    
                    # Mostrar progreso cada 100 combinaciones
                    if idx % 100 == 0 or idx == total:
                        self.logger.info(f"   Progreso: {idx}/{total} combinaciones (mejor score: {mejor_score:.4f})")
                    
                    # Actualizar mejor configuración
                    if score > mejor_score:
                        mejor_score = score
                        mejores_param = {
                            'trend_threshold_degrees': trend_thresh,
                            'min_trend_strength_degrees': min_strength,
                            'entry_margin': entry_marg,
                            'score': score,
                            'evaluated_samples': len(self.datos),
                            'total_combinations': total,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                except Exception as e:
                    self.logger.warning(f"Error evaluando combinación {idx}: {e}")
                    continue
            
            # Guardar resultados si se encontró configuración
            if mejores_param:
                self.logger.info("✅ Optimizador: mejores parámetros encontrados:")
                self.logger.info(f"   📊 Score: {mejor_score:.4f}")
                self.logger.info(f"   📈 Threshold: {mejores_param['trend_threshold_degrees']}°")
                self.logger.info(f"   💪 Fuerza: {mejores_param['min_trend_strength_degrees']}°")
                self.logger.info(f"   🎯 Margen: {mejores_param['entry_margin']:.4f}")
                
                # Guardar en archivo
                try:
                    with open(config.mejores_parametros_file, "w", encoding='utf-8') as f:
                        json.dump(mejores_param, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"✅ Parámetros guardados en: {config.mejores_parametros_file}")
                except Exception as e:
                    self.logger.warning(f"Error guardando parámetros: {e}")
            else:
                self.logger.warning("⚠️ No se encontró una configuración mejor")
                
            return mejores_param
            
        except Exception as e:
            self.logger.error(f"Error en búsqueda de parámetros: {e}")
            return None
    
    def generar_reporte_optimizacion(self, parametros: Dict[str, Any]) -> str:
        """
        Genera un reporte detallado de la optimización
        
        Args:
            parametros: Parámetros encontrados
            
        Returns:
            String con reporte formateado
        """
        try:
            reporte = f"""
📊 REPORTE DE OPTIMIZACIÓN IA
{'='*50}
⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📈 Muestras analizadas: {parametros.get('evaluated_samples', 0)}
🔄 Combinaciones probadas: {parametros.get('total_combinations', 0)}
🏆 Score final: {parametros.get('score', 0):.4f}

📋 PARÁMETROS ÓPTIMOS:
• Threshold de tendencia: {parametros.get('trend_threshold_degrees', 0)}°
• Fuerza mínima: {parametros.get('min_trend_strength_degrees', 0)}°
• Margen de entrada: {parametros.get('entry_margin', 0):.4f}

💡 RECOMENDACIONES:
- Aplicar estos parámetros automáticamente
- Revisar semanalmente
- Monitorear rendimiento
{'='*50}
            """
            return reporte
            
        except Exception as e:
            self.logger.error(f"Error generando reporte: {e}")
            return "Error generando reporte de optimización"
    
    def validar_datos_suficientes(self) -> bool:
        """
        Valida si hay suficientes datos para optimizar
        
        Returns:
            True si hay suficientes datos
        """
        try:
            if not self.datos:
                self.logger.warning("No hay datos disponibles")
                return False
                
            if len(self.datos) < self.min_samples:
                self.logger.warning(f"Datos insuficientes: {len(self.datos)} < {self.min_samples}")
                return False
                
            # Verificar distribución de resultados
            resultados = [op.get('resultado', '') for op in self.datos]
            wins = resultados.count('TP')
            losses = resultados.count('SL')
            
            if wins + losses < self.min_samples:
                self.logger.warning(f"Resultados insuficientes: {wins + losses} operaciones")
                return False
                
            self.logger.info(f"✅ Validación exitosa: {len(self.datos)} operaciones ({wins} wins, {losses} losses)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validando datos: {e}")
            return False

# Instancia global del optimizador
optimizador_ia = OptimizadorIA()

print("🤖 Optimizador IA cargado correctamente")
