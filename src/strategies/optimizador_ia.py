"""
Optimizador IA para el bot de trading.
Utiliza machine learning para optimizar parámetros basado en datos históricos.
"""
import os
import json
import csv
import math
import itertools
import statistics
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from ..config.settings import config

class OptimizadorIA:
    """Optimizador IA que utiliza datos históricos para mejorar parámetros"""
    
    def __init__(self, log_path: str = "operaciones_log_v23.csv", min_samples: int = 15):
        """
        Inicializa el optimizador IA
        
        Args:
            log_path: Ruta del archivo de log de operaciones
            min_samples: Mínimo de muestras para optimización
        """
        self.log_path = log_path
        self.min_samples = min_samples
        self.logger = logging.getLogger(__name__)
        
        # Cargar datos históricos
        self.datos = self.cargar_datos()
        
        self.logger.info(f"🤖 Optimizador IA inicializado:")
        self.logger.info(f" • Archivo de datos: {log_path}")
        self.logger.info(f" • Mínimo de muestras: {min_samples}")
        self.logger.info(f" • Datos cargados: {len(self.datos)} operaciones")
    
    def cargar_datos(self) -> List[Dict[str, Any]]:
        """
        Carga datos históricos de operaciones desde CSV
        
        Returns:
            Lista de diccionarios con datos de operaciones
        """
        datos = []
        
        try:
            if not os.path.exists(self.log_path):
                self.logger.warning(f"⚠️ Archivo de log no encontrado: {self.log_path}")
                return datos
            
            with open(self.log_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Extraer y validar datos
                        operacion = {
                            'pnl': float(row.get('pnl_percent', 0)),
                            'angulo_tendencia': float(row.get('angulo_tendencia', 0)),
                            'pearson': float(row.get('pearson', 0)),
                            'r2_score': float(row.get('r2_score', 0)),
                            'ancho_canal_relativo': float(row.get('ancho_canal_relativo', 0)),
                            'ancho_canal_porcentual': float(row.get('ancho_canal_porcentual', 0)),
                            'nivel_fuerza': int(row.get('nivel_fuerza', 1)),
                            'timeframe': row.get('timeframe_utilizado', 'N/A'),
                            'velas_utilizadas': int(row.get('velas_utilizadas', 0)),
                            'stoch_k': float(row.get('stoch_k', 50)),
                            'stoch_d': float(row.get('stoch_d', 50)),
                            'breakout_usado': row.get('breakout_usado', 'False') == 'True',
                            'resultado': row.get('resultado', 'N/A'),
                            'tipo': row.get('tipo', 'N/A'),
                            'timestamp': row.get('timestamp', '')
                        }
                        
                        # Validar datos esenciales
                        if not math.isnan(operacion['pnl']) and operacion['timestamp']:
                            datos.append(operacion)
                            
                    except (ValueError, TypeError) as e:
                        self.logger.debug(f"⚠️ Error procesando fila: {e}")
                        continue
            
            self.logger.info(f"📊 Datos cargados exitosamente: {len(datos)} operaciones válidas")
            
        except Exception as e:
            self.logger.error(f"❌ Error cargando datos históricos: {e}")
            
        return datos
    
    def evaluar_configuracion(self, trend_threshold: float, min_strength: float, entry_margin: float) -> float:
        """
        Evalúa una configuración específica basada en datos históricos
        
        Args:
            trend_threshold: Umbral de tendencia
            min_strength: Fuerza mínima de tendencia
            entry_margin: Margen de entrada
            
        Returns:
            Score de la configuración (mayor es mejor)
        """
        if not self.datos:
            return -99999
        
        # Filtrar operaciones que cumplen criterios
        filtradas = []
        for op in self.datos:
            if (abs(op['angulo_tendencia']) >= trend_threshold and
                abs(op['angulo_tendencia']) >= min_strength and
                abs(op['pearson']) >= 0.4 and
                op.get('nivel_fuerza', 1) >= 2 and
                op.get('r2_score', 0) >= 0.4):
                filtradas.append(op)
        
        # Verificar cantidad mínima de datos
        min_requerido = max(8, int(0.15 * len(self.datos)))
        if len(filtradas) < min_requerido:
            return -10000 - len(filtradas)
        
        # Calcular métricas de rendimiento
        pnls = [op['pnl'] for op in filtradas]
        pnl_mean = statistics.mean(pnls) if pnls else 0
        pnl_std = statistics.stdev(pnls) if len(pnls) > 1 else 0
        
        # Win rate
        wins = sum(1 for op in filtradas if op['pnl'] > 0)
        winrate = wins / len(filtradas) if filtradas else 0
        
        # Score compuesto: (PnL promedio - penalización por volatilidad) * winrate * sqrt(n)
        score = (pnl_mean - 0.5 * pnl_std) * winrate * math.sqrt(len(filtradas))
        
        # Bonus por operaciones de alta calidad
        ops_calidad = [op for op in filtradas 
                      if op.get('r2_score', 0) >= 0.6 and op.get('nivel_fuerza', 1) >= 3]
        if ops_calidad:
            score *= 1.2
        
        return score
    
    def buscar_mejores_parametros(self) -> Optional[Dict[str, Any]]:
        """
        Busca los mejores parámetros usando búsqueda grid
        
        Returns:
            Diccionario con los mejores parámetros encontrados
        """
        if not self.datos or len(self.datos) < self.min_samples:
            self.logger.warning(f"⚠️ No hay suficientes datos para optimizar (requeridos: {self.min_samples}, disponibles: {len(self.datos)})")
            return None
        
        self.logger.info("🔍 Iniciando búsqueda de mejores parámetros...")
        
        # Definir rangos de búsqueda
        trend_values = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40]
        strength_values = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30]
        margin_values = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.008, 0.01]
        
        # Generar todas las combinaciones
        combinaciones = list(itertools.product(trend_values, strength_values, margin_values))
        total_combinaciones = len(combinaciones)
        
        self.logger.info(f"🧪 Probando {total_combinaciones} combinaciones...")
        
        mejor_score = -1e9
        mejores_parametros = None
        resultados = []
        
        for i, (trend, strength, margin) in enumerate(combinaciones, 1):
            try:
                score = self.evaluar_configuracion(trend, strength, margin)
                resultados.append({
                    'trend_threshold': trend,
                    'min_strength': strength,
                    'entry_margin': margin,
                    'score': score
                })
                
                if score > mejor_score:
                    mejor_score = score
                    mejores_parametros = {
                        'trend_threshold_degrees': trend,
                        'min_trend_strength_degrees': strength,
                        'entry_margin': margin,
                        'score': score,
                        'evaluated_samples': len(self.datos),
                        'total_combinations': total_combinaciones,
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Log de progreso cada 100 combinaciones
                if i % 100 == 0 or i == total_combinaciones:
                    self.logger.info(f"   📊 Progreso: {i}/{total_combinaciones} - Mejor score: {mejor_score:.4f}")
                    
            except Exception as e:
                self.logger.debug(f"⚠️ Error evaluando combinación {trend}-{strength}-{margin}: {e}")
                continue
        
        if mejores_parametros:
            self.logger.info("✅ Mejores parámetros encontrados:")
            self.logger.info(f"   • Trend threshold: {mejores_parametros['trend_threshold_degrees']}°")
            self.logger.info(f"   • Min strength: {mejores_parametros['min_trend_strength_degrees']}°")
            self.logger.info(f"   • Entry margin: {mejores_parametros['entry_margin']}")
            self.logger.info(f"   • Score: {mejores_parametros['score']:.4f}")
            
            # Guardar resultados
            self._guardar_resultados_optimizacion(resultados, mejores_parametros)
            
        else:
            self.logger.warning("⚠️ No se encontraron parámetros válidos")
        
        return mejores_parametros
    
    def _guardar_resultados_optimizacion(self, resultados: List[Dict], mejores: Dict[str, Any]):
        """
        Guarda los resultados de la optimización
        
        Args:
            resultados: Lista de todos los resultados evaluados
            mejores: Diccionario con los mejores parámetros
        """
        try:
            # Guardar mejores parámetros
            with open('mejores_parametros.json', 'w', encoding='utf-8') as f:
                json.dump(mejores, f, indent=2, ensure_ascii=False)
            
            # Guardar todos los resultados para análisis
            with open('resultados_optimizacion.csv', 'w', newline='', encoding='utf-8') as f:
                if resultados:
                    writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
                    writer.writeheader()
                    writer.writerows(resultados)
            
            self.logger.info("💾 Resultados de optimización guardados:")
            self.logger.info("   • mejores_parametros.json")
            self.logger.info("   • resultados_optimizacion.csv")
            
        except Exception as e:
            self.logger.error(f"❌ Error guardando resultados: {e}")
    
    def obtener_estadisticas_historicas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de los datos históricos
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.datos:
            return {
                'total_operaciones': 0,
                'mensaje': 'No hay datos históricos disponibles'
            }
        
        # Calcular estadísticas
        pnls = [op['pnl'] for op in self.datos]
        wins = sum(1 for op in self.datos if op['pnl'] > 0)
        losses = sum(1 for op in self.datos if op['pnl'] < 0)
        
        stats = {
            'total_operaciones': len(self.datos),
            'operaciones_ganadoras': wins,
            'operaciones_perdedoras': losses,
            'win_rate': wins / len(self.datos) if self.datos else 0,
            'pnl_promedio': statistics.mean(pnls) if pnls else 0,
            'pnl_std': statistics.stdev(pnls) if len(pnls) > 1 else 0,
            'mejor_pnl': max(pnls) if pnls else 0,
            'peor_pnl': min(pnls) if pnls else 0,
            'breakouts_utilizados': sum(1 for op in self.datos if op.get('breakout_usado', False)),
            'timeframes_analizados': list(set(op.get('timeframe', 'N/A') for op in self.datos)),
            'rango_fechas': {
                'inicio': min(op.get('timestamp', '') for op in self.datos if op.get('timestamp')),
                'fin': max(op.get('timestamp', '') for op in self.datos if op.get('timestamp'))
            }
        }
        
        return stats

# Instancia global del optimizador
optimizador_ia = OptimizadorIA()
