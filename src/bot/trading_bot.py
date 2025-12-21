"""
Bot principal de trading con logs detallados.
Orquesta toda la lógica de trading, manejo de estado y coordinación.
MEJORADO CON LOGS EXTENSIVOS PARA MAYOR VISIBILIDAD
"""
import os
import json
import time
import random
import logging
import threading
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from io import BytesIO
from ..config.settings import config, Constants
from ..api.clients import telegram_client, binance_client
from ..strategies.breakout_reentry import estrategia, CanalInfo, DatosMercado
from ..strategies.optimizador_ia import optimizador_ia
import csv # Explicitly added from context, as it's used but not in initial imports
import matplotlib.pyplot as plt # Explicitly added from context
import pandas as pd # Explicitly added from context

class TradingBot:
    """Bot principal de trading Breakout + Reentry con logs extensivos"""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Inicializa el bot de trading con logs detallados
        Args:
            config_dict: Configuración del bot (diccionario o instancia)
        """
        timestamp_inicio = datetime.now()
        
        # Manejar tanto diccionario como instancia
        if isinstance(config_dict, dict):
            self.config = config_dict
            # Usar config global para obtener propiedades que no están en el diccionario
            self._config_instance = config
        else:
            self.config = config_dict
            self._config_instance = config_dict

        self.logger = logging.getLogger(__name__)

        # Estado del bot
        self.log_path = self._get_config_value('log_path', self._config_instance.log_path)
        self.auto_optimize = self._get_config_value('auto_optimize', self._config_instance.auto_optimize)
        self.ultima_optimizacion = datetime.now()
        self.operaciones_desde_optimizacion = 0
        self.total_operaciones = 0

        # Historiales
        self.breakout_history = {}
        self.breakouts_detectados = {}
        self.esperando_reentry = {}

        # Estado de operaciones
        self.operaciones_activas = {}
        self.senales_enviadas = set()

        self.config_optima_por_simbolo = {}
        self.ultima_busqueda_config = {}

        # Estadísticas de logs
        self.stats_bot = {
            'ciclos_escaneo': 0,
            'simbolos_analizados': 0,
            'breakouts_registrados': 0,
            'reentries_confirmados': 0,
            'senales_generadas': 0,
            'operaciones_activas': 0,
            'errores_por_simbolo': {},
            'tiempo_total_procesamiento': 0.0
        }

        self.logger.info("🤖 [INICIO] Inicializando Bot de Trading Breakout + Reentry")
        self.logger.info(f"   • Auto-optimización: {'✅ Habilitada' if self.auto_optimize else '❌ Deshabilitada'}")
        self.logger.info(f"   • Archivo de logs: {self.log_path}")
        
        # Cargar estado previo
        self.cargar_estado()

        # Optimización automática
        if self.auto_optimize:
            self._ejecutar_optimizacion_inicial()

        # Inicializar log
        self.inicializar_log()
        
        timestamp_fin = datetime.now()
        tiempo_inicializacion = (timestamp_fin - timestamp_inicio).total_seconds()
        
        self.logger.info("✅ [INICIO] Bot inicializado correctamente")
        self.logger.info(f"   • Tiempo de inicialización: {tiempo_inicializacion:.3f}s")
        self.logger.info(f"   • Total operaciones históricas: {self.total_operaciones}")

    def _get_config_value(self, key: str, default_value: Any = None):
        """Obtiene valor de configuración de manera segura"""
        return self.config.get(key, default_value)

    def _ejecutar_optimizacion_inicial(self):
        """Ejecuta optimización inicial si está habilitada con logs detallados"""
        try:
            self.logger.info("🔄 [OPTIMIZACION] Ejecutando optimización inicial...")
            timestamp_inicio = datetime.now()
            
            parametros_optimizados = optimizador_ia.buscar_mejores_parametros()
            timestamp_optimizacion = datetime.now()
            tiempo_optimizacion = (timestamp_optimizacion - timestamp_inicio).total_seconds()
            
            if parametros_optimizados:
                # Log de parámetros antes del cambio
                self.logger.debug(f"📊 [OPTIMIZACION] Parámetros actuales:")
                self.logger.debug(f"   • trend_threshold_degrees: {self.config.get('trend_threshold_degrees', 16)}")
                self.logger.debug(f"   • min_trend_strength_degrees: {self.config.get('min_trend_strength_degrees', 16)}")
                self.logger.debug(f"   • entry_margin: {self.config.get('entry_margin', 0.001)}")
                
                # Aplicar parámetros optimizados
                self.config['trend_threshold_degrees'] = parametros_optimizados.get('trend_threshold_degrees',
                                                                                    self.config.get('trend_threshold_degrees', 16))
                self.config['min_trend_strength_degrees'] = parametros_optimizados.get('min_trend_strength_degrees',
                                                                                       self.config.get('min_trend_strength_degrees', 16))
                self.config['entry_margin'] = parametros_optimizados.get('entry_margin',
                                                                         self.config.get('entry_margin', 0.001))
                
                # Log de parámetros después del cambio
                self.logger.info("✅ [OPTIMIZACION] Parámetros optimizados aplicados:")
                self.logger.info(f"   • trend_threshold_degrees: {self.config['trend_threshold_degrees']}")
                self.logger.info(f"   • min_trend_strength_degrees: {self.config['min_trend_strength_degrees']}")
                self.logger.info(f"   • entry_margin: {self.config['entry_margin']}")
                self.logger.info(f"   • Tiempo de optimización: {tiempo_optimizacion:.3f}s")
            else:
                self.logger.warning("⚠️ [OPTIMIZACION] No se pudieron obtener parámetros optimizados")
                self.logger.warning(f"   • Tiempo invertido: {tiempo_optimizacion:.3f}s")
        except Exception as e:
            self.logger.error(f"❌ [OPTIMIZACION] Error en optimización inicial: {e}")

    def cargar_estado(self):
        """Carga el estado previo del bot desde archivo con logs detallados"""
        try:
            estado_file = self._get_config_value('estado_file', self._config_instance.estado_file)
            self.logger.debug(f"💾 [ESTADO] Cargando estado desde: {estado_file}")
            
            if os.path.exists(estado_file):
                with open(estado_file, 'r', encoding='utf-8') as f:
                    estado = json.load(f)

                # Convertir fechas
                if 'ultima_optimizacion' in estado:
                    estado['ultima_optimizacion'] = datetime.fromisoformat(estado['ultima_optimizacion'])
                if 'ultima_busqueda_config' in estado:
                    for simbolo, fecha_str in estado['ultima_busqueda_config'].items():
                        estado['ultima_busqueda_config'][simbolo] = datetime.fromisoformat(fecha_str)
                if 'breakout_history' in estado:
                    for simbolo, fecha_str in estado['breakout_history'].items():
                        estado['breakout_history'][simbolo] = datetime.fromisoformat(fecha_str)

                # Cargar breakouts y reingresos
                if 'esperando_reentry' in estado:
                    for simbolo, info in estado['esperando_reentry'].items():
                        info['timestamp'] = datetime.fromisoformat(info['timestamp'])
                        estado['esperando_reentry'][simbolo] = info
                    self.esperando_reentry = estado['esperando_reentry']

                if 'breakouts_detectados' in estado:
                    for simbolo, info in estado['breakouts_detectados'].items():
                        info['timestamp'] = datetime.fromisoformat(info['timestamp'])
                        estado['breakouts_detectados'][simbolo] = info
                    self.breakouts_detectados = estado['breakouts_detectados']

                # Cargar otros estados
                self.ultima_optimizacion = estado.get('ultima_optimizacion', datetime.now())
                self.operaciones_desde_optimizacion = estado.get('operaciones_desde_optimizacion', 0)
                self.total_operaciones = estado.get('total_operaciones', 0)
                self.breakout_history = estado.get('breakout_history', {})
                self.config_optima_por_simbolo = estado.get('config_optima_por_simbolo', {})
                self.ultima_busqueda_config = estado.get('ultima_busqueda_config', {})
                self.operaciones_activas = estado.get('operaciones_activas', {})
                self.senales_enviadas = set(estado.get('senales_enviadas', []))

                # Log detallado del estado cargado
                self.logger.info("✅ [ESTADO] Estado anterior cargado correctamente")
                self.logger.info(f"   • Operaciones activas: {len(self.operaciones_activas)}")
                self.logger.info(f"   • Esperando reentry: {len(self.esperando_reentry)}")
                self.logger.info(f"   • Breakouts detectados: {len(self.breakouts_detectados)}")
                self.logger.info(f"   • Configuraciones óptimas cacheadas: {len(self.config_optima_por_simbolo)}")
                self.logger.info(f"   • Señales enviadas (ciclo actual): {len(self.senales_enviadas)}")
                self.logger.info(f"   • Última optimización: {self.ultima_optimizacion.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Log de operaciones activas si las hay
                if self.operaciones_activas:
                    self.logger.info("📊 [ESTADO] Operaciones activas cargadas:")
                    for symbol, op in self.operaciones_activas.items():
                        tiempo_transcurrido = (datetime.now() - datetime.fromisoformat(op['timestamp_entrada'])).total_seconds() / 60
                        self.logger.info(f"   • {symbol}: {op['tipo']} - {tiempo_transcurrido:.1f}min transcurridos")
                
                # Log de reentries en espera
                if self.esperando_reentry:
                    self.logger.info("⏳ [ESTADO] Reentries en espera:")
                    for symbol, info in self.esperando_reentry.items():
                        tiempo_espera = (datetime.now() - info['timestamp']).total_seconds() / 60
                        self.logger.info(f"   • {symbol}: {info['tipo']} - {tiempo_espera:.1f}min esperando")

            else:
                self.logger.info("🆕 [ESTADO] Iniciando con estado limpio (archivo no existe)")
        except Exception as e:
            self.logger.error(f"❌ [ESTADO] Error cargando estado: {e}")
            self.logger.info("🔄 [ESTADO] Continuando con estado limpio")

    def guardar_estado(self):
        """Guarda el estado actual del bot con logs detallados"""
        try:
            timestamp_inicio = datetime.now()
            
            estado = {
                'ultima_optimizacion': self.ultima_optimizacion.isoformat(),
                'operaciones_desde_optimizacion': self.operaciones_desde_optimizacion,
                'total_operaciones': self.total_operaciones,
                'breakout_history': {k: v.isoformat() for k, v in self.breakout_history.items()},
                'config_optima_por_simbolo': self.config_optima_por_simbolo,
                'ultima_busqueda_config': {k: v.isoformat() for k, v in self.ultima_busqueda_config.items()},
                'operaciones_activas': self.operaciones_activas,
                'senales_enviadas': list(self.senales_enviadas),
                'esperando_reentry': {
                    k: {
                        'tipo': v['tipo'],
                        'timestamp': v['timestamp'].isoformat(),
                        'precio_breakout': v['precio_breakout'],
                        'config': v.get('config', {})
                    } for k, v in self.esperando_reentry.items()
                },
                'breakouts_detectados': {
                    k: {
                        'tipo': v['tipo'],
                        'timestamp': v['timestamp'].isoformat(),
                        'precio_breakout': v.get('precio_breakout', 0)
                    } for k, v in self.breakouts_detectados.items()
                },
                'timestamp_guardado': datetime.now().isoformat()
            }
            estado_file = self._get_config_value('estado_file', self._config_instance.estado_file)
            with open(estado_file, 'w', encoding='utf-8') as f:
                json.dump(estado, f, indent=2, ensure_ascii=False)
            
            timestamp_fin = datetime.now()
            tiempo_guardado = (timestamp_fin - timestamp_inicio).total_seconds()
            
            self.logger.debug(f"💾 [ESTADO] Estado guardado correctamente en {tiempo_guardado:.3f}s")
            self.logger.debug(f"   • Archivo: {estado_file}")
            self.logger.debug(f"   • Operaciones activas: {len(self.operaciones_activas)}")
            self.logger.debug(f"   • Reentries en espera: {len(self.esperando_reentry)}")
            
        except Exception as e:
            self.logger.error(f"❌ [ESTADO] Error guardando estado: {e}")

    def inicializar_log(self):
        """Inicializa el sistema de logging con logs detallados"""
        try:
            # Crear directorio de logs si no existe
            log_dir = os.path.dirname(self.log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                self.logger.debug(f"📁 [LOG] Directorio creado: {log_dir}")

            # Configurar logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(self.log_path, encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )
            
            # Configurar matplotlib para evitar problemas con fuentes
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji']
            
            self.logger.info("📝 [LOG] Sistema de logging configurado correctamente")
            self.logger.info(f"   • Archivo de log: {self.log_path}")
            self.logger.info(f"   • Nivel: INFO")
            self.logger.info(f"   • Formato: timestamp - level - message")
            
        except Exception as e:
            self.logger.error(f"❌ [LOG] Error configurando logging: {e}")

    def buscar_configuracion_optima_simbolo(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Busca la mejor configuración para un símbolo con logs detallados
        Args:
            symbol: Símbolo de trading
        Returns:
            Configuración óptima o None
        """
        try:
            timestamp_inicio = datetime.now()
            
            # Verificar cache
            if symbol in self.config_optima_por_simbolo:
                config_optima = self.config_optima_por_simbolo[symbol]
                ultima_busqueda = self.ultima_busqueda_config.get(symbol)
                if ultima_busqueda and (datetime.now() - ultima_busqueda).total_seconds() < 7200:
                    tiempo_cache = (datetime.now() - ultima_busqueda).total_seconds() / 60
                    self.logger.debug(f"🔄 [CONFIG] {symbol}: Usando configuración cacheada ({tiempo_cache:.1f}min old)")
                    return config_optima
                else:
                    self.logger.debug(f"🔄 [CONFIG] {symbol}: Reevaluando configuración (pasó 2 horas)")

            self.logger.debug(f"🔍 [CONFIG] {symbol}: Buscando configuración óptima...")
            
            mejor_config = None
            mejor_puntaje = -999999
            configs_probadas = 0

            # Obtener configuración del config
            timeframes = self._get_config_value('timeframes', self._config_instance.timeframes)
            velas_options = self._get_config_value('velas_options', self._config_instance.velas_options)
            min_channel_width_percent = self._get_config_value('min_channel_width_percent', self._config_instance.min_channel_width_percent)

            self.logger.debug(f"📊 [CONFIG] {symbol}: Parámetros de búsqueda:")
            self.logger.debug(f"   • Timeframes: {timeframes}")
            self.logger.debug(f"   • Opciones de velas: {velas_options}")
            self.logger.debug(f"   • Ancho mínimo canal: {min_channel_width_percent}%")

            # Prioridad de timeframes (más cortos = mejor)
            prioridad_timeframe = {'1m': 200, '3m': 150, '5m': 120, '15m': 100, '30m': 80, '1h': 60, '4h': 40}

            configs_validas = []
            
            # Probar combinaciones
            for timeframe in timeframes:
                for num_velas in velas_options:
                    try:
                        configs_probadas += 1
                        self.logger.debug(f"🧪 [CONFIG] {symbol}: Probando {timeframe} - {num_velas} velas")
                        
                        datos = estrategia.obtener_datos_mercado(symbol, timeframe, num_velas)
                        if not datos:
                            self.logger.debug(f"   ❌ No se pudieron obtener datos")
                            continue

                        canal_info = estrategia.calcular_canal_regresion(datos, num_velas)
                        if not canal_info:
                            self.logger.debug(f"   ❌ No se pudo calcular canal de regresión")
                            continue

                        # Filtros de calidad
                        if (canal_info.nivel_fuerza >= Constants.MIN_NIVEL_FUERZA and
                                abs(canal_info.coeficiente_pearson) >= Constants.MIN_PEARSON and
                                canal_info.r2_score >= Constants.MIN_R2_SCORE):
                            ancho_actual = canal_info.ancho_canal_porcentual
                            if ancho_actual >= min_channel_width_percent:
                                # Calcular puntaje
                                puntaje_ancho = ancho_actual * 10
                                puntaje_timeframe = prioridad_timeframe.get(timeframe, 50) * 100
                                puntaje_total = puntaje_timeframe + puntaje_ancho

                                config_info = {
                                    'timeframe': timeframe,
                                    'num_velas': num_velas,
                                    'ancho_canal': ancho_actual,
                                    'puntaje_total': puntaje_total,
                                    'nivel_fuerza': canal_info.nivel_fuerza,
                                    'pearson': canal_info.coeficiente_pearson,
                                    'r2_score': canal_info.r2_score,
                                    'direccion': canal_info.direccion
                                }
                                configs_validas.append(config_info)

                                if puntaje_total > mejor_puntaje:
                                    mejor_puntaje = puntaje_total
                                    mejor_config = config_info.copy()
                                    
                                    self.logger.debug(f"   ✅ Nueva mejor configuración: {puntaje_total} pts")
                        else:
                            self.logger.debug(f"   ❌ No cumple filtros de calidad:")
                            self.logger.debug(f"      • Nivel fuerza: {canal_info.nivel_fuerza} (< {Constants.MIN_NIVEL_FUERZA})")
                            self.logger.debug(f"      • Pearson: {abs(canal_info.coeficiente_pearson):.3f} (< {Constants.MIN_PEARSON})")
                            self.logger.debug(f"      • R²: {canal_info.r2_score:.3f} (< {Constants.MIN_R2_SCORE})")
                            
                    except Exception as e:
                        self.logger.debug(f"   ❌ Error probando {timeframe}-{num_velas}: {e}")
                        continue

            timestamp_fin = datetime.now()
            tiempo_busqueda = (timestamp_fin - timestamp_inicio).total_seconds()

            if mejor_config:
                self.config_optima_por_simbolo[symbol] = mejor_config
                self.ultima_busqueda_config[symbol] = datetime.now()
                
                self.logger.info(f"✅ [CONFIG] {symbol}: Configuración óptima encontrada:")
                self.logger.info(f"   • Timeframe: {mejor_config['timeframe']}")
                self.logger.info(f"   • Velas: {mejor_config['num_velas']}")
                self.logger.info(f"   • Ancho canal: {mejor_config['ancho_canal']:.1f}%")
                self.logger.info(f"   • Puntaje total: {mejor_config['puntaje_total']}")
                self.logger.info(f"   • Configuraciones probadas: {configs_probadas}")
                self.logger.info(f"   • Configuraciones válidas: {len(configs_validas)}")
                self.logger.info(f"   • Tiempo búsqueda: {tiempo_busqueda:.3f}s")
                
                if configs_validas:
                    self.logger.debug(f"📊 [CONFIG] {symbol}: Top 3 configuraciones válidas:")
                    configs_ordenadas = sorted(configs_validas, key=lambda x: x['puntaje_total'], reverse=True)
                    for i, config in enumerate(configs_ordenadas[:3], 1):
                        self.logger.debug(f"   {i}. {config['timeframe']}-{config['num_velas']}v: {config['puntaje_total']}pts")
                
                return mejor_config
            else:
                self.logger.warning(f"⚠️ [CONFIG] {symbol}: No se encontró configuración válida")
                self.logger.warning(f"   • Configuraciones probadas: {configs_probadas}")
                self.logger.warning(f"   • Tiempo búsqueda: {tiempo_busqueda:.3f}s")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ [CONFIG] Error buscando configuración para {symbol}: {e}")
            return None

    def detectar_breakout_y_manejar(self, symbol: str, canal_info: CanalInfo, datos_mercado: DatosMercado, config_optima: Dict[str, Any]) -> Optional[str]:
        """
        Detecta breakout y maneja el estado con logs extensivos
        Args:
            symbol: Símbolo de trading
            canal_info: Información del canal
            datos_mercado: Datos del mercado
            config_optima: Configuración óptima
        Returns:
            Tipo de breakout o None
        """
        try:
            timestamp_inicio = datetime.now()
            
            # Verificar timeout de breakout reciente
            if symbol in self.breakouts_detectados:
                ultimo_breakout = self.breakouts_detectados[symbol]
                tiempo_desde_ultimo = (datetime.now() - ultimo_breakout['timestamp']).total_seconds() / 60
                
                self.logger.debug(f"🔍 [BREAKOUT] {symbol}: Verificando timeout de breakout anterior")
                self.logger.debug(f"   • Último breakout: {ultimo_breakout['tipo']} hace {tiempo_desde_ultimo:.1f}min")
                self.logger.debug(f"   • Timeout configurado: {Constants.TIMEOUT_BREAKOUT_MINUTOS}min")
                
                if tiempo_desde_ultimo < Constants.TIMEOUT_BREAKOUT_MINUTOS:
                    self.logger.debug(f"⏰ [BREAKOUT] {symbol}: Breakout reciente, omitiendo ({tiempo_desde_ultimo:.1f} < {Constants.TIMEOUT_BREAKOUT_MINUTOS})")
                    return None
                else:
                    self.logger.debug(f"✅ [BREAKOUT] {symbol}: Timeout superado, permitiendo nuevo breakout")

            self.logger.debug(f"🔍 [BREAKOUT] {symbol}: Iniciando detección de breakout...")
            
            tipo_breakout = estrategia.detectar_breakout(symbol, canal_info, datos_mercado)
            timestamp_deteccion = datetime.now()
            tiempo_deteccion = (timestamp_deteccion - timestamp_inicio).total_seconds()
            
            if tipo_breakout:
                # Registrar breakout
                self.esperando_reentry[symbol] = {
                    'tipo': tipo_breakout,
                    'timestamp': datetime.now(),
                    'precio_breakout': datos_mercado.precio_actual,
                    'config': config_optima
                }
                self.breakouts_detectados[symbol] = {
                    'tipo': tipo_breakout,
                    'timestamp': datetime.now(),
                    'precio_breakout': datos_mercado.precio_actual
                }
                
                self.stats_bot['breakouts_registrados'] += 1
                
                self.logger.info(f"🎯 [BREAKOUT] {symbol} - {tipo_breakout} DETECTADO Y REGISTRADO:")
                self.logger.info(f"   • Precio breakout: {datos_mercado.precio_actual:.8f}")
                self.logger.info(f"   • Configuración: {config_optima['timeframe']} - {config_optima['num_velas']} velas")
                self.logger.info(f"   • Ancho canal: {canal_info.ancho_canal_porcentual:.1f}%")
                self.logger.info(f"   • Tiempo detección: {tiempo_deteccion:.3f}s")
                self.logger.info(f"   • Estado: Esperando reingreso al canal...")
                
                # Enviar alerta de breakout
                self.enviar_alerta_breakout(symbol, tipo_breakout, canal_info, datos_mercado, config_optima)

                return tipo_breakout
            else:
                self.logger.debug(f"💤 [BREAKOUT] {symbol}: No se detectó breakout en este ciclo")
                self.logger.debug(f"   • Tiempo de análisis: {tiempo_deteccion:.3f}s")
                
        except Exception as e:
            self.logger.error(f"❌ [BREAKOUT] Error manejando breakout en {symbol}: {e}")
        return None

    def enviar_alerta_breakout(self, symbol: str, tipo_breakout: str, canal_info: CanalInfo, datos_mercado: DatosMercado, config_optima: Dict[str, Any]):
        """
        Envía alerta de breakout detectado con logs detallados
        Args:
            symbol: Símbolo de trading
            tipo_breakout: Tipo de breakout
            canal_info: Información del canal
            datos_mercado: Datos del mercado
            config_optima: Configuración óptima
        """
        try:
            precio_cierre = datos_mercado.precio_actual
            resistencia = canal_info.resistencia
            soporte = canal_info.soporte
            direccion_canal = canal_info.direccion

            # Formatear mensaje según tipo
            if tipo_breakout == Constants.BREAKOUT_LONG:
                emoji_principal = "🚀"
                tipo_texto = "RUPTURA ALCISTA de RESISTENCIA"
                nivel_roto = f"Resistencia: {resistencia:.8f}"
                direccion_emoji = "⬆️"
                contexto = f"Canal {direccion_canal} → Ruptura de RESISTENCIA"
                expectativa = "posible entrada en long si el precio reingresa al canal"
            else: # BREAKOUT_SHORT
                emoji_principal = "📉"
                tipo_texto = "RUPTURA BAJISTA de SOPORTE"
                nivel_roto = f"Soporte: {soporte:.8f}"
                direccion_emoji = "⬇️"
                contexto = f"Canal {direccion_canal} → Ruptura de SOPORTE"
                expectativa = "posible entrada en short si el precio reingresa al canal"

            self.logger.debug(f"📱 [ALERTA] {symbol}: Preparando mensaje de breakout...")
            
            mensaje = f"""
{emoji_principal} **¡BREAKOUT DETECTADO! - {symbol}**
⚠️ **{tipo_texto}** {direccion_emoji}
⏰ **Hora:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
⏳ **ESPERANDO REINGRESO...**
👁️ Máximo 30 minutos para confirmación
📍 {expectativa}
"""

            # Enviar por Telegram si está configurado
            telegram_token = self._get_config_value('telegram_token', self._config_instance.telegram_token)
            telegram_chat_ids = self._get_config_value('telegram_chat_ids', self._config_instance.telegram_chat_ids)

            if telegram_token and telegram_chat_ids:
                try:
                    self.logger.debug(f"📊 [ALERTA] {symbol}: Generando gráfico de breakout...")
                    buf = self.generar_grafico_breakout(symbol, canal_info, datos_mercado, tipo_breakout, config_optima)
                    if buf:
                        self.logger.debug(f"📨 [ALERTA] {symbol}: Enviando gráfico por Telegram...")
                        telegram_client.enviar_grafico(telegram_chat_ids, buf)
                        time.sleep(0.5) # Pausa para evitar rate limits
                        self.logger.debug(f"📨 [ALERTA] {symbol}: Enviando mensaje por Telegram...")
                        telegram_client.enviar_mensaje(telegram_chat_ids, mensaje)
                        self.logger.info(f"✅ [ALERTA] {symbol}: Alerta de breakout enviada exitosamente")
                    else:
                        self.logger.warning(f"⚠️ [ALERTA] {symbol}: No se pudo generar gráfico")
                except Exception as e:
                    self.logger.error(f"❌ [ALERTA] {symbol}: Error enviando alerta: {e}")
            else:
                self.logger.debug(f"📢 [ALERTA] {symbol}: Breakout detectado (Telegram no configurado)")
                self.logger.debug(f"   • Token: {'✅ Configurado' if telegram_token else '❌ No configurado'}")
                self.logger.debug(f"   • Chat IDs: {'✅ Configurados' if telegram_chat_ids else '❌ No configurados'}")
                
        except Exception as e:
            self.logger.error(f"❌ [ALERTA] Error enviando alerta de breakout: {e}")

    def generar_grafico_breakout(self, symbol: str, canal_info: CanalInfo, datos_mercado: DatosMercado, tipo_breakout: str, config_optima: Dict[str, Any]) -> Optional[BytesIO]:
        """Genera gráfico especial para el momento del breakout usando matplotlib puro"""
        try:
            # Importar aquí para evitar problemas de dependencias
            import matplotlib.pyplot as plt
            import pandas as pd
            import matplotlib.patches as patches

            plt.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji']

            # Obtener datos frescos
            datos_raw = binance_client.obtener_datos_klines(symbol, config_optima['timeframe'], config_optima['num_velas'])
            if not datos_raw:
                self.logger.debug(f"📊 [GRAFICO] {symbol}: No se pudieron obtener datos para el gráfico")
                return None

            # Crear DataFrame
            df_data = []
            for kline in datos_raw:
                df_data.append({
                    'Date': pd.to_datetime(kline[0], unit='ms'),
                    'Open': float(kline[1]),
                    'High': float(kline[2]),
                    'Low': float(kline[3]),
                    'Close': float(kline[4]),
                    'Volume': float(kline[5])
                })
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)

            # Calcular líneas del canal
            tiempos_reg = list(range(len(df)))
            resistencia_values = []
            soporte_values = []
            for i, t in enumerate(tiempos_reg):
                resist = canal_info.pendiente_resistencia * t + (canal_info.resistencia - canal_info.pendiente_resistencia * tiempos_reg[-1])
                sop = canal_info.pendiente_soporte * t + (canal_info.soporte - canal_info.pendiente_soporte * tiempos_reg[-1])
                resistencia_values.append(resist)
                soporte_values.append(sop)

            # Marcar zona de breakout
            precio_breakout = datos_mercado.precio_actual
            breakout_line = [precio_breakout] * len(df)
            color_breakout = "#D68F01"
            titulo_extra = "🚀 RUPTURA ALCISTA" if tipo_breakout == Constants.BREAKOUT_LONG else "📉 RUPTURA BAJISTA"

            # Crear gráfico con matplotlib puro
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                          gridspec_kw={'height_ratios': [3, 1]},
                                          facecolor='#1a1a1a')

            # Panel principal - Gráfico de velas
            for i, (date, row) in enumerate(df.iterrows()):
                open_price = row['Open']
                high_price = row['High']
                low_price = row['Low']
                close_price = row['Close']

                # Color de la vela
                color = '#00FF00' if close_price >= open_price else '#FF0000'
                edge_color = '#FFFFFF' if close_price >= open_price else '#FF0000'

                # Cuerpo de la vela
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)

                # Dibujar cuerpo
                rect = patches.Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                                         linewidth=1, edgecolor=edge_color,
                                         facecolor=color, alpha=0.8)
                ax1.add_patch(rect)

                # Mecha superior
                ax1.plot([i, i], [high_price, max(open_price, close_price)],
                         color=edge_color, linewidth=1)
                # Mecha inferior
                ax1.plot([i, i], [low_price, min(open_price, close_price)],
                         color=edge_color, linewidth=1)

            # Agregar líneas del canal
            ax1.plot(tiempos_reg, resistencia_values, color='#5444ff',
                     linestyle='--', linewidth=2, label='Resistencia', alpha=0.8)
            ax1.plot(tiempos_reg, soporte_values, color='#5444ff',
                     linestyle='--', linewidth=2, label='Soporte', alpha=0.8)

            # Línea de breakout
            ax1.plot(tiempos_reg, breakout_line, color=color_breakout,
                     linestyle='-', linewidth=3, label='Precio Breakout', alpha=0.9)

            # Configurar ejes
            ax1.set_title(f'{symbol} | {titulo_extra} | {config_optima["timeframe"]} | ⏳ ESPERANDO REENTRY',
                          color='white', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Precio', color='white')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            # Configurar colores del panel principal
            ax1.tick_params(colors='white')
            ax1.spines['bottom'].set_color('white')
            ax1.spines['top'].set_color('white')
            ax1.spines['right'].set_color('white')
            ax1.spines['left'].set_color('white')

            # Panel de volumen
            colors = ['#00FF00' if close >= open else '#FF0000'
                      for open, close in zip(df['Open'], df['Close'])]
            ax2.bar(range(len(df)), df['Volume'], color=colors, alpha=0.7)
            ax2.set_ylabel('Volumen', color='white')
            ax2.set_xlabel('Tiempo', color='white')
            ax2.tick_params(colors='white')
            ax2.spines['bottom'].set_color('white')
            ax2.spines['top'].set_color('white')
            ax2.spines['right'].set_color('white')
            ax2.spines['left'].set_color('white')

            # Configurar fondo
            fig.patch.set_facecolor('#1a1a1a')

            # Ajustar layout
            plt.tight_layout()

            # Guardar en buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                        facecolor='#1a1a1a', edgecolor='none')
            buf.seek(0)
            plt.close(fig)
            
            self.logger.debug(f"📊 [GRAFICO] {symbol}: Gráfico de breakout generado exitosamente")
            return buf
            
        except Exception as e:
            self.logger.error(f"❌ [GRAFICO] Error generando gráfico de breakout: {e}")
            return None

    def escanear_mercado(self) -> int:
        """
        Escanea el mercado completo buscando señales con logs extensivos
        Returns:
            Número de señales encontradas
        """
        try:
            timestamp_ciclo_inicio = datetime.now()
            
            symbols = self._get_config_value('symbols', self._config_instance.symbols)
            self.stats_bot['ciclos_escaneo'] += 1
            
            self.logger.info(f"\n🔍 [ESCANEO] Iniciando ciclo #{self.stats_bot['ciclos_escaneo']} - Escaneando {len(symbols)} símbolos")
            self.logger.info(f"   • Estrategia: Breakout + Reentry")
            self.logger.info(f"   • Hora inicio: {timestamp_ciclo_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"   • Operaciones activas: {len(self.operaciones_activas)}")
            self.logger.info(f"   • Reentries en espera: {len(self.esperando_reentry)}")
            
            senales_encontradas = 0
            simbolos_procesados = 0
            simbolos_con_error = 0
            errores_por_simbolo = {}
            
            # Limpiar señales enviadas del ciclo anterior
            self.senales_enviadas.clear()
            
            # Estado inicial para logging
            estado_inicial = {
                'breakouts_en_espera': len(self.esperando_reentry),
                'operaciones_activas': len(self.operaciones_activas),
                'breakouts_recientes': len(self.breakouts_detectados)
            }

            for symbol in symbols:
                try:
                    timestamp_simbolo_inicio = datetime.now()
                    simbolos_procesados += 1
                    
                    self.logger.debug(f"\n📊 [ESCANEO] Procesando {symbol} ({simbolos_procesados}/{len(symbols)})")
                    
                    # Verificar si hay operación activa
                    if symbol in self.operaciones_activas:
                        operacion = self.operaciones_activas[symbol]
                        tiempo_transcurrido = (datetime.now() - datetime.fromisoformat(operacion['timestamp_entrada'])).total_seconds() / 60
                        self.logger.debug(f"   ⚡ {symbol}: Operación activa ({operacion['tipo']}, {tiempo_transcurrido:.1f}min), omitiendo...")
                        continue

                    # Buscar configuración óptima
                    config_optima = self.buscar_configuracion_optima_simbolo(symbol)
                    if not config_optima:
                        self.logger.debug(f"   ❌ {symbol}: No se encontró configuración válida")
                        continue

                    # Obtener datos del mercado
                    datos_mercado = estrategia.obtener_datos_mercado(symbol, config_optima['timeframe'], config_optima['num_velas'])
                    if not datos_mercado:
                        self.logger.debug(f"   ❌ {symbol}: Error obteniendo datos")
                        continue

                    # Calcular canal de regresión
                    canal_info = estrategia.calcular_canal_regresion(datos_mercado, config_optima['num_velas'])
                    if not canal_info:
                        self.logger.debug(f"   ❌ {symbol}: Error calculando canal")
                        continue

                    # Log de estado actual con detalles técnicos
                    estado_stoch = ""
                    if canal_info.stoch_k <= Constants.STOCH_OVERSOLD:
                        estado_stoch = "📉 OVERSOLD"
                    elif canal_info.stoch_k >= Constants.STOCH_OVERBOUGHT:
                        estado_stoch = "📈 OVERBOUGHT"
                    else:
                        estado_stoch = "➖ NEUTRO"

                    posicion = ""
                    if datos_mercado.precio_actual > canal_info.resistencia:
                        posicion = "🔼 FUERA (arriba)"
                    elif datos_mercado.precio_actual < canal_info.soporte:
                        posicion = "🔽 FUERA (abajo)"
                    else:
                        posicion = "📍 DENTRO"

                    self.logger.debug(
                        f"📊 {symbol} - {config_optima['timeframe']} - {config_optima['num_velas']}v | "
                        f"{canal_info.direccion} ({canal_info.angulo_tendencia:.1f}° - {canal_info.fuerza_texto}) | "
                        f"Ancho: {canal_info.ancho_canal_porcentual:.1f}% - Stoch: {canal_info.stoch_k:.1f}/{canal_info.stoch_d:.1f} {estado_stoch} | "
                        f"Precio: {posicion}"
                    )

                    # Filtros de calidad con logs detallados
                    filtros_pasados = True
                    filtros_detalle = []
                    
                    if canal_info.nivel_fuerza < Constants.MIN_NIVEL_FUERZA:
                        filtros_pasados = False
                        filtros_detalle.append(f"Fuerza insuficiente: {canal_info.nivel_fuerza} < {Constants.MIN_NIVEL_FUERZA}")
                    
                    if abs(canal_info.coeficiente_pearson) < Constants.MIN_PEARSON:
                        filtros_pasados = False
                        filtros_detalle.append(f"Pearson insuficiente: {abs(canal_info.coeficiente_pearson):.3f} < {Constants.MIN_PEARSON}")
                    
                    if canal_info.r2_score < Constants.MIN_R2_SCORE:
                        filtros_pasados = False
                        filtros_detalle.append(f"R² insuficiente: {canal_info.r2_score:.3f} < {Constants.MIN_R2_SCORE}")
                    
                    if not filtros_pasados:
                        self.logger.debug(f"   ❌ {symbol}: No pasa filtros de calidad:")
                        for detalle in filtros_detalle:
                            self.logger.debug(f"      • {detalle}")
                        continue
                    
                    self.logger.debug(f"   ✅ {symbol}: Pasa todos los filtros de calidad")

                    # Manejar breakouts y reentries
                    if symbol not in self.esperando_reentry:
                        # Detectar nuevo breakout
                        self.logger.debug(f"   🔍 {symbol}: Buscando nuevo breakout...")
                        tipo_breakout = self.detectar_breakout_y_manejar(symbol, canal_info, datos_mercado, config_optima)
                        if tipo_breakout:
                            self.logger.debug(f"   ✅ {symbol}: Breakout detectado, saltando a siguiente símbolo")
                            continue
                    else:
                        # Verificar reentry
                        breakout_info = self.esperando_reentry[symbol]
                        tiempo_espera = (datetime.now() - breakout_info['timestamp']).total_seconds() / 60
                        
                        self.logger.debug(f"   🔄 {symbol}: Verificando reentry (esperando {tiempo_espera:.1f}min)...")
                        
                        tipo_operacion = estrategia.detectar_reentry(symbol, canal_info, datos_mercado, breakout_info)

                        if tipo_operacion:
                            # Generar señal de operación
                            precio_entrada, tp, sl = estrategia.calcular_niveles_entrada(tipo_operacion, canal_info, datos_mercado.precio_actual)
                            if precio_entrada and tp and sl:
                                # Verificar cooldown
                                cooldown_ok = True
                                if symbol in self.breakout_history:
                                    ultimo_breakout = self.breakout_history[symbol]
                                    tiempo_desde_ultimo = (datetime.now() - ultimo_breakout).total_seconds() / 3600
                                    if tiempo_desde_ultimo < 2: # Cooldown de 2 horas entre señales para el mismo símbolo
                                        cooldown_ok = False
                                        self.logger.debug(f"   ⏳ {symbol}: Cooldown activo ({tiempo_desde_ultimo:.1f}h < 2h), omitiendo...")
                                
                                if cooldown_ok:
                                    timestamp_operacion = datetime.now()
                                    self.generar_senal_operacion(symbol, tipo_operacion, precio_entrada, tp, sl, canal_info, datos_mercado, config_optima, breakout_info)
                                    senales_encontradas += 1
                                    self.breakout_history[symbol] = datetime.now()
                                    self.stats_bot['senales_generadas'] += 1
                                    
                                    timestamp_operacion_fin = datetime.now()
                                    tiempo_operacion = (timestamp_operacion_fin - timestamp_operacion).total_seconds()
                                    
                                    self.logger.info(f"✅ [SEÑAL] {symbol} - {tipo_operacion} GENERADA:")
                                    self.logger.info(f"   • Tiempo procesamiento: {tiempo_operacion:.3f}s")
                                    del self.esperando_reentry[symbol]
                                else:
                                    self.logger.debug(f"   ⏳ {symbol}: Señal bloqueada por cooldown")
                            else:
                                self.logger.debug(f"   ❌ {symbol}: Niveles de entrada inválidos")
                        else:
                            self.logger.debug(f"   💤 {symbol}: No hay reentry confirmado")
                    
                    # Log de tiempo de procesamiento del símbolo
                    timestamp_simbolo_fin = datetime.now()
                    tiempo_simbolo = (timestamp_simbolo_fin - timestamp_simbolo_inicio).total_seconds()
                    self.logger.debug(f"   ⏱️ {symbol}: Tiempo procesamiento: {tiempo_simbolo:.3f}s")

                except Exception as e:
                    simbolos_con_error += 1
                    errores_por_simbolo[symbol] = str(e)
                    self.stats_bot['errores_por_simbolo'][symbol] = self.stats_bot['errores_por_simbolo'].get(symbol, 0) + 1
                    
                    self.logger.error(f"⚠️ [ESCANEO] Error analizando {symbol}: {e}")
                    continue # Continuar con el siguiente símbolo

            # Log de estado de espera actualizado
            if self.esperando_reentry:
                self.logger.info(f"\n⏳ [ESCANEO] Esperando reingreso en {len(self.esperando_reentry)} símbolos:")
                for symbol, info in self.esperando_reentry.items():
                    tiempo_espera = (datetime.now() - info['timestamp']).total_seconds() / 60
                    self.logger.info(f"   • {symbol} - {info['tipo']} - Esperando {tiempo_espera:.1f} min")

            if self.breakouts_detectados:
                self.logger.info(f"\n⏰ [ESCANEO] Breakouts detectados recientemente:")
                for symbol, info in self.breakouts_detectados.items():
                    tiempo_desde_deteccion = (datetime.now() - info['timestamp']).total_seconds() / 60
                    self.logger.info(f"   • {symbol} - {info['tipo']} - Hace {tiempo_desde_deteccion:.1f} min")

            # Resumen del ciclo
            timestamp_ciclo_fin = datetime.now()
            tiempo_ciclo_total = (timestamp_ciclo_fin - timestamp_ciclo_inicio).total_seconds()
            self.stats_bot['tiempo_total_procesamiento'] += tiempo_ciclo_total
            
            self.stats_bot['simbolos_analizados'] += simbolos_procesados
            
            self.logger.info(f"\n📊 [ESCANEO] Resumen del ciclo #{self.stats_bot['ciclos_escaneo']}:")
            self.logger.info(f"   • Símbolos procesados: {simbolos_procesados}/{len(symbols)}")
            self.logger.info(f"   • Símbolos con error: {simbolos_con_error}")
            self.logger.info(f"   • Señales encontradas: {senales_encontradas}")
            self.logger.info(f"   • Tiempo total ciclo: {tiempo_ciclo_total:.3f}s")
            self.logger.info(f"   • Tiempo promedio por símbolo: {tiempo_ciclo_total/simbolos_procesados if simbolos_procesados > 0 else 0:.3f}s")
            
            if senales_encontradas > 0:
                self.logger.info(f"✅ [ESCANEO] Se encontraron {senales_encontradas} señales de trading en este ciclo")
            else:
                self.logger.info(f"❌ [ESCANEO] No se encontraron señales en este ciclo")
            
            # Estadísticas acumuladas
            self.logger.info(f"\n📈 [ESTADISTICAS] Acumuladas:")
            self.logger.info(f"   • Ciclos ejecutados: {self.stats_bot['ciclos_escaneo']}")
            self.logger.info(f"   • Símbolos analizados total: {self.stats_bot['simbolos_analizados']}")
            self.logger.info(f"   • Breakouts registrados: {self.stats_bot['breakouts_registrados']}")
            self.logger.info(f"   • Señales generadas: {self.stats_bot['senales_generadas']}")
            self.logger.info(f"   • Tiempo total procesamiento: {self.stats_bot['tiempo_total_procesamiento']:.3f}s")
            
            if errores_por_simbolo:
                self.logger.warning(f"⚠️ [ERRORES] Símbolos con problemas:")
                for symbol, error in errores_por_simbolo.items():
                    self.logger.warning(f"   • {symbol}: {error}")

            return senales_encontradas

        except Exception as e:
            self.logger.error(f"❌ [ESCANEO] Error en escaneo de mercado: {e}")
            return 0

    def generar_senal_operacion(self, symbol: str, tipo_operacion: str, precio_entrada: float, tp: float, sl: float, canal_info: CanalInfo, datos_mercado: DatosMercado, config_optima: Dict[str, Any], breakout_info: Dict[str, Any] = None):
        """Genera y envía señal de operación con logs extensivos"""
        try:
            timestamp_inicio = datetime.now()
            
            if symbol in self.senales_enviadas:
                # Ya se envió una señal para este símbolo en este ciclo, evitar duplicados
                self.logger.debug(f"⚠️ [SEÑAL] {symbol}: Señal ya enviada en este ciclo, omitiendo")
                return

            if not all([precio_entrada, tp, sl]):
                self.logger.error(f"❌ [SEÑAL] {symbol}: Niveles inválidos, omitiendo señal")
                return

            # Calcular métricas
            riesgo = abs(precio_entrada - sl)
            beneficio = abs(tp - precio_entrada)
            ratio_rr = beneficio / riesgo if riesgo > 0 else 0
            sl_percent = abs((sl - precio_entrada) / precio_entrada) * 100
            tp_percent = abs((tp - precio_entrada) / precio_entrada) * 100
            stoch_estado = "📉 SOBREVENTA" if tipo_operacion == Constants.OPERACION_LONG else "📈 SOBRECOMPRA"

            self.logger.info(f"💰 [SEÑAL] {symbol} - {tipo_operacion}: Calculando niveles de trading...")
            self.logger.info(f"   • Precio entrada: {precio_entrada:.8f}")
            self.logger.info(f"   • Take Profit: {tp:.8f}")
            self.logger.info(f"   • Stop Loss: {sl:.8f}")
            self.logger.info(f"   • Ratio R/R: {ratio_rr:.2f}:1")
            self.logger.info(f"   • SL %: {sl_percent:.2f}%")
            self.logger.info(f"   • TP %: {tp_percent:.2f}%")

            # Formatear mensaje
            breakout_texto = ""
            if breakout_info:
                tiempo_breakout = (datetime.now() - breakout_info['timestamp']).total_seconds() / 60
                breakout_texto = f"""
🚀 **BREAKOUT + REENTRY DETECTADO:**
⏰ Tiempo desde breakout: {tiempo_breakout:.1f} minutos
💰 Precio breakout: {breakout_info['precio_breakout']:.8f}
"""

            mensaje = f"""
🎯 **SEÑAL DE {tipo_operacion} - {symbol}**
{breakout_texto}
⏱️ **Configuración óptima:**
📊 Timeframe: {config_optima['timeframe']}
🕯️ Velas: {config_optima['num_velas']}
📏 Ancho Canal: {canal_info.ancho_canal_porcentual:.1f}% ⭐
💰 **Precio Actual:** {datos_mercado.precio_actual:.8f}
🎯 **Entrada:** {precio_entrada:.8f}
🛑 **Stop Loss:** {sl:.8f}
🎯 **Take Profit:** {tp:.8f}
📊 **Ratio R/B:** {ratio_rr:.2f}:1
🎯 **SL:** {sl_percent:.2f}%
🎯 **TP:** {tp_percent:.2f}%
💰 **Riesgo:** {riesgo:.8f}
🎯 **Beneficio Objetivo:** {beneficio:.8f}
📈 **Tendencia:** {canal_info.direccion}
💪 **Fuerza:** {canal_info.fuerza_texto}
📏 **Ángulo:** {canal_info.angulo_tendencia:.1f}°
📊 **Pearson:** {canal_info.coeficiente_pearson:.3f}
🎯 **R² Score:** {canal_info.r2_score:.3f}
🎰 **Stochástico:** {stoch_estado}
📊 **Stoch K:** {canal_info.stoch_k:.1f}
📈 **Stoch D:** {canal_info.stoch_d:.1f}
⏰ **Hora:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
💡 **Estrategia:** BREAKOUT + REENTRY con confirmación Stochastic
"""

            # Enviar por Telegram
            telegram_token = self._get_config_value('telegram_token', config.telegram_token)
            telegram_chat_ids = self._get_config_value('telegram_chat_ids', config.telegram_chat_ids)

            if telegram_token and telegram_chat_ids:
                try:
                    self.logger.debug(f"📊 [SEÑAL] {symbol}: Generando gráfico profesional...")
                    buf = self.generar_grafico_profesional(symbol, canal_info, datos_mercado, precio_entrada, tp, sl, tipo_operacion)
                    if buf:
                        self.logger.debug(f"📨 [SEÑAL] {symbol}: Enviando gráfico por Telegram...")
                        telegram_client.enviar_grafico(telegram_chat_ids, buf)
                        time.sleep(1) # Pausa para evitar rate limits
                        self.logger.debug(f"📨 [SEÑAL] {symbol}: Enviando mensaje por Telegram...")
                        telegram_client.enviar_mensaje(telegram_chat_ids, mensaje)
                        self.logger.info(f"✅ [SEÑAL] {symbol}: Señal {tipo_operacion} enviada exitosamente")
                    else:
                        self.logger.warning(f"⚠️ [SEÑAL] {symbol}: No se pudo generar gráfico, enviando solo mensaje")
                        telegram_client.enviar_mensaje(telegram_chat_ids, mensaje)
                except Exception as e:
                    self.logger.error(f"❌ [SEÑAL] {symbol}: Error enviando señal: {e}")
            else:
                self.logger.debug(f"📢 [SEÑAL] {symbol}: Señal generada (Telegram no configurado)")
                self.logger.debug(f"   • Token: {'✅ Configurado' if telegram_token else '❌ No configurado'}")
                self.logger.debug(f"   • Chat IDs: {'✅ Configurados' if telegram_chat_ids else '❌ No configurados'}")

            # Registrar operación activa
            self.operaciones_activas[symbol] = {
                'tipo': tipo_operacion,
                'precio_entrada': precio_entrada,
                'take_profit': tp,
                'stop_loss': sl,
                'timestamp_entrada': datetime.now().isoformat(),
                'angulo_tendencia': canal_info.angulo_tendencia,
                'pearson': canal_info.coeficiente_pearson,
                'r2_score': canal_info.r2_score,
                'ancho_canal_relativo': canal_info.ancho_canal / precio_entrada,
                'ancho_canal_porcentual': canal_info.ancho_canal_porcentual,
                'nivel_fuerza': canal_info.nivel_fuerza,
                'timeframe_utilizado': config_optima['timeframe'],
                'velas_utilizadas': config_optima['num_velas'],
                'stoch_k': canal_info.stoch_k,
                'stoch_d': canal_info.stoch_d,
                'breakout_usado': breakout_info is not None
            }
            
            self.senales_enviadas.add(symbol)
            self.total_operaciones += 1
            self.stats_bot['operaciones_activas'] = len(self.operaciones_activas)
            
            timestamp_fin = datetime.now()
            tiempo_procesamiento = (timestamp_fin - timestamp_inicio).total_seconds()
            
            self.logger.info(f"📝 [SEÑAL] {symbol}: Operación registrada en estado activo")
            self.logger.info(f"   • Total operaciones: {self.total_operaciones}")
            self.logger.info(f"   • Operaciones activas: {len(self.operaciones_activas)}")
            self.logger.info(f"   • Tiempo procesamiento: {tiempo_procesamiento:.3f}s")

        except Exception as e:
            self.logger.error(f"❌ [SEÑAL] Error generando señal de operación: {e}")

    def generar_grafico_profesional(self, symbol: str, canal_info: CanalInfo, datos_mercado: DatosMercado, precio_entrada: float, tp: float, sl: float, tipo_operacion: str) -> Optional[BytesIO]:
        """Genera gráfico profesional para señales usando matplotlib puro"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import matplotlib.patches as patches

            plt.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji']

            config_optima = self.config_optima_por_simbolo.get(symbol)
            if not config_optima:
                return None

            # Obtener datos
            datos_raw = binance_client.obtener_datos_klines(symbol, config_optima['timeframe'], config_optima['num_velas'])
            if not datos_raw:
                return None

            # Crear DataFrame
            df_data = []
            for kline in datos_raw:
                df_data.append({
                    'Date': pd.to_datetime(kline[0], unit='ms'),
                    'Open': float(kline[1]),
                    'High': float(kline[2]),
                    'Low': float(kline[3]),
                    'Close': float(kline[4]),
                    'Volume': float(kline[5])
                })
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)

            # Líneas del canal
            tiempos_reg = list(range(len(df)))
            resistencia_values = []
            soporte_values = []
            for i, t in enumerate(tiempos_reg):
                resist = canal_info.pendiente_resistencia * t + (canal_info.resistencia - canal_info.pendiente_resistencia * tiempos_reg[-1])
                sop = canal_info.pendiente_soporte * t + (canal_info.soporte - canal_info.pendiente_soporte * tiempos_reg[-1])
                resistencia_values.append(resist)
                soporte_values.append(sop)

            # Niveles de entrada
            entry_line = [precio_entrada] * len(df)
            tp_line = [tp] * len(df)
            sl_line = [sl] * len(df)

            # Crear gráfico con matplotlib puro
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                          gridspec_kw={'height_ratios': [3, 1]},
                                          facecolor='#1a1a1a')

            # Panel principal - Gráfico de velas
            for i, (date, row) in enumerate(df.iterrows()):
                open_price = row['Open']
                high_price = row['High']
                low_price = row['Low']
                close_price = row['Close']

                # Color de la vela
                color = '#00FF00' if close_price >= open_price else '#FF0000'
                edge_color = '#FFFFFF' if close_price >= open_price else '#FF0000'

                # Cuerpo de la vela
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)

                # Dibujar cuerpo
                rect = patches.Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                                         linewidth=1, edgecolor=edge_color,
                                         facecolor=color, alpha=0.8)
                ax1.add_patch(rect)

                # Mecha superior
                ax1.plot([i, i], [high_price, max(open_price, close_price)],
                         color=edge_color, linewidth=1)
                # Mecha inferior
                ax1.plot([i, i], [low_price, min(open_price, close_price)],
                         color=edge_color, linewidth=1)

            # Agregar líneas del canal
            ax1.plot(tiempos_reg, resistencia_values, color='#5444ff',
                     linestyle='--', linewidth=2, label='Resistencia', alpha=0.8)
            ax1.plot(tiempos_reg, soporte_values, color='#5444ff',
                     linestyle='--', linewidth=2, label='Soporte', alpha=0.8)

            # Líneas de niveles de trading
            ax1.plot(tiempos_reg, entry_line, color='#FFD700',
                     linestyle='-', linewidth=2, label='Entrada', alpha=0.9)
            ax1.plot(tiempos_reg, tp_line, color='#00FF00',
                     linestyle='-', linewidth=2, label='Take Profit', alpha=0.9)
            ax1.plot(tiempos_reg, sl_line, color='#FF0000',
                     linestyle='-', linewidth=2, label='Stop Loss', alpha=0.9)

            # Configurar ejes
            ax1.set_title(f'{symbol} | {tipo_operacion} | {config_optima["timeframe"]} | Breakout+Reentry',
                          color='white', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Precio', color='white')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            # Configurar colores del panel principal
            ax1.tick_params(colors='white')
            ax1.spines['bottom'].set_color('white')
            ax1.spines['top'].set_color('white')
            ax1.spines['right'].set_color('white')
            ax1.spines['left'].set_color('white')

            # Panel de volumen
            colors = ['#00FF00' if close >= open else '#FF0000'
                      for open, close in zip(df['Open'], df['Close'])]
            ax2.bar(range(len(df)), df['Volume'], color=colors, alpha=0.7)
            ax2.set_ylabel('Volumen', color='white')
            ax2.set_xlabel('Tiempo', color='white')
            ax2.tick_params(colors='white')
            ax2.spines['bottom'].set_color('white')
            ax2.spines['top'].set_color('white')
            ax2.spines['right'].set_color('white')
            ax2.spines['left'].set_color('white')

            # Configurar fondo
            fig.patch.set_facecolor('#1a1a1a')

            # Ajustar layout
            plt.tight_layout()

            # Guardar en buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                        facecolor='#1a1a1a', edgecolor='none')
            buf.seek(0)
            plt.close(fig)
            
            return buf
            
        except Exception as e:
            self.logger.error(f"❌ [GRAFICO] Error generando gráfico profesional: {e}")
            return None

    def obtener_estadisticas_bot(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas del bot"""
        stats = self.stats_bot.copy()
        stats.update({
            'total_operaciones': self.total_operaciones,
            'operaciones_activas': len(self.operaciones_activas),
            'reentries_en_espera': len(self.esperando_reentry),
            'breakouts_recientes': len(self.breakouts_detectados),
            'configuraciones_cacheadas': len(self.config_optima_por_simbolo)
        })
        return stats

    def iniciar(self):
        """Inicia el bot con logs detallados"""
        self.logger.info("🚀 [BOT] Iniciando bot de trading...")
        self.logger.info(f"   • Configuración cargada: ✅")
        self.logger.info(f"   • Estado inicializado: ✅")
        self.logger.info(f"   • Auto-optimización: {'✅' if self.auto_optimize else '❌'}")
        
        try:
            # Bucle principal del bot
            while True:
                timestamp_ciclo = datetime.now()
                
                # Escanear mercado
                senales_encontradas = self.escanear_mercado()
                
                # Guardar estado cada ciclo
                self.guardar_estado()
                
                # Log de ciclo completado
                timestamp_ciclo_fin = datetime.now()
                tiempo_ciclo = (timestamp_ciclo_fin - timestamp_ciclo).total_seconds()
                
                self.logger.info(f"🏁 [BOT] Ciclo completado en {tiempo_ciclo:.3f}s - {senales_encontradas} señales")
                
                # Esperar antes del siguiente ciclo
                interval = self._get_config_value('scan_interval_minutes', self._config_instance.scan_interval_minutes)
                tiempo_espera = interval * 60
                
                self.logger.debug(f"⏰ [BOT] Esperando {interval} minutos hasta el siguiente ciclo...")
                time.sleep(tiempo_espera)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 [BOT] Deteniendo bot por interrupción del usuario...")
        except Exception as e:
            self.logger.error(f"❌ [BOT] Error en bucle principal: {e}")
        finally:
            self.logger.info("👋 [BOT] Bot detenido correctamente")
            self.guardar_estado()
