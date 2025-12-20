"""
Bot principal de trading.
Orquesta toda la lógica de trading, manejo de estado y coordinación.
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
import mplfinance as mpf # Explicitly added from context
import pandas as pd # Explicitly added from context

class TradingBot:
    """Bot principal de trading Breakout + Reentry"""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Inicializa el bot de trading
        Args:
            config_dict: Configuración del bot (diccionario o instancia)
        """
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
        self.log_path = self._get_config_value('log_path', config.log_path)
        self.auto_optimize = self._get_config_value('auto_optimize', config.auto_optimize)
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

        # Cargar estado previo
        self.cargar_estado()

        # Optimización automática
        if self.auto_optimize:
            self._ejecutar_optimizacion_inicial()

        # Inicializar log
        self.inicializar_log()
        self.logger.info("🤖 Bot de trading inicializado correctamente")
    
    def _get_config_value(self, key: str, default_value: Any = None):
        """Obtiene valor de configuración de manera segura"""
        return self.config.get(key, default_value)

    def _ejecutar_optimizacion_inicial(self):
        """Ejecuta optimización inicial si está habilitada"""
        try:
            self.logger.info("🔄 Ejecutando optimización inicial...")
            parametros_optimizados = optimizador_ia.buscar_mejores_parametros()
            if parametros_optimizados:
                self.config['trend_threshold_degrees'] = parametros_optimizados.get('trend_threshold_degrees',
                                                                                self.config.get('trend_threshold_degrees', 16))
                self.config['min_trend_strength_degrees'] = parametros_optimizados.get('min_trend_strength_degrees',
                                                                                self.config.get('min_trend_strength_degrees', 16))
                self.config['entry_margin'] = parametros_optimizados.get('entry_margin',
                                                                                self.config.get('entry_margin', 0.001))
                self.logger.info("✅ Parámetros optimizados aplicados")
            else:
                self.logger.warning("⚠️ No se pudieron obtener parámetros optimizados")
        except Exception as e:
            self.logger.error(f"Error en optimización inicial: {e}")

    def cargar_estado(self):
        """Carga el estado previo del bot desde archivo"""
        try:
            estado_file = self._get_config_value('estado_file', config.estado_file)
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
                self.logger.info("✅ Estado anterior cargado correctamente")
                self.logger.info(f" 📊 Operaciones activas: {len(self.operaciones_activas)}")
                self.logger.info(f" ⏳ Esperando reentry: {len(self.esperando_reentry)}")
            else:
                self.logger.info("🆕 Iniciando con estado limpio")
        except Exception as e:
            self.logger.error(f"Error cargando estado: {e}")
            self.logger.info("Continuando con estado limpio")

    def guardar_estado(self):
        """Guarda el estado actual del bot"""
        try:
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
            estado_file = self._get_config_value('estado_file', config.estado_file)
            with open(estado_file, 'w', encoding='utf-8') as f:
                json.dump(estado, f, indent=2, ensure_ascii=False)
            self.logger.debug("💾 Estado guardado correctamente")
        except Exception as e:
            self.logger.error(f"Error guardando estado: {e}")

    def inicializar_log(self):
        """Inicializa el sistema de logging"""
        try:
            # Crear directorio de logs si no existe
            log_dir = os.path.dirname(self.log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
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
            
            self.logger.info("📝 Sistema de logging configurado correctamente")
        except Exception as e:
            self.logger.error(f"Error configurando logging: {e}")

    def buscar_configuracion_optima_simbolo(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Busca la mejor configuración para un símbolo
        Args:
            symbol: Símbolo de trading
        Returns:
            Configuración óptima o None
        """
        try:
            # Verificar cache
            if symbol in self.config_optima_por_simbolo:
                config_optima = self.config_optima_por_simbolo[symbol]
                ultima_busqueda = self.ultima_busqueda_config.get(symbol)
                if ultima_busqueda and (datetime.now() - ultima_busqueda).total_seconds() < 7200:
                    return config_optima
                else:
                    self.logger.debug(f" 🔄 Reevaluando configuración para {symbol} (pasó 2 horas)")

            self.logger.debug(f" 🔍 Buscando configuración óptima para {symbol}...")
            mejor_config = None
            mejor_puntaje = -999999

            # Obtener configuración del config
            timeframes = self._get_config_value('timeframes', config.timeframes)
            velas_options = self._get_config_value('velas_options', config.velas_options)
            min_channel_width_percent = self._get_config_value('min_channel_width_percent', config.min_channel_width_percent)

            # Prioridad de timeframes (más cortos = mejor)
            prioridad_timeframe = {'1m': 200, '3m': 150, '5m': 120, '15m': 100, '30m': 80, '1h': 60, '4h': 40}

            # Probar combinaciones
            for timeframe in timeframes:
                for num_velas in velas_options:
                    try:
                        datos = estrategia.obtener_datos_mercado(symbol, timeframe, num_velas)
                        if not datos:
                            continue
                        canal_info = estrategia.calcular_canal_regresion(datos, num_velas)
                        if not canal_info:
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

                                if puntaje_total > mejor_puntaje:
                                    mejor_puntaje = puntaje_total
                                    mejor_config = {
                                        'timeframe': timeframe,
                                        'num_velas': num_velas,
                                        'ancho_canal': ancho_actual,
                                        'puntaje_total': puntaje_total
                                    }
                    except Exception as e:
                        self.logger.debug(f"Error probando {symbol}-{timeframe}-{num_velas}: {e}")
                        continue

            if mejor_config:
                self.config_optima_por_simbolo[symbol] = mejor_config
                self.ultima_busqueda_config[symbol] = datetime.now()
                self.logger.debug(f" ✅ Config óptima: {mejor_config['timeframe']} - {mejor_config['num_velas']} velas - Ancho: {mejor_config['ancho_canal']:.1f}%")
                return mejor_config

        except Exception as e:
            self.logger.error(f"Error buscando configuración para {symbol}: {e}")
        return None

    def detectar_breakout_y_manejar(self, symbol: str, canal_info: CanalInfo, datos_mercado: DatosMercado, config_optima: Dict[str, Any]) -> Optional[str]:
        """
        Detecta breakout y maneja el estado
        Args:
            symbol: Símbolo de trading
            canal_info: Información del canal
            datos_mercado: Datos del mercado
            config_optima: Configuración óptima
        Returns:
            Tipo de breakout o None
        """
        try:
            # Verificar timeout de breakout reciente
            if symbol in self.breakouts_detectados:
                ultimo_breakout = self.breakouts_detectados[symbol]
                tiempo_desde_ultimo = (datetime.now() - ultimo_breakout['timestamp']).total_seconds() / 60
                if tiempo_desde_ultimo < Constants.TIMEOUT_BREAKOUT_MINUTOS:
                    self.logger.debug(f" ⏰ {symbol} - Breakout detectado recientemente ({tiempo_desde_ultimo:.1f} min), omitiendo...")
                    return None

            tipo_breakout = estrategia.detectar_breakout(symbol, canal_info, datos_mercado)
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
                self.logger.info(f" 🎯 {symbol} - Breakout registrado, esperando reingreso...")

                # Enviar alerta de breakout
                self.enviar_alerta_breakout(symbol, tipo_breakout, canal_info, datos_mercado, config_optima)
                return tipo_breakout

        except Exception as e:
            self.logger.error(f"Error manejando breakout en {symbol}: {e}")
        return None

    def enviar_alerta_breakout(self, symbol: str, tipo_breakout: str, canal_info: CanalInfo, datos_mercado: DatosMercado, config_optima: Dict[str, Any]):
        """
        Envía alerta de breakout detectado
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
                tipo_texto = "RUPTURA de SOPORTE"
                nivel_roto = f"Soporte: {soporte:.8f}"
                direccion_emoji = "⬇️"
                contexto = f"Canal {direccion_canal} → Ruptura de SOPORTE"
                expectativa = "posible entrada en long si el precio reingresa al canal"
            else: # BREAKOUT_SHORT
                emoji_principal = "📉"
                tipo_texto = "RUPTURA BAJISTA de RESISTENCIA"
                nivel_roto = f"Resistencia: {resistencia:.8f}"
                direccion_emoji = "⬆️"
                contexto = f"Canal {direccion_canal} → Rechazo desde RESISTENCIA"
                expectativa = "posible entrada en short si el precio reingresa al canal"

            mensaje = f"""
{emoji_principal} **¡BREAKOUT DETECTADO! - {symbol}**
⚠️ **{tipo_texto}** {direccion_emoji}
⏰ **Hora:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
⏳ **ESPERANDO REINGRESO...**
👁️ Máximo 30 minutos para confirmación
📍 {expectativa}
"""

            # Enviar por Telegram si está configurado
            telegram_token = self._get_config_value('telegram_token', config.telegram_token)
            telegram_chat_ids = self._get_config_value('telegram_chat_ids', config.telegram_chat_ids)
            
            if telegram_token and telegram_chat_ids:
                try:
                    self.logger.debug(f" 📊 Generando gráfico de breakout para {symbol}...")
                    buf = self.generar_grafico_breakout(symbol, canal_info, datos_mercado, tipo_breakout, config_optima)
                    if buf:
                        self.logger.debug(f" 📨 Enviando alerta de breakout por Telegram...")
                        telegram_client.enviar_grafico(telegram_chat_ids, buf)
                        time.sleep(0.5)
                    telegram_client.enviar_mensaje(telegram_chat_ids, mensaje)
                    self.logger.debug(f" ✅ Alerta de breakout enviada para {symbol}")
                except Exception as e:
                    self.logger.error(f" ❌ Error enviando alerta de breakout: {e}")
            else:
                self.logger.debug(f" 📢 Breakout detectado en {symbol} (sin Telegram)")

        except Exception as e:
            self.logger.error(f"Error enviando alerta de breakout: {e}")

    def generar_grafico_breakout(self, symbol: str, canal_info: CanalInfo, datos_mercado: DatosMercado, tipo_breakout: str, config_optima: Dict[str, Any]) -> Optional[BytesIO]:
        """Genera gráfico especial para el momento del breakout"""
        try:
            # Importar aquí para evitar problemas de dependencias
            import matplotlib.pyplot as plt
            import mplfinance as mpf
            import pandas as pd

            plt.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji']

            # Obtener datos frescos
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

            # Calcular líneas del canal
            tiempos_reg = list(range(len(df)))
            resistencia_values = []
            soporte_values = []
            for i, t in enumerate(tiempos_reg):
                resist = canal_info.pendiente_resistencia * t + (canal_info.resistencia - canal_info.pendiente_resistencia * tiempos_reg[-1])
                sop = canal_info.pendiente_soporte * t + (canal_info.soporte - canal_info.pendiente_soporte * tiempos_reg[-1])
                resistencia_values.append(resist)
                soporte_values.append(sop)

            df['Resistencia'] = resistencia_values
            df['Soporte'] = soporte_values

            # Marcar zona de breakout
            precio_breakout = datos_mercado.precio_actual
            breakout_line = [precio_breakout] * len(df)
            color_breakout = "#D68F01"
            titulo_extra = "🚀 RUPTURA ALCISTA" if tipo_breakout == Constants.BREAKOUT_LONG else "📉 RUPTURA BAJISTA"

            # Preparar gráficos adicionales
            apds = [
                mpf.make_addplot(df['Resistencia'], color='#5444ff', linestyle='--', width=2, panel=0),
                mpf.make_addplot(df['Soporte'], color="#5444ff", linestyle='--', width=2, panel=0),
                mpf.make_addplot(breakout_line, color=color_breakout, linestyle='-', width=3, panel=0, alpha=0.8),
            ]

            # Crear gráfico
            fig, axes = mpf.plot(df, type='candle', style='nightclouds',
                                 title=f'{symbol} | {titulo_extra} | {config_optima["timeframe"]} | ⏳ ESPERANDO REENTRY',
                                 ylabel='Precio',
                                 addplot=apds,
                                 volume=False,
                                 returnfig=True,
                                 figsize=(14, 10),
                                 panel_ratios=(3, 1))

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a1a')
            buf.seek(0)
            plt.close(fig)
            return buf

        except Exception as e:
            self.logger.error(f"Error generando gráfico de breakout: {e}")
        return None

    def escanear_mercado(self) -> int:
        """
        Escanea el mercado completo buscando señales
        Returns:
            Número de señales encontradas
        """
        try:
            symbols = self._get_config_value('symbols', config.symbols)
            self.logger.info(f"\n🔍 Escaneando {len(symbols)} símbolos (Estrategia: Breakout + Reentry)...")
            senales_encontradas = 0

            for symbol in symbols:
                try:
                    # Verificar si hay operación activa
                    if symbol in self.operaciones_activas:
                        self.logger.debug(f" ⚡ {symbol} - Operación activa, omitiendo...")
                        continue

                    # Buscar configuración óptima
                    config_optima = self.buscar_configuracion_optima_simbolo(symbol)
                    if not config_optima:
                        self.logger.debug(f" ❌ {symbol} - No se encontró configuración válida")
                        continue

                    # Obtener datos del mercado
                    datos_mercado = estrategia.obtener_datos_mercado(symbol, config_optima['timeframe'], config_optima['num_velas'])
                    if not datos_mercado:
                        self.logger.debug(f" ❌ {symbol} - Error obteniendo datos")
                        continue

                    # Calcular canal de regresión
                    canal_info = estrategia.calcular_canal_regresion(datos_mercado, config_optima['num_velas'])
                    if not canal_info:
                        self.logger.debug(f" ❌ {symbol} - Error calculando canal")
                        continue

                    # Log de estado actual
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

                    # Filtros de calidad
                    if (canal_info.nivel_fuerza < Constants.MIN_NIVEL_FUERZA or
                        abs(canal_info.coeficiente_pearson) < Constants.MIN_PEARSON or
                        canal_info.r2_score < Constants.MIN_R2_SCORE):
                        continue

                    # Manejar breakouts y reentries
                    if symbol not in self.esperando_reentry:
                        # Detectar nuevo breakout
                        tipo_breakout = self.detectar_breakout_y_manejar(symbol, canal_info, datos_mercado, config_optima)
                        if tipo_breakout:
                            continue
                    else:
                        # Verificar reentry
                        breakout_info = self.esperando_reentry[symbol]
                        tipo_operacion = estrategia.detectar_reentry(symbol, canal_info, datos_mercado, breakout_info)
                        if tipo_operacion:
                            # Generar señal de operación
                            precio_entrada, tp, sl = estrategia.calcular_niveles_entrada(tipo_operacion, canal_info, datos_mercado.precio_actual)
                            if precio_entrada and tp and sl:
                                # Verificar cooldown
                                if symbol in self.breakout_history:
                                    ultimo_breakout = self.breakout_history[symbol]
                                    tiempo_desde_ultimo = (datetime.now() - ultimo_breakout).total_seconds() / 3600
                                    if tiempo_desde_ultimo < 2:
                                        self.logger.debug(f" ⏳ {symbol} - Señal reciente, omitiendo...")
                                        continue

                                self.generar_senal_operacion(symbol, tipo_operacion, precio_entrada, tp, sl, canal_info, datos_mercado, config_optima, breakout_info)
                                senales_encontradas += 1
                                self.breakout_history[symbol] = datetime.now()
                                del self.esperando_reentry[symbol]

                except Exception as e:
                    self.logger.error(f"⚠️ Error analizando {symbol}: {e}")
                    continue

            # Log de estado de espera
            if self.esperando_reentry:
                self.logger.info(f"\n⏳ Esperando reingreso en {len(self.esperando_reentry)} símbolos:")
                for symbol, info in self.esperando_reentry.items():
                    tiempo_espera = (datetime.now() - info['timestamp']).total_seconds() / 60
                    self.logger.info(f" • {symbol} - {info['tipo']} - Esperando {tiempo_espera:.1f} min")

            if self.breakouts_detectados:
                self.logger.info(f"\n⏰ Breakouts detectados recientemente:")
                for symbol, info in self.breakouts_detectados.items():
                    tiempo_desde_deteccion = (datetime.now() - info['timestamp']).total_seconds() / 60
                    self.logger.info(f" • {symbol} - {info['tipo']} - Hace {tiempo_desde_deteccion:.1f} min")

            if senales_encontradas > 0:
                self.logger.info(f"✅ Se encontraron {senales_encontradas} señales de trading")
            else:
                self.logger.info("❌ No se encontraron señales en este ciclo")

            return senales_encontradas

        except Exception as e:
            self.logger.error(f"Error en escaneo de mercado: {e}")
        return 0

    def generar_senal_operacion(self, symbol: str, tipo_operacion: str, precio_entrada: float, tp: float, sl: float, canal_info: CanalInfo, datos_mercado: DatosMercado, config_optima: Dict[str, Any], breakout_info: Dict[str, Any] = None):
        """Genera y envía señal de operación"""
        try:
            if symbol in self.senales_enviadas:
                return

            if not all([precio_entrada, tp, sl]):
                self.logger.error(f" ❌ Niveles inválidos para {symbol}, omitiendo señal")
                return

            # Calcular métricas
            riesgo = abs(precio_entrada - sl)
            beneficio = abs(tp - precio_entrada)
            ratio_rr = beneficio / riesgo if riesgo > 0 else 0
            sl_percent = abs((sl - precio_entrada) / precio_entrada) * 100
            tp_percent = abs((tp - precio_entrada) / precio_entrada) * 100
            stoch_estado = "📉 SOBREVENTA" if tipo_operacion == Constants.OPERACION_LONG else "📈 SOBRECOMPRA"

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
                    self.logger.debug(f" 📊 Generando gráfico para {symbol}...")
                    buf = self.generar_grafico_profesional(symbol, canal_info, datos_mercado, precio_entrada, tp, sl, tipo_operacion)
                    if buf:
                        self.logger.debug(f" 📨 Enviando gráfico por Telegram...")
                        telegram_client.enviar_grafico(telegram_chat_ids, buf)
                        time.sleep(1)
                    telegram_client.enviar_mensaje(telegram_chat_ids, mensaje)
                    self.logger.debug(f" ✅ Señal {tipo_operacion} para {symbol} enviada")
                except Exception as e:
                    self.logger.error(f" ❌ Error enviando señal: {e}")

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

        except Exception as e:
            self.logger.error(f"Error generando señal de operación: {e}")

    def generar_grafico_profesional(self, symbol: str, canal_info: CanalInfo, datos_mercado: DatosMercado, precio_entrada: float, tp: float, sl: float, tipo_operacion: str) -> Optional[BytesIO]:
        """Genera gráfico profesional para señales"""
        try:
            import matplotlib.pyplot as plt
            import mplfinance as mpf
            import pandas as pd
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

            df['Resistencia'] = resistencia_values
            df['Soporte'] = soporte_values

            # Niveles de entrada
            entry_line = [precio_entrada] * len(df)
            tp_line = [tp] * len(df)
            sl_line = [sl] * len(df)

            apds = [
                mpf.make_addplot(df['Resistencia'], color='#5444ff', linestyle='--', width=2, panel=0),
                mpf.make_addplot(df['Soporte'], color="#5444ff", linestyle='--', width=2, panel=0),
                mpf.make_addplot(entry_line, color='#FFD700', linestyle='-', width=2, panel=0),
                mpf.make_addplot(tp_line, color='#00FF00', linestyle='-', width=2, panel=0),
                mpf.make_addplot(sl_line, color='#FF0000', linestyle='-', width=2, panel=0),
            ]

            # Crear gráfico
            fig, axes = mpf.plot(df, type='candle', style='nightclouds',
                                 title=f'{symbol} | {tipo_operacion} | {config_optima["timeframe"]} | Breakout+Reentry',
                                 ylabel='Precio',
                                 addplot=apds,
                                 volume=False,
                                 returnfig=True,
                                 figsize=(14, 10),
                                 panel_ratios=(3, 1))

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a1a')
            buf.seek(0)
            plt.close(fig)
            return buf

        except Exception as e:
            self.logger.error(f"Error generando gráfico profesional: {e}")
        return None

    def verificar_cierre_operaciones(self) -> List[str]:
        """Verifica y cierra operaciones que llegaron a TP o SL"""
        try:
            operaciones_cerradas = []
            if not self.operaciones_activas:
                return operaciones_cerradas

            for symbol, operacion in list(self.operaciones_activas.items()):
                try:
                    config_optima = self.config_optima_por_simbolo.get(symbol)
                    if not config_optima:
                        continue

                    # Obtener datos actuales
                    datos_mercado = estrategia.obtener_datos_mercado(symbol, config_optima['timeframe'], config_optima['num_velas'])
                    if not datos_mercado:
                        continue

                    precio_actual = datos_mercado.precio_actual
                    tp = operacion['take_profit']
                    sl = operacion['stop_loss']
                    tipo = operacion['tipo']
                    resultado = None

                    # Verificar condiciones de cierre
                    if tipo == Constants.OPERACION_LONG:
                        if precio_actual >= tp:
                            resultado = Constants.OPERACION_TP
                        elif precio_actual <= sl:
                            resultado = Constants.OPERACION_SL
                    else: # SHORT
                        if precio_actual <= tp:
                            resultado = Constants.OPERACION_TP
                        elif precio_actual >= sl:
                            resultado = Constants.OPERACION_SL

                    if resultado:
                        # Calcular PnL
                        if tipo == Constants.OPERACION_LONG:
                            pnl_percent = ((precio_actual - operacion['precio_entrada']) / operacion['precio_entrada']) * 100
                        else:
                            pnl_percent = ((operacion['precio_entrada'] - precio_actual) / operacion['precio_entrada']) * 100

                        # Calcular duración
                        tiempo_entrada = datetime.fromisoformat(operacion['timestamp_entrada'])
                        duracion_minutos = (datetime.now() - tiempo_entrada).total_seconds() / 60

                        # Crear datos de operación
                        datos_operacion = {
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            'tipo': tipo,
                            'precio_entrada': operacion['precio_entrada'],
                            'take_profit': tp,
                            'stop_loss': sl,
                            'precio_salida': precio_actual,
                            'resultado': resultado,
                            'pnl_percent': pnl_percent,
                            'duracion_minutos': duracion_minutos,
                            'angulo_tendencia': operacion.get('angulo_tendencia', 0),
                            'pearson': operacion.get('pearson', 0),
                            'r2_score': operacion.get('r2_score', 0),
                            'ancho_canal_relativo': operacion.get('ancho_canal_relativo', 0),
                            'ancho_canal_porcentual': operacion.get('ancho_canal_porcentual', 0),
                            'nivel_fuerza': operacion.get('nivel_fuerza', 1),
                            'timeframe_utilizado': operacion.get('timeframe_utilizado', 'N/A'),
                            'velas_utilizadas': operacion.get('velas_utilizadas', 0),
                            'stoch_k': operacion.get('stoch_k', 0),
                            'stoch_d': operacion.get('stoch_d', 0),
                            'breakout_usado': operacion.get('breakout_usado', False)
                        }

                        # Enviar mensaje de cierre
                        mensaje_cierre = self.generar_mensaje_cierre(datos_operacion)
                        telegram_token = self._get_config_value('telegram_token', config.telegram_token)
                        telegram_chat_ids = self._get_config_value('telegram_chat_ids', config.telegram_chat_ids)
                        
                        if telegram_token and telegram_chat_ids:
                            telegram_client.enviar_mensaje(telegram_chat_ids, mensaje_cierre)

                        # Registrar operación
                        self.registrar_operacion(datos_operacion)

                        # Limpiar estado
                        operaciones_cerradas.append(symbol)
                        del self.operaciones_activas[symbol]
                        if symbol in self.senales_enviadas:
                            self.senales_enviadas.remove(symbol)
                        self.operaciones_desde_optimizacion += 1
                        self.logger.info(f" 📊 {symbol} Operación {resultado} - PnL: {pnl_percent:.2f}%")

                except Exception as e:
                    self.logger.error(f"Error verificando operación {symbol}: {e}")
                    continue
            return operaciones_cerradas

        except Exception as e:
            self.logger.error(f"Error verificando cierre de operaciones: {e}")
        return []

    def generar_mensaje_cierre(self, datos_operacion: Dict[str, Any]) -> str:
        """Genera mensaje de cierre de operación"""
        try:
            emoji = "🟢" if datos_operacion['resultado'] == Constants.OPERACION_TP else "🔴"
            color_emoji = "✅" if datos_operacion['resultado'] == Constants.OPERACION_TP else "❌"

            if datos_operacion['tipo'] == Constants.OPERACION_LONG:
                pnl_absoluto = datos_operacion['precio_salida'] - datos_operacion['precio_entrada']
            else:
                pnl_absoluto = datos_operacion['precio_entrada'] - datos_operacion['precio_salida']
            
            breakout_usado = "🚀 Sí" if datos_operacion.get('breakout_usado', False) else "❌ No"

            mensaje = f"""
{emoji} **OPERACIÓN CERRADA - {datos_operacion['symbol']}**
{color_emoji} **RESULTADO: {datos_operacion['resultado']}**
📊 Tipo: {datos_operacion['tipo']}
💰 Entrada: {datos_operacion['precio_entrada']:.8f}
🎯 Salida: {datos_operacion['precio_salida']:.8f}
💵 PnL Absoluto: {pnl_absoluto:.8f}
📈 PnL %: {datos_operacion['pnl_percent']:.2f}%
⏰ Duración: {datos_operacion['duracion_minutos']:.1f} minutos
🚀 Breakout+Reentry: {breakout_usado}
📏 Ángulo: {datos_operacion['angulo_tendencia']:.1f}°
📊 Pearson: {datos_operacion['pearson']:.3f}
🎯 R²: {datos_operacion['r2_score']:.3f}
📏 Ancho: {datos_operacion.get('ancho_canal_porcentual', 0):.1f}%
⏱️ TF: {datos_operacion.get('timeframe_utilizado', 'N/A')}
🕯️ Velas: {datos_operacion.get('velas_utilizadas', 0)}
🕒 {datos_operacion['timestamp']}
"""
            return mensaje

        except Exception as e:
            self.logger.error(f"Error generando mensaje de cierre: {e}")
            return f"❌ Error generando mensaje de cierre: {str(e)}"

    def registrar_operacion(self, datos_operacion: Dict[str, Any]):
        """Registra operación en el archivo de log"""
        try:
            # Crear archivo CSV si no existe
            if not os.path.exists(self.log_path):
                with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Escribir encabezados
                    writer.writerow([
                        'timestamp', 'symbol', 'tipo', 'precio_entrada', 'take_profit', 
                        'stop_loss', 'precio_salida', 'resultado', 'pnl_percent', 
                        'duracion_minutos', 'angulo_tendencia', 'pearson', 'r2_score',
                        'ancho_canal_porcentual', 'nivel_fuerza', 'timeframe_utilizado',
                        'velas_utilizadas', 'stoch_k', 'stoch_d', 'breakout_usado'
                    ])

            # Agregar operación al archivo
            with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datos_operacion['timestamp'],
                    datos_operacion['symbol'],
                    datos_operacion['tipo'],
                    datos_operacion['precio_entrada'],
                    datos_operacion['take_profit'],
                    datos_operacion['stop_loss'],
                    datos_operacion['precio_salida'],
                    datos_operacion['resultado'],
                    datos_operacion['pnl_percent'],
                    datos_operacion['duracion_minutos'],
                    datos_operacion['angulo_tendencia'],
                    datos_operacion['pearson'],
                    datos_operacion['r2_score'],
                    datos_operacion['ancho_canal_porcentual'],
                    datos_operacion['nivel_fuerza'],
                    datos_operacion['timeframe_utilizado'],
                    datos_operacion['velas_utilizadas'],
                    datos_operacion['stoch_k'],
                    datos_operacion['stoch_d'],
                    datos_operacion['breakout_usado']
                ])

        except Exception as e:
            self.logger.error(f"Error registrando operación: {e}")

    def iniciar(self):
        """Inicia el bot de trading"""
        try:
            self.logger.info("🤖 Iniciando bot de trading...")
            
            # Bucle principal del bot
            while True:
                try:
                    # Escanear mercado
                    senales_encontradas = self.escanear_mercado()
                    
                    # Verificar operaciones activas
                    operaciones_cerradas = self.verificar_cierre_operaciones()
                    
                    # Guardar estado
                    self.guardar_estado()
                    
                    # Log de estado
                    if senales_encontradas > 0 or operaciones_cerradas:
                        self.logger.info(f"🔄 Estado: {len(self.operaciones_activas)} activas, {len(self.esperando_reentry)} esperando reentry")
                    
                    # Esperar antes del siguiente ciclo
                    scan_interval = self._get_config_value('scan_interval_minutes', config.scan_interval_minutes)
                    time.sleep(scan_interval * 60)
                    
                except KeyboardInterrupt:
                    self.logger.info("🛑 Bot detenido por el usuario")
                    break
                except Exception as e:
                    self.logger.error(f"Error en ciclo principal: {e}")
                    time.sleep(60)  # Esperar 1 minuto antes de reintentar
                    
        except Exception as e:
            self.logger.error(f"Error iniciando bot: {e}")
            raise
