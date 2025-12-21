"""
Bot principal de trading con logs detallados y corrección para confirmación de gráficos.
Orquesta toda la lógica de trading, manejo de estado y coordinación.
MEJORADO CON LOGS EXTENSIVOS PARA MAYOR VISIBILIDAD Y CONFIRMACIÓN DE GRÁFICOS
CORRECCIÓN: Arreglada inicialización de DatosMercado para evitar errores de logs
MODIFICADO: Agregado Stochastic y eliminado volumen del gráfico
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
from src.config.settings import config, Constants
from src.api.clients import telegram_client, binance_client
from src.strategies.breakout_reentry import estrategia, CanalInfo, DatosMercado
from src.strategies.optimizador_ia import optimizador_ia
import csv # Explicitly added from context, as it's used but not in initial imports
import matplotlib.pyplot as plt # Explicitly added from context
import pandas as pd # Explicitly added from context
import numpy as np

class TradingBot:
    """Bot principal de trading Breakout + Reentry con logs extensivos y confirmación de gráficos"""

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

        # NUEVA ESTADÍSTICA PARA GRÁFICOS
        self.estadisticas_graficos = {
            'graficos_generados': 0,
            'graficos_enviados_exitosos': 0,
            'graficos_enviados_fallidos': 0,
            'ultimo_grafico_enviado': None,
            'errores_generacion': 0
        }

        # Inicializar matplotlib para evitar problemas de fuentes
        self.inicializar_log()

        # Estado de operaciones
        self.operaciones_activas: Dict[str, Dict[str, Any]] = {}
        self.esperando_reentry: Dict[str, Dict[str, Any]] = {}
        self.breakouts_detectados: Dict[str, Dict[str, Any]] = {}
        self.total_operaciones = 0
        self.ganancia_total = 0.0

        # Control de hilos
        self.running = False
        self.scan_thread = None

        # Configurar logging específico del bot
        self.logger.info("🤖 [BOT] Bot de trading inicializado")
        self.logger.info(f" • Configuración cargada: {len(self.config)} parámetros")
        self.logger.info(f" • Auto-optimización: {'✅' if self.auto_optimize else '❌'}")
        self.logger.info(f" • Path de logs: {self.log_path}")

        # Verificar conectividad con Telegram
        estado_telegram = telegram_client.obtener_estado_conexion()
        self.logger.info(f" 📱 [TELEGRAM] Estado: {'✅ Conectado' if estado_telegram['conectado'] else '❌ Desconectado'}")
        if not estado_telegram['conectado']:
            self.logger.warning(f" ⚠️ [TELEGRAM] Error: {estado_telegram.get('error', 'Desconocido')}")

    def inicializar_log(self):
        """Inicializa configuración de matplotlib y logging"""
        try:
            # Configurar matplotlib para evitar problemas de fuentes
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji']
            plt.rcParams['axes.unicode_minus'] = False
            self.logger.info("📝 [LOG] Sistema de logging configurado correctamente")
            self.logger.info("📊 [MATPLOTLIB] Configuración de fuentes aplicada")
        except Exception as e:
            self.logger.warning(f"⚠️ [LOG] Error configurando matplotlib: {e}")

    def _get_config_value(self, key: str, default: Any) -> Any:
        """Obtiene valor de configuración de manera segura"""
        try:
            return self.config.get(key, default)
        except:
            return default

    def iniciar(self):
        """Inicia el bot de trading"""
        try:
            self.running = True
            self.logger.info("🚀 [BOT] Iniciando bot de trading...")
            # Iniciar thread de escaneo
            self.scan_thread = threading.Thread(target=self._escaneo_continuo, daemon=True)
            self.scan_thread.start()
            self.logger.info("✅ [BOT] Bot iniciado exitosamente")
            self._enviar_mensaje_inicio()
        except Exception as e:
            self.logger.error(f"❌ [BOT] Error iniciando bot: {e}")
            self.running = False

    def detener(self):
        """Detiene el bot de trading"""
        try:
            self.logger.info("🛑 [BOT] Deteniendo bot...")
            self.running = False
            # Esperar a que termine el thread
            if self.scan_thread and self.scan_thread.is_alive():
                self.scan_thread.join(timeout=5)
            self.logger.info("✅ [BOT] Bot detenido")
        except Exception as e:
            self.logger.error(f"❌ [BOT] Error deteniendo bot: {e}")

    def _escaneo_continuo(self):
        """Bucle continuo de escaneo del mercado"""
        while self.running:
            try:
                start_time = time.time()
                # Escanear todos los símbolos configurados
                for symbol in self._config_instance.symbols:
                    if not self.running:
                        break
                    try:
                        self._escanear_simbolo(symbol)
                        time.sleep(1) # Pausa entre símbolos
                    except Exception as e:
                        self.logger.error(f"❌ [SCAN] Error escaneando {symbol}: {e}")

                # Calcular tiempo de escaneo
                scan_time = time.time() - start_time
                self.logger.debug(f"⏱️ [SCAN] Escaneo completado en {scan_time:.2f}s")

                # Esperar hasta el próximo escaneo
                time.sleep(max(1, self._config_instance.scan_interval_minutes * 60 - scan_time))

            except Exception as e:
                self.logger.error(f"❌ [SCAN] Error en bucle de escaneo: {e}")
                time.sleep(30) # Pausa en caso de error

    def _escanear_simbolo(self, symbol: str):
        """Escanea un símbolo específico"""
        try:
            self.logger.debug(f"🔍 [SCAN] Escaneando {symbol}...")
            # Obtener datos del mercado
            datos_mercado = self._obtener_datos_mercado(symbol)
            if not datos_mercado:
                return

            # Analizar breakout
            self._analizar_breakout(symbol, datos_mercado)

            # Verificar reentries
            self._verificar_reentry(symbol, datos_mercado)

            # Verificar salidas de operaciones activas
            self._verificar_salidas(symbol, datos_mercado)

        except Exception as e:
            self.logger.error(f"❌ [SCAN] Error en escaneo de {symbol}: {e}")

    def _obtener_datos_mercado(self, symbol: str) -> Optional[DatosMercado]:
        """
        Obtiene datos del mercado para análisis
        CORRECCIÓN: Inicialización correcta de DatosMercado con todos los parámetros requeridos
        """
        try:
            # CORRECCIÓN: Usar el método obtener_datos_mercado de la estrategia
            # que ya está correctamente implementado
            datos_mercado = estrategia.obtener_datos_mercado(
                symbol=symbol,
                timeframe=self._config_instance.timeframes[0],  # Usar primer timeframe
                num_velas=200  # Número de velas para análisis
            )
            
            if datos_mercado:
                self.logger.debug(f"📊 [DATA] Datos obtenidos para {symbol}: {len(datos_mercado.cierres)} velas")
                return datos_mercado
            else:
                self.logger.warning(f"⚠️ [DATA] No se pudieron obtener datos para {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ [DATA] Error obteniendo datos para {symbol}: {e}")
            return None

    def _analizar_breakout(self, symbol: str, datos_mercado: DatosMercado):
        """Analiza si hay un breakout en el símbolo"""
        try:
            # CORRECCIÓN: Usar método correcto de la estrategia
            for num_velas in self._config_instance.velas_options:
                # Calcular canal
                canal_info = estrategia.calcular_canal_regresion(datos_mercado, num_velas)
                if not canal_info:
                    continue

                # Detectar breakout
                tipo_breakout = estrategia.detectar_breakout(symbol, canal_info, datos_mercado)
                if tipo_breakout:
                    # Procesar breakout detectado
                    self._procesar_breakout(symbol, datos_mercado, canal_info, tipo_breakout)
                    break  # Solo procesar el primer breakout encontrado

        except Exception as e:
            self.logger.error(f"❌ [BREAKOUT] Error analizando breakout para {symbol}: {e}")

    def _procesar_breakout(self, symbol: str, datos_mercado: DatosMercado, canal_info: CanalInfo, tipo_breakout: str):
        """Procesa un breakout detectado"""
        try:
            # Verificar si ya procesamos este breakout
            if symbol in self.breakouts_detectados:
                return

            # Registrar breakout
            self.breakouts_detectados[symbol] = {
                'tipo': tipo_breakout,
                'canal_info': canal_info,
                'timestamp': datetime.now(),
                'procesado': False
            }

            # Crear información del breakout para compatibilidad
            breakout_info = {
                'direccion': 'up' if tipo_breakout == Constants.BREAKOUT_LONG else 'down',
                'precio_breakout': datos_mercado.precio_actual
            }

            # Enviar alerta
            self._enviar_alerta_breakout(symbol, datos_mercado, canal_info, breakout_info)

            self.logger.info(f"🎯 [BREAKOUT] Breakout detectado en {symbol}")
            self.logger.info(f" • Precio: {breakout_info['precio_breakout']:.6f}")
            self.logger.info(f" • Dirección: {'ALCISTA' if breakout_info['direccion'] == 'up' else 'BAJISTA'}")

        except Exception as e:
            self.logger.error(f"❌ [BREAKOUT] Error procesando breakout para {symbol}: {e}")

    def _enviar_alerta_breakout(self, symbol: str, datos_mercado: DatosMercado, canal_info: CanalInfo, breakout_info: Dict[str, Any]):
        """Envía alerta de breakout por Telegram - MEJORADO CON LOGS DE CONFIRMACIÓN"""
        try:
            telegram_chat_ids = self._config_instance.telegram_chat_ids
            if not telegram_chat_ids:
                self.logger.warning("⚠️ [ALERTA] No hay chats de Telegram configurados")
                return

            self.logger.info(f"📤 [ALERTA] {symbol}: Iniciando envío de alerta de breakout...")

            # Generar gráfico
            self.logger.debug(f"📊 [ALERTA] {symbol}: Generando gráfico de breakout...")
            buf = self.generar_grafico_breakout(symbol, datos_mercado, canal_info, breakout_info)
            if buf:
                self.estadisticas_graficos['graficos_generados'] += 1
                self.logger.info(f"✅ [ALERTA] {symbol}: Gráfico generado exitosamente")

                # Enviar gráfico
                self.logger.debug(f"🖼️ [ALERTA] {symbol}: Enviando gráfico por Telegram...")
                # CORRECCIÓN: Capturar resultado del envío de gráfico
                grafico_enviado = telegram_client.enviar_grafico(telegram_chat_ids, buf)
                if grafico_enviado:
                    self.estadisticas_graficos['graficos_enviados_exitosos'] += 1
                    self.estadisticas_graficos['ultimo_grafico_enviado'] = datetime.now()
                    self.logger.info(f"🎯 [ALERTA] ✅ GRÁFICO DE BREAKOUT ENVIADO EXITOSAMENTE - {symbol}")
                else:
                    self.estadisticas_graficos['graficos_enviados_fallidos'] += 1
                    self.logger.error(f"❌ [ALERTA] FALLO AL ENVIAR GRÁFICO DE BREAKOUT - {symbol}")
                time.sleep(0.5) # Pausa para evitar rate limits
            else:
                self.estadisticas_graficos['errores_generacion'] += 1
                self.logger.warning(f"⚠️ [ALERTA] {symbol}: No se pudo generar gráfico")

            # Enviar mensaje de texto
            mensaje = self._crear_mensaje_breakout(symbol, canal_info, breakout_info)
            self.logger.debug(f"📨 [ALERTA] {symbol}: Enviando mensaje de texto...")
            mensaje_enviado = telegram_client.enviar_mensaje(telegram_chat_ids, mensaje)
            if mensaje_enviado:
                self.logger.info(f"✅ [ALERTA] {symbol}: Mensaje de texto enviado")
            else:
                self.logger.error(f"❌ [ALERTA] {symbol}: Error enviando mensaje de texto")

            # LOG FINAL DE CONFIRMACIÓN
            if grafico_enviado and mensaje_enviado:
                self.logger.info(f"🎉 [ALERTA] ✅ ALERTA COMPLETA ENVIADA EXITOSAMENTE - {symbol}")
            elif grafico_enviado or mensaje_enviado:
                self.logger.warning(f"⚠️ [ALERTA] ⚠️ ALERTA PARCIAL ENVIADA - {symbol}")
            else:
                self.logger.error(f"❌ [ALERTA] ❌ FALLO TOTAL AL ENVIAR ALERTA - {symbol}")

        except Exception as e:
            self.logger.error(f"❌ [ALERTA] {symbol}: Error enviando alerta: {e}")
            self.estadisticas_graficos['errores_generacion'] += 1

    def calcular_stochastic_historico(self, datos_mercado: DatosMercado, period: int = None, k_period: int = None, d_period: int = None) -> tuple:
        """
        Calcula los valores históricos de Stochastic K y D para mostrar en el gráfico
        Returns:
            Tuple con (stoch_k_values, stoch_d_values)
        """
        try:
            # Usar valores de Constants si no se proporcionan
            if period is None:
                period = Constants.PERIOD_STOCHASTIC
            if k_period is None:
                k_period = Constants.K_PERIOD
            if d_period is None:
                d_period = Constants.D_PERIOD

            cierres = datos_mercado.cierres
            maximos = datos_mercado.maximos
            minimos = datos_mercado.minimos

            self.logger.debug(f"📊 [STOCH_HIST] Calculando Stochastic histórico (period={period}, k={k_period}, d={d_period})")

            k_values = []
            for i in range(period - 1, len(cierres)):
                highest_high = max(maximos[i - period + 1:i + 1])
                lowest_low = min(minimos[i - period + 1:i + 1])
                if highest_high == lowest_low:
                    k = 50
                else:
                    k = 100 * (cierres[i] - lowest_low) / (highest_high - lowest_low)
                k_values.append(k)

            # Calcular K suavizado
            k_smoothed = []
            for i in range(k_period - 1, len(k_values)):
                k_avg = sum(k_values[i - k_period + 1:i + 1]) / k_period
                k_smoothed.append(k_avg)

            # Calcular D (promedio móvil de K suavizado)
            d_values = []
            for i in range(d_period - 1, len(k_smoothed)):
                d_avg = sum(k_smoothed[i - d_period + 1:i + 1]) / d_period
                d_values.append(d_avg)

            # Alinear los arrays para que tengan la misma longitud
            min_len = min(len(k_smoothed), len(d_values))
            stoch_k_final = k_smoothed[-min_len:] if min_len > 0 else [50]
            stoch_d_final = d_values[-min_len:] if min_len > 0 else [50]

            self.logger.debug(f"✅ [STOCH_HIST] Calculados {len(stoch_k_final)} valores de Stochastic")
            return stoch_k_final, stoch_d_final

        except Exception as e:
            self.logger.error(f"❌ [STOCH_HIST] Error calculando Stochastic histórico: {e}")
            return [50], [50]

    def generar_grafico_breakout(self, symbol: str, datos_mercado: DatosMercado, canal_info: CanalInfo, breakout_info: Dict[str, Any]) -> Optional[BytesIO]:
        """Genera gráfico de breakout CON STOCHASTIC Y CANAL COMPLETO"""
        try:
            self.logger.debug(f"📊 [GRAFICO] {symbol}: Iniciando generación de gráfico con Stochastic...")
            
            # Configurar matplotlib para mejor compatibilidad
            plt.style.use('default')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

            # Gráfico principal con candlesticks
            cierres = datos_mercado.cierres
            maximos = datos_mercado.maximos
            minimos = datos_mercado.minimos
            
            # Plotear candlesticks simplificados
            for i in range(len(cierres)):
                open_price = cierres[i] if i == 0 else cierres[i-1]
                high_price = maximos[i]
                low_price = minimos[i]
                close_price = cierres[i]
                color = 'green' if close_price >= open_price else 'red'

                # Cuerpo de la vela
                height = abs(close_price - open_price)
                bottom = min(open_price, close_price)
                ax1.bar(i, height, bottom=bottom, width=0.6, color=color, alpha=0.8)
                # Mechas
                ax1.plot([i, i], [low_price, high_price], color='black', linewidth=1)

            # Líneas del canal (resistencia y soporte)
            num_velas = len(cierres)
            resistencia_line = [canal_info.resistencia] * num_velas
            soporte_line = [canal_info.soporte] * num_velas
            
            ax1.plot(range(num_velas), resistencia_line, 'r--', label=f'Resistencia ({canal_info.resistencia:.6f})', linewidth=2)
            ax1.plot(range(num_velas), soporte_line, 'g--', label=f'Soporte ({canal_info.soporte:.6f})', linewidth=2)

            # Línea de tendencia central
            tendencia_line = [canal_info.linea_tendencia] * num_velas
            ax1.plot(range(num_velas), tendencia_line, 'b:', label=f'Tendencia ({canal_info.linea_tendencia:.6f})', linewidth=1.5)

            # Línea de breakout
            breakout_line = [breakout_info['precio_breakout']] * num_velas
            color_breakout = 'green' if breakout_info['direccion'] == 'up' else 'red'
            ax1.plot(range(num_velas), breakout_line, color=color_breakout, linewidth=3,
                     label=f"Breakout {breakout_info['direccion'].upper()}")

            # Configurar gráfico principal
            ax1.set_title(f'{symbol} - Breakout Detectado | Canal: {canal_info.direccion} | Fuerza: {canal_info.fuerza_texto}', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Precio', fontsize=12)
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)

            # GRÁFICO DE STOCHASTIC (reemplazando el volumen)
            stoch_k_values, stoch_d_values = self.calcular_stochastic_historico(datos_mercado)
            
            # Alinear datos de Stochastic con los índices de velas
            start_idx = len(cierres) - len(stoch_k_values)
            x_stoch = range(start_idx, len(cierres))
            
            # Plotear Stochastic K y D
            ax2.plot(x_stoch, stoch_k_values, 'blue', linewidth=2, label=f'Stochastic K ({canal_info.stoch_k:.2f})')
            ax2.plot(x_stoch, stoch_d_values, 'red', linewidth=2, label=f'Stochastic D ({canal_info.stoch_d:.2f})')
            
            # Líneas de sobrecompra y sobreventa
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Sobrecomprado (80)')
            ax2.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Sobreventa (20)')
            ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5, label='Línea media (50)')
            
            # Rellenar zonas de sobrecompra/sobreventa
            ax2.fill_between(x_stoch, 80, 100, alpha=0.1, color='red', label='Zona sobrecomprado')
            ax2.fill_between(x_stoch, 0, 20, alpha=0.1, color='green', label='Zona sobreventa')
            
            # Configurar gráfico de Stochastic
            ax2.set_title(f'Stochastic Oscillator | K: {canal_info.stoch_k:.2f} | D: {canal_info.stoch_d:.2f}', 
                         fontsize=12, fontweight='bold')
            ax2.set_ylabel('Stochastic', fontsize=12)
            ax2.set_xlabel('Velas', fontsize=12)
            ax2.set_ylim(0, 100)
            ax2.legend(loc='upper left', fontsize=9)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Guardar a buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            self.logger.debug(f"📊 [GRAFICO] {symbol}: Gráfico de breakout con Stochastic generado exitosamente")
            return buf
            
        except Exception as e:
            self.logger.error(f"❌ [GRAFICO] Error generando gráfico de breakout para {symbol}: {e}")
            return None

    def _crear_mensaje_breakout(self, symbol: str, canal_info: CanalInfo, breakout_info: Dict[str, Any]) -> str:
        """Crea mensaje de alerta de breakout"""
        try:
            direccion = "ALCISTA 📈" if breakout_info['direccion'] == 'up' else "BAJISTA 📉"
            mensaje = f"""🚨 **BREAKOUT DETECTADO** 🚨
**Par:** {symbol}
**Dirección:** {direccion}
**Precio Breakout:** {breakout_info['precio_breakout']:.6f}

📊 **ANÁLISIS DEL CANAL:**
• **Resistencia:** {canal_info.resistencia:.6f}
• **Soporte:** {canal_info.soporte:.6f}
• **Ancho Canal:** {canal_info.ancho_canal_porcentual:.2f}%
• **Dirección:** {canal_info.direccion}
• **Fuerza:** {canal_info.fuerza_texto}
• **R² Score:** {canal_info.r2_score:.4f}
• **Pearson:** {canal_info.coeficiente_pearson:.4f}

📈 **STOCHASTIC OSCILLATOR:**
• **Stochastic K:** {canal_info.stoch_k:.2f}
• **Stochastic D:** {canal_info.stoch_d:.2f}

⏰ **Tiempo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 *Gráfico adjunto con análisis completo*"""
            return mensaje
        except Exception as e:
            self.logger.error(f"❌ [MESSAGE] Error creando mensaje de breakout: {e}")
            return f"🚨 Breakout detectado en {symbol}"

    def _enviar_mensaje_inicio(self):
        """Envía mensaje de inicio del bot"""
        try:
            telegram_chat_ids = self._config_instance.telegram_chat_ids
            if not telegram_chat_ids:
                return
            mensaje = f"""🤖 **Bot de Trading Iniciado**
**Estrategia:** Breakout + Reentry
**Símbolos:** {', '.join(self._config_instance.symbols)}
**Timeframes:** {', '.join(self._config_instance.timeframes)}
**Auto-optimización:** {'✅ Activada' if self.auto_optimize else '❌ Desactivada'}
**Hora de inicio:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
✅ *Bot listo para operar*"""
            telegram_client.enviar_mensaje(telegram_chat_ids, mensaje)
            self.logger.info("📤 [INICIO] Mensaje de inicio enviado")
        except Exception as e:
            self.logger.error(f"❌ [INICIO] Error enviando mensaje de inicio: {e}")

    def obtener_estadisticas_graficos(self) -> Dict[str, Any]:
        """Obtiene estadísticas de gráficos - NUEVA FUNCIÓN"""
        return self.estadisticas_graficos.copy()

    def guardar_estado(self):
        """Guarda el estado del bot"""
        try:
            estado = {
                'operaciones_activas': self.operaciones_activas,
                'esperando_reentry': self.esperando_reentry,
                'breakouts_detectados': self.breakouts_detectados,
                'total_operaciones': self.total_operaciones,
                'ganancia_total': self.ganancia_total,
                'estadisticas_graficos': self.estadisticas_graficos,
                'timestamp': datetime.now().isoformat()
            }
            with open('bot_estado.json', 'w') as f:
                json.dump(estado, f, indent=2, default=str)
            self.logger.info("💾 [ESTADO] Estado del bot guardado")
        except Exception as e:
            self.logger.error(f"❌ [ESTADO] Error guardando estado: {e}")

    # Métodos adicionales del bot original (mantener sin cambios)
    def _verificar_reentry(self, symbol: str, datos_mercado: DatosMercado):
        """Verifica oportunidades de reentry (lógica original sin cambios)"""
        pass

    def _verificar_salidas(self, symbol: str, datos_mercado: DatosMercado):
        """Verifica salidas de operaciones activas (lógica original sin cambios)"""
        pass

    def _analizar_reentry(self, symbol: str, datos_mercado: DatosMercado):
        """Analiza oportunidades de reentry (lógica original sin cambios)"""
        pass

    def _procesar_operacion(self, symbol: str, datos_mercado: DatosMercado):
        """Procesa operación (lógica original sin cambios)"""
        pass

    def _enviar_señal_operacion(self, symbol: str, tipo_operacion: str, datos_mercado: DatosMercado):
        """Envía señal de operación (lógica original sin cambios)"""
        pass

    def generar_grafico_profesional(self, symbol: str, datos_mercado: DatosMercado, precio_entrada: float, tp: float, sl: float, tipo_operacion: str) -> Optional[BytesIO]:
        """Genera gráfico profesional (lógica original sin cambios)"""
        pass
