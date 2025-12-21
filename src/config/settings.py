"""
Configuración y constantes del bot de trading.
Contiene todas las configuraciones cargadas ÚNICAMENTE desde variables de entorno.
CORREGIDO: Sin valores por defecto hardcodeados - Solo variables de entorno de Render
"""
import os
import logging
from typing import List, Dict, Any

# Configurar logger para este módulo
logger = logging.getLogger(__name__)

class ConfigSettings:
    """Configuración centralizada del bot desde variables de entorno ÚNICAMENTE"""

    def __init__(self):
        self.directory_actual = os.path.dirname(os.path.abspath(__file__))
        self._cargar_configuracion_desde_entorno()

    def _cargar_configuracion_desde_entorno(self):
        """Carga toda la configuración DESDE VARIABLES DE ENVIRONMENT DE RENDER"""
        try:
            # Cargar chat IDs de Telegram - REQUERIDO
            telegram_chat_ids_str = os.environ.get('TELEGRAM_CHAT_ID')
            if not telegram_chat_ids_str:
                raise ValueError("❌ TELEGRAM_CHAT_ID es REQUERIDO - Configurar en variables de entorno de Render")
            self.telegram_chat_ids = [cid.strip() for cid in telegram_chat_ids_str.split(',') if cid.strip()]
            
            # Configuración del trading - REQUERIDA
            self.min_channel_width_percent = float(os.environ.get('MIN_CHANNEL_WIDTH_PERCENT'))
            self.trend_threshold_degrees = float(os.environ.get('TREND_THRESHOLD_DEGREES'))
            self.min_trend_strength_degrees = float(os.environ.get('MIN_TREND_STRENGTH_DEGREES'))
            self.entry_margin = float(os.environ.get('ENTRY_MARGIN'))
            self.min_rr_ratio = float(os.environ.get('MIN_RR_RATIO'))
            self.scan_interval_minutes = int(os.environ.get('SCAN_INTERVAL_MINUTES'))

            # Timeframes y velas - REQUERIDOS
            timeframes_env = os.environ.get('TIMEFRAMES')
            if not timeframes_env:
                raise ValueError("❌ TIMEFRAMES es REQUERIDO - Configurar en variables de entorno de Render")
            self.timeframes = [tf.strip() for tf in timeframes_env.split(',')]
            
            velas_env = os.environ.get('VELAS_OPTIONS')
            if not velas_env:
                raise ValueError("❌ VELAS_OPTIONS es REQUERIDO - Configurar en variables de entorno de Render")
            self.velas_options = [int(v) for v in velas_env.split(',')]

            # Símbolos de trading - REQUERIDO
            symbols_env = os.environ.get('SYMBOLS')
            if not symbols_env:
                raise ValueError("❌ SYMBOLS es REQUERIDO - Configurar en variables de entorno de Render")
            self.symbols = [symbol.strip() for symbol in symbols_env.split(',')]

            # Tokens y configuraciones de APIs - REQUERIDOS
            self.telegram_token = os.environ.get('TELEGRAM_TOKEN')
            if not self.telegram_token:
                raise ValueError("❌ TELEGRAM_TOKEN es REQUERIDO - Configurar en variables de entorno de Render")
            
            self.webhook_url = os.environ.get('WEBHOOK_URL')
            if not self.webhook_url:
                raise ValueError("❌ WEBHOOK_URL es REQUERIDO - Configurar en variables de entorno de Render")
            
            self.render_url = os.environ.get('RENDER_EXTERNAL_URL')
            if not self.render_url:
                raise ValueError("❌ RENDER_EXTERNAL_URL es REQUERIDO - Configurar en variables de entorno de Render")

            # Configuración de optimización - REQUERIDA
            self.auto_optimize = os.environ.get('AUTO_OPTIMIZE')
            if not self.auto_optimize:
                raise ValueError("❌ AUTO_OPTIMIZE es REQUERIDO - Configurar en variables de entorno de Render")
            self.auto_optimize = self.auto_optimize.lower() == 'true'
            
            self.min_samples_optimizacion = int(os.environ.get('MIN_SAMPLES_OPTIMIZACION'))
            self.reevaluacion_horas = int(os.environ.get('REEVALUACION_HORAS'))

            # Rutas de archivos
            self.log_path = os.path.join(self.directory_actual, 'operaciones_log_v23.csv')
            self.estado_file = os.path.join(self.directory_actual, 'estado_bot_v23.json')
            self.mejores_parametros_file = 'mejores_parametros.json'
            self.ultimo_reporte_file = 'ultimo_reporte.txt'

            # URLs de APIs - REQUERIDO
            self.binance_api_base = 'https://api.binance.com'
            self.binance_klines_endpoint = '/api/v3/klines'
            self.binance_api_key = os.environ.get('BINANCE_API_KEY')
            if not self.binance_api_key:
                raise ValueError("❌ BINANCE_API_KEY es REQUERIDO - Configurar en variables de entorno de Render")
            
            self.telegram_api_base = f'https://api.telegram.org'

            # Configuración Flask - REQUERIDA
            self.flask_port = int(os.environ.get('PORT'))
            if not self.flask_port:
                raise ValueError("❌ PORT es REQUERIDO - Configurar en variables de entorno de Render")
            
            self.flask_debug = os.environ.get('FLASK_DEBUG')
            if not self.flask_debug:
                raise ValueError("❌ FLASK_DEBUG es REQUERIDO - Configurar en variables de entorno de Render")
            self.flask_debug = self.flask_debug.lower() == 'true'

            logger.info("✅ Configuración cargada correctamente DESDE VARIABLES DE ENTORNO DE RENDER")
            logger.info(f" 📊 Símbolos configurados: {len(self.symbols)}")
            logger.info(f" ⏰ Timeframes configurados: {', '.join(self.timeframes)}")
            logger.info(f" 🕯️ Velas configuradas: {', '.join(map(str, self.velas_options))}")
            logger.info(f" 📱 Telegram configurado: {'✅' if self.telegram_token else '❌'}")
            logger.info(f" 🤖 Auto-optimización: {'✅' if self.auto_optimize else '❌'}")

        except ValueError as e:
            logger.error(f"❌ Error en configuración - Variable de entorno faltante: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error cargando configuración: {e}")
            raise

    def get_config_dict(self) -> Dict[str, Any]:
        """Retorna la configuración como diccionario para compatibilidad"""
        return {
            'min_channel_width_percent': self.min_channel_width_percent,
            'trend_threshold_degrees': self.trend_threshold_degrees,
            'min_trend_strength_degrees': self.min_trend_strength_degrees,
            'entry_margin': self.entry_margin,
            'min_rr_ratio': self.min_rr_ratio,
            'scan_interval_minutes': self.scan_interval_minutes,
            'timeframes': self.timeframes,
            'velas_options': self.velas_options,
            'symbols': self.symbols,
            'telegram_token': self.telegram_token,
            'telegram_chat_ids': self.telegram_chat_ids,
            'auto_optimize': self.auto_optimize,
            'min_samples_optimizacion': self.min_samples_optimizacion,
            'reevaluacion_horas': self.reevaluacion_horas,
            'log_path': self.log_path,
            'estado_file': self.estado_file
        }

# Instancia global de configuración
config = ConfigSettings()

# Constantes del sistema (NO MODIFICADAS)
class Constants:
    """Constantes del sistema que no cambian - LÓGICA DE TRADING INTACTA"""
    # Estados de operación
    OPERACION_TP = "TP"
    OPERACION_SL = "SL"

    # Tipos de señales
    BREAKOUT_LONG = "BREAKOUT_LONG"
    BREAKOUT_SHORT = "BREAKOUT_SHORT"
    OPERACION_LONG = "LONG"
    OPERACION_SHORT = "SHORT"

    # Direcciones de tendencia
    DIRECCION_ALCISTA = "🟢 ALCISTA"
    DIRECCION_BAJISTA = "🔴 BAJISTA"
    DIRECCION_RANGO = "⚪ RANGO"

    # Estados de Stochastic
    STOCH_OVERBOUGHT = 70
    STOCH_OVERSOLD = 30

    # Configuración de optimización
    MIN_MUESTRAS_OPTIMIZACION = 15
    TIMEOUT_REENTRY_MINUTOS = 120
    TIMEOUT_BREAKOUT_MINUTOS = 115
    OPERACIONES_POR_REEVALUACION = 8
    HORAS_REEVALUACION_DEFAULT = 24

    # Configuración de análisis técnico
    PERIOD_STOCHASTIC = 14
    K_PERIOD = 3
    D_PERIOD = 3
    MIN_PEARSON = 0.4
    MIN_R2_SCORE = 0.4
    MIN_NIVEL_FUERZA = 2

    # Configuración de reportes
    DIAS_REPORTE_SEMANAL = 7
    HORA_REPORTE_SEMANAL = 9

    # Configuración de logging (valores directos para evitar dependencias)
    LOG_LEVEL = "INFO"
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Configuración de logging - DEFINIDA DESPUÉS DE Constants PARA EVITAR PROBLEMAS
LOGGING_CONFIG = {
    'level': Constants.LOG_LEVEL,
    'format': Constants.LOG_FORMAT,
    'stream': None # Se configura en el módulo principal
}

logger.info("📋 Configuración y constantes cargadas correctamente")
logger.info("🔒 MODO RENDER: Solo variables de entorno configuradas en Render.com")
