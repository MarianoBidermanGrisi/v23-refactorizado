"""
Configuración y constantes del bot de trading.
Contiene todas las configuraciones cargadas desde variables de entorno.
"""
import os
from typing import List, Dict, Any

class ConfigSettings:
    """Configuración centralizada del bot desde variables de entorno"""
    
    def __init__(self):
        self.directory_actual = os.path.dirname(os.path.abspath(__file__))
        self._cargar_configuracion_desde_entorno()
    
    def _cargar_configuracion_desde_entorno(self):
        """Carga toda la configuración desde variables de entorno"""
        try:
            # Cargar chat IDs de Telegram
            telegram_chat_ids_str = os.environ.get('TELEGRAM_CHAT_ID', '-1002272872445')
            self.telegram_chat_ids = [cid.strip() for cid in telegram_chat_ids_str.split(',') if cid.strip()]
            
            # Configuración del trading
            self.min_channel_width_percent = float(os.environ.get('MIN_CHANNEL_WIDTH_PERCENT', '4.0'))
            self.trend_threshold_degrees = float(os.environ.get('TREND_THRESHOLD_DEGREES', '16.0'))
            self.min_trend_strength_degrees = float(os.environ.get('MIN_TREND_STRENGTH_DEGREES', '16.0'))
            self.entry_margin = float(os.environ.get('ENTRY_MARGIN', '0.001'))
            self.min_rr_ratio = float(os.environ.get('MIN_RR_RATIO', '1.2'))
            self.scan_interval_minutes = int(os.environ.get('SCAN_INTERVAL_MINUTES', '1'))
            
            # Timeframes y velas
            timeframes_env = os.environ.get('TIMEFRAMES', '5m,15m,30m,1h,4h')
            self.timeframes = [tf.strip() for tf in timeframes_env.split(',')]
            
            velas_env = os.environ.get('VELAS_OPTIONS', '80,100,120,150,200')
            self.velas_options = [int(v) for v in velas_env.split(',')]
            
            # Símbolos de trading
            symbols_env = os.environ.get('SYMBOLS', 
                'BTCUSDT,ETHUSDT,DOTUSDT,LINKUSDT,BNBUSDT,XRPUSDT,SOLUSDT,AVAXUSDT,DOGEUSDT,LTCUSDT,ATOMUSDT,XLMUSDT,ALGOUSDT,VETUSDT,ICPUSDT,FILUSDT,BCHUSDT,EOSUSDT,TRXUSDT,XTZUSDT,SUSHIUSDT,COMPUSDT,YFIUSDT,ETCUSDT,SNXUSDT,RENUSDT,1INCHUSDT,NEOUSDT,ZILUSDT,HOTUSDT,ENJUSDT,ZECUSDT'
            )
            self.symbols = [symbol.strip() for symbol in symbols_env.split(',')]
            
            # Tokens y configuraciones de APIs
            self.telegram_token = os.environ.get('TELEGRAM_TOKEN')
            self.webhook_url = os.environ.get('WEBHOOK_URL')
            self.render_url = os.environ.get('RENDER_EXTERNAL_URL')
            
            # Configuración de optimización
            self.auto_optimize = os.environ.get('AUTO_OPTIMIZE', 'true').lower() == 'true'
            self.min_samples_optimizacion = int(os.environ.get('MIN_SAMPLES_OPTIMIZACION', '30'))
            self.reevaluacion_horas = int(os.environ.get('REEVALUACION_HORAS', '24'))
            
            # Rutas de archivos
            self.log_path = os.path.join(self.directory_actual, 'operaciones_log_v23.csv')
            self.estado_file = os.path.join(self.directory_actual, 'estado_bot_v23.json')
            self.mejores_parametros_file = 'mejores_parametros.json'
            self.ultimo_reporte_file = 'ultimo_reporte.txt'
            
            # URLs de APIs
            self.binance_api_base = 'https://api.binance.com'
            self.binance_klines_endpoint = '/api/v3/klines'
            self.telegram_api_base = f'https://api.telegram.org'
            
            # Configuración Flask
            self.flask_port = int(os.environ.get('PORT', '5000'))
            self.flask_debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
            
            print("✅ Configuración cargada correctamente desde variables de entorno")
            print(f"   📊 Símbolos: {len(self.symbols)}")
            print(f"   ⏰ Timeframes: {', '.join(self.timeframes)}")
            print(f"   🕯️ Velas: {', '.join(map(str, self.velas_options))}")
            
        except Exception as e:
            print(f"❌ Error cargando configuración: {e}")
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

# Constantes del sistema
class Constants:
    """Constantes del sistema que no cambian"""
    
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
    'level': 'INFO',  # Usamos valor directo en lugar de Constants.LOG_LEVEL
    'format': '%(asctime)s - %(levelname)s - %(message)s',  # Valor directo en lugar de Constants.LOG_FORMAT
    'stream': None  # Se configura en el módulo principal
}

print("📋 Configuración y constantes cargadas correctamente")
