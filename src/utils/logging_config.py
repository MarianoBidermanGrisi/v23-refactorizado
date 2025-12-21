"""
Configuración de logging para el bot.
Configura los logs del sistema de manera centralizada.
CORRECCIÓN: Sin cambios en lógica de trading, solo mejoras en configuración de logs
"""
import logging
import sys
from datetime import datetime
from ..config.settings import Constants, LOGGING_CONFIG

def configurar_logging():
    """Configura el sistema de logging del bot - SIN DUPLICACIÓN"""
    try:
        # Usar LOGGING_CONFIG importado para evitar problemas de dependencias
        log_format = LOGGING_CONFIG['format']
        log_level = LOGGING_CONFIG['level']

        # Configurar formato
        formatter = logging.Formatter(
            fmt=log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Configurar nivel
        level = getattr(logging, log_level.upper(), logging.INFO)

        # CORRECCIÓN: Solo un handler para consola para evitar duplicación
        console_handler = logging.StreamHandler(sys.stdout)

        # Handler para archivo con rotación
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler('bot_trading.log', maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
        except Exception as e:
            # Usar sys.stderr directamente ya que el logging aún no está configurado
            sys.stderr.write(f"⚠️ No se pudo crear handler de archivo: {e}\n")
            file_handler = None

        # Configurar logger principal
        logger = logging.getLogger()
        logger.setLevel(level)

        # CORRECCIÓN: Limpiar TODOS los handlers existentes primero
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # CORRECCIÓN: Solo agregar handlers únicos (SIN duplicación)
        logger.addHandler(console_handler)
        if file_handler:
            logger.addHandler(file_handler)

        # Configurar loggers específicos
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)

        # Usar el logger configurado
        logger.info("📝 Sistema de logging configurado correctamente")
        return logger

    except Exception as e:
        # Usar stderr directamente ya que el logging aún no está configurado
        sys.stderr.write(f"❌ Error configurando logging: {e}\n")
        # Fallback a configuración básica con valores directos
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger()

def obtener_logger(nombre: str = None) -> logging.Logger:
    """
    Obtiene un logger configurado
    Args:
        nombre: Nombre del logger (opcional)
    Returns:
        Logger configurado
    """
    if nombre:
        return logging.getLogger(nombre)
    else:
        return logging.getLogger(__name__)

# Configurar logging al importar - CORREGIDO para evitar duplicación
try:
    logger_base = configurar_logging()
    logger_base.info("🔧 Logging configurado")
except Exception as e: # CORRECCIÓN: Error tipográfico corregido
    # Usar stderr directamente
    sys.stderr.write(f"⚠️ Error configurando logging durante importación: {e}\n")
    # Configuración de emergencia sin Constants
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger_base = logging.getLogger()
    logger_base.info("🔧 Logging configurado (modo emergencia)")
