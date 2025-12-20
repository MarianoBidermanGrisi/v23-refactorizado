"""
Configuración de logging para el bot.
Configura los logs del sistema de manera centralizada.
"""

import logging
import sys
from datetime import datetime

from ..config.settings import Constants, LOGGING_CONFIG

def configurar_logging():
    """Configura el sistema de logging del bot"""
    try:
        # Configurar formato
        formatter = logging.Formatter(
            fmt=Constants.LOG_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Configurar nivel
        level = getattr(logging, Constants.LOG_LEVEL.upper(), logging.INFO)
        
        # Handler para consola (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # Handler para archivo (opcional)
        try:
            file_handler = logging.FileHandler('bot_trading.log', encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
        except Exception as e:
            print(f"⚠️ No se pudo crear handler de archivo: {e}")
            file_handler = None
        
        # Configurar logger principal
        logger = logging.getLogger()
        logger.setLevel(level)
        
        # Limpiar handlers existentes
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Agregar handlers
        logger.addHandler(console_handler)
        if file_handler:
            logger.addHandler(file_handler)
        
        # Configurar loggers específicos
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        
        print("📝 Sistema de logging configurado correctamente")
        
        return logger
        
    except Exception as e:
        print(f"❌ Error configurando logging: {e}")
        # Fallback a configuración básica
        logging.basicConfig(level=logging.INFO, format=Constants.LOG_FORMAT)
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

# Configurar logging al importar
logger_base = configurar_logging()

print("🔧 Logging configurado")
