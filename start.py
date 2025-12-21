#!/usr/bin/env python3
"""
Script de inicio simplificado para Render.com
Ejecuta el bot de trading con configuración automática.
"""

import os
import sys
import time
import signal
import threading
import logging
from datetime import datetime

# Configurar path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def configurar_logging_basico():
    """Configura logging básico para start.py"""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('start.log', encoding='utf-8')
            ]
        )
        return logging.getLogger(__name__)
    except Exception:
        # Fallback si no se puede configurar
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

def main():
    """Función principal de inicio"""
    logger = configurar_logging_basico()
    
    try:
        logger.info("🚀 Iniciando Bot de Trading Breakout + Reentry")
        logger.info("=" * 60)
        
        # Importar y ejecutar aplicación Flask
        from flask_app import app, set_orchestrator_instance
        
        # Importar orquestador
        from index import orchestrator
        
        # Configurar referencia al orquestador
        set_orchestrator_instance(orchestrator)
        
        # Inicializar sistema
        logger.info("🔧 Inicializando sistema...")
        if not orchestrator.inicializar_sistema():
            logger.error("❌ Error inicializando sistema")
            return False
        
        # Iniciar bot en background
        logger.info("🤖 Iniciando bot en background...")
        orchestrator.iniciar_bot_background()
        
        # Configurar manejo de señales
        def signal_handler(signum, frame):
            logger.info(f"\n🛑 Señal {signum} recibida...")
            orchestrator.detener_sistema()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("✅ Sistema iniciado correctamente")
        logger.info("📡 Iniciando servidor web...")
        
        # Iniciar Flask en el hilo principal
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Deteniendo sistema...")
        if 'orchestrator' in locals():
            orchestrator.detener_sistema()
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}", exc_info=True)
        return False
    finally:
        logger.info("👋 Sistema finalizado")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
