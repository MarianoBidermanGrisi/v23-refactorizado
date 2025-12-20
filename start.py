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
from datetime import datetime

# Configurar path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Función principal de inicio"""
    try:
        print("🚀 Iniciando Bot de Trading Breakout + Reentry")
        print("=" * 60)
        
        # Importar y ejecutar aplicación Flask
        from flask_app import app, set_orchestrator_instance
        
        # Importar orquestador
        from index import orchestrator
        
        # Configurar referencia al orquestador
        set_orchestrator_instance(orchestrator)
        
        # Inicializar sistema
        print("🔧 Inicializando sistema...")
        if not orchestrator.inicializar_sistema():
            print("❌ Error inicializando sistema")
            return False
        
        # Iniciar bot en background
        print("🤖 Iniciando bot en background...")
        orchestrator.iniciar_bot_background()
        
        # Configurar manejo de señales
        def signal_handler(signum, frame):
            print(f"\n🛑 Señal {signum} recibida...")
            orchestrator.detener_sistema()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("✅ Sistema iniciado correctamente")
        print("📡 Iniciando servidor web...")
        
        # Iniciar Flask en el hilo principal
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo sistema...")
        if 'orchestrator' in locals():
            orchestrator.detener_sistema()
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        return False
    finally:
        print("👋 Sistema finalizado")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)