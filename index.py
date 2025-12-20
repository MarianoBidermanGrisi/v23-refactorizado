"""
Archivo principal del bot de trading.
Punto de entrada que orquesta todos los módulos y componentes.
"""

import os
import sys
import signal
import threading
import time
from datetime import datetime

# Agregar el directorio src al path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar configuración primero
from src.config.settings import config

# Configurar logging
from src.utils.logging_config import configurar_logging, obtener_logger
logger = configurar_logging()

# Importar componentes principales
from src.bot.trading_bot import TradingBot
from src.utils.utilidades import utilidades_bot
from src.api.clients import render_health_client

class BotOrchestrator:
    """Orquestador principal del bot"""
    
    def __init__(self):
        self.logger = obtener_logger(__name__)
        self.bot_instance = None
        self.running = False
        self.bot_thread = None
        
    def inicializar_sistema(self):
        """Inicializa todo el sistema"""
        try:
            self.logger.info("🚀 INICIANDO SISTEMA DE TRADING")
            self.logger.info("=" * 60)
            
            # 1. Validar configuración
            self.logger.info("1️⃣ Validando configuración...")
            validacion = utilidades_bot.validar_configuracion()
            
            if not validacion['valida']:
                self.logger.error("❌ Configuración inválida:")
                for error in validacion['errores']:
                    self.logger.error(f"   • {error}")
                raise Exception("Configuración inválida")
            
            for advertencia in validacion['advertencias']:
                self.logger.warning(f"   ⚠️ {advertencia}")
            
            self.logger.info("✅ Configuración válida")
            
            # 2. Verificar integridad de datos
            self.logger.info("2️⃣ Verificando integridad de datos...")
            integridad = utilidades_bot.verificar_integridad_datos()
            
            if integridad.get('errores'):
                self.logger.error("❌ Errores encontrados:")
                for error in integridad['errores']:
                    self.logger.error(f"   • {error}")
                # No fallar por errores de integridad, solo advertir
            
            self.logger.info("✅ Integridad de datos verificada")
            
            # 3. Crear instancia del bot
            self.logger.info("3️⃣ Inicializando bot de trading...")
            config_dict = config.get_config_dict()
            self.bot_instance = TradingBot(config_dict)
            self.logger.info("✅ Bot inicializado correctamente")
            
            # 4. Configurar manejo de señales
            self._configurar_manejo_senales()
            
            self.logger.info("✅ Sistema inicializado correctamente")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error inicializando sistema: {e}")
            return False
    
    def _configurar_manejo_senales(self):
        """Configura el manejo de señales del sistema"""
        def signal_handler(signum, frame):
            self.logger.info(f"\n🛑 Señal {signum} recibida, cerrando sistema...")
            self.detener_sistema()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        self.logger.info("✅ Manejo de señales configurado")
    
    def iniciar_bot_background(self):
        """Inicia el bot en un hilo separado"""
        try:
            if self.running:
                self.logger.warning("⚠️ Bot ya está ejecutándose")
                return
            
            self.running = True
            
            def run_bot():
                """Función que ejecuta el bot en background"""
                try:
                    self.logger.info("🤖 Iniciando bot en hilo separado...")
                    self.bot_instance.iniciar()
                except Exception as e:
                    self.logger.error(f"❌ Error en bot background: {e}")
                    self.running = False
            
            self.bot_thread = threading.Thread(target=run_bot, daemon=True)
            self.bot_thread.start()
            
            self.logger.info("✅ Bot iniciado en background")
            
        except Exception as e:
            self.logger.error(f"❌ Error iniciando bot: {e}")
            self.running = False
    
    def detener_sistema(self):
        """Detiene todo el sistema ordenadamente"""
        try:
            self.logger.info("🛑 Deteniendo sistema...")
            
            self.running = False
            
            # Guardar estado del bot
            if self.bot_instance:
                self.logger.info("💾 Guardando estado del bot...")
                self.bot_instance.guardar_estado()
            
            # Crear backup
            self.logger.info("💾 Creando backup...")
            utilidades_bot.crear_backup_estado()
            
            self.logger.info("👋 Sistema detenido correctamente")
            
        except Exception as e:
            self.logger.error(f"❌ Error deteniendo sistema: {e}")
    
    def obtener_estado_sistema(self) -> dict:
        """Obtiene el estado actual del sistema"""
        try:
            estado = {
                'timestamp': datetime.now().isoformat(),
                'running': self.running,
                'bot_activo': bool(self.bot_instance and self.bot_instance.operaciones_activas),
                'health': render_health_client.verificar_health(),
                'estadisticas': utilidades_bot.generar_estadisticas_sistema(),
                'configuracion': config.get_config_dict()
            }
            
            if self.bot_instance:
                estado['bot'] = {
                    'operaciones_activas': len(self.bot_instance.operaciones_activas),
                    'esperando_reentry': len(self.bot_instance.esperando_reentry),
                    'total_operaciones': self.bot_instance.total_operaciones,
                    'breakouts_detectados': len(self.bot_instance.breakouts_detectados)
                }
            
            return estado
            
        except Exception as e:
            self.logger.error(f"Error obteniendo estado: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

# Instancia global del orquestador
orchestrator = BotOrchestrator()

def main():
    """Función principal"""
    try:
        # Inicializar sistema
        if not orchestrator.inicializar_sistema():
            sys.exit(1)
        
        # Iniciar bot
        orchestrator.iniciar_bot_background()
        
        # Mantener el programa ejecutándose
        self.logger.info("🎯 Sistema listo y ejecutándose...")
        self.logger.info("📊 Para ver estado: curl http://localhost:5000/health")
        self.logger.info("🛑 Para detener: Ctrl+C")
        
        # Bucle principal de monitoreo
        while orchestrator.running:
            try:
                time.sleep(30)  # Verificar cada 30 segundos
                
                # Verificar que el bot sigue vivo
                if not orchestrator.bot_thread or not orchestrator.bot_thread.is_alive():
                    self.logger.warning("⚠️ Bot thread no está activo, reiniciando...")
                    orchestrator.iniciar_bot_background()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Error en bucle principal: {e}")
                time.sleep(10)
        
    except Exception as e:
        self.logger.error(f"❌ Error fatal en main: {e}")
        sys.exit(1)
    finally:
        orchestrator.detener_sistema()

if __name__ == "__main__":
    main()
