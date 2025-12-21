"""
Aplicación Flask para Render.com
Maneja health checks, webhooks y endpoints de monitoreo.
CORRECCIÓN: Logging mejorado sin cambios en lógica de trading
"""
import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from threading import Thread

# Agregar el directorio src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar componentes
from src.config.settings import config
from src.utils.logging_config import obtener_logger
from src.api.clients import telegram_client, render_health_client
from src.utils.utilidades import utilidades_bot

# Configurar logging
logger = obtener_logger(__name__)

# Crear aplicación Flask
app = Flask(__name__)

# Instancia global del orquestador (se inicializará en el thread principal)
orchestrator = None

def set_orchestrator_instance(orchestrator_instance):
    """Configura la instancia del orquestador desde el módulo principal"""
    global orchestrator
    orchestrator = orchestrator_instance

@app.route('/')
def index():
    """Endpoint principal"""
    try:
        estado = "🤖 Bot Breakout + Reentry está en línea" if orchestrator and orchestrator.running else "⚠️ Bot en mantenimiento"
        info_sistema = {
            'status': 'ok' if orchestrator and orchestrator.running else 'maintenance',
            'timestamp': datetime.now().isoformat(),
            'bot': {
                'activo': bool(orchestrator and orchestrator.running),
                'operaciones_activas': len(orchestrator.bot_instance.operaciones_activas) if orchestrator and orchestrator.bot_instance else 0,
                'esperando_reentry': len(orchestrator.bot_instance.esperando_reentry) if orchestrator and orchestrator.bot_instance else 0,
                'total_operaciones': orchestrator.bot_instance.total_operaciones if orchestrator and orchestrator.bot_instance else 0
            } if orchestrator and orchestrator.bot_instance else {},
            'version': '2.3.0',
            'estrategia': 'Breakout + Reentry'
        }
        return jsonify(info_sistema), 200 if orchestrator and orchestrator.running else 503
    except Exception as e:
        logger.error(f"Error en endpoint principal: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health')
def health_check():
    """Endpoint de health check para Render"""
    try:
        # Verificar salud del sistema
        health_status = render_health_client.verificar_health()

        # Agregar información del bot si está disponible
        if orchestrator and orchestrator.running:
            health_status['bot_status'] = 'running'
            health_status['operaciones_activas'] = len(orchestrator.bot_instance.operaciones_activas)
            health_status['esperando_reentry'] = len(orchestrator.bot_instance.esperando_reentry)
        else:
            health_status['bot_status'] = 'stopped'

        # Status code basado en salud general
        status_code = 200 if health_status.get('status') == 'ok' else 503
        return jsonify(health_status), status_code
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/status')
def status_completo():
    """Endpoint de status completo del sistema"""
    try:
        if not orchestrator:
            return jsonify({
                'status': 'error',
                'message': 'Orquestador no inicializado',
                'timestamp': datetime.now().isoformat()
            }), 503

        # Obtener estado completo
        estado = orchestrator.obtener_estado_sistema()
        return jsonify(estado), 200
    except Exception as e:
        logger.error(f"Error en status: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    """Webhook para recibir actualizaciones de Telegram"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        update = request.get_json()
        logger.debug(f"Update recibido: {json.dumps(update)}")
        # Procesar actualización (aquí se pueden agregar comandos)
        # Por ahora solo confirmamos recepción
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Error procesando webhook: {e}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/stats', methods=['GET'])
def estadisticas_api():
    """API para obtener estadísticas del sistema"""
    try:
        estadisticas = utilidades_bot.generar_estadisticas_sistema()
        return jsonify(estadisticas), 200
    except Exception as e:
        logger.error(f"Error en estadísticas API: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/config', methods=['GET'])
def configuracion_api():
    """API para obtener configuración (sin datos sensibles)"""
    try:
        config_limpia = {
            'symbols_count': len(config.symbols),
            'timeframes': config.timeframes,
            'auto_optimize': config.auto_optimize,
            'scan_interval_minutes': config.scan_interval_minutes,
            'min_channel_width_percent': config.min_channel_width_percent,
            'telegram_configurado': bool(config.telegram_token),
            'telegram_chats_count': len(config.telegram_chat_ids)
        }
        return jsonify(config_limpia), 200
    except Exception as e:
        logger.error(f"Error en configuración API: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/control', methods=['POST'])
def control_bot():
    """API para controlar el bot (restart, stop, etc.)"""
    try:
        data = request.get_json() or {}
        accion = data.get('accion', '').lower()

        if not orchestrator:
            return jsonify({
                'status': 'error',
                'message': 'Orquestador no inicializado'
            }), 503

        resultado = {}
        if accion == 'restart':
            if orchestrator.running:
                orchestrator.detener_sistema()
                time.sleep(2)  # Pausa para que el sistema se detenga completamente
            orchestrator.inicializar_sistema()
            orchestrator.iniciar_bot_background()
            resultado['message'] = 'Bot reiniciado'
        elif accion == 'stop':
            orchestrator.detener_sistema()
            resultado['message'] = 'Bot detenido'
        elif accion == 'status':
            resultado = orchestrator.obtener_estado_sistema()
        else:
            return jsonify({
                'status': 'error',
                'message': 'Acción no válida. Use: restart, stop, status'
            }), 400

        return jsonify({
            'status': 'ok',
            'resultado': resultado,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error en control API: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Manejador para rutas no encontradas"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint no encontrado',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Manejador para errores internos"""
    logger.error(f"Error interno: {error}")
    return jsonify({
        'status': 'error',
        'message': 'Error interno del servidor',
        'timestamp': datetime.now().isoformat()
    }), 500

def setup_telegram_webhook():
    """Configura el webhook de Telegram"""
    try:
        if not config.telegram_token:
            logger.warning("Token de Telegram no configurado, omitiendo webhook")
            return

        # Construir URL del webhook
        webhook_url = config.webhook_url
        if not webhook_url and config.render_url:
            webhook_url = f"{config.render_url}/webhook"

        if webhook_url:
            logger.info(f"Configurando webhook: {webhook_url}")
            if telegram_client.configurar_webhook(webhook_url):
                logger.info("✅ Webhook configurado correctamente")
            else:
                logger.error("❌ Error configurando webhook")
        else:
            logger.warning("URL de webhook no disponible")
    except Exception as e:
        logger.error(f"Error configurando webhook: {e}")

def iniciar_flask_app():
    """Inicia la aplicación Flask"""
    try:
        logger.info("🌐 Iniciando servidor Flask...")
        # Configurar webhook si está disponible
        setup_telegram_webhook()

        # Iniciar servidor
        port = config.flask_port
        debug = config.flask_debug
        logger.info(f"📡 Servidor ejecutándose en puerto {port}")
        logger.info(f"🔍 Debug: {'activado' if debug else 'desactivado'}")
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            threaded=True
        )
    except Exception as e:
        logger.error(f"❌ Error iniciando Flask: {e}")
        raise

if __name__ == '__main__':
    iniciar_flask_app()
