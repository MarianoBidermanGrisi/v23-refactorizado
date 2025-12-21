import os
import json
import time
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from io import BytesIO
from ..config.settings import config

class TelegramAPIClient:
    """Cliente de Telegram API con logs detallados y corrección para envío de gráficos"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = config.telegram_api_base
        self.token = config.telegram_token
        self.session = requests.Session()
        # Estadísticas de uso
        self.stats = {
            'mensajes_enviados': 0,
            'graficos_enviados': 0,
            'errores_envio': 0,
            'tiempo_total_envio': 0.0,
            'chat_ids_atendidos': set()
        }
        self.logger.info("📱 [TELEGRAM] Cliente de Telegram API inicializado")
        self.logger.info(f" • Base URL: {self.base_url}")
        self.logger.info(f" • Token configurado: {'✅' if self.token else '❌'}")

    def _hacer_request(self, method: str, data: Dict[str, Any] = None, files: Dict[str, Any] = None) -> Optional[Dict[Any, Any]]:
        """
        Realiza request a la API de Telegram con logs detallados - CORREGIDO para archivos
        Args:
            method: Método de la API
            data: Datos del request (para parámetros normales)
            files: Archivos a enviar (para sendPhoto, sendDocument, etc.)
        Returns:
            Respuesta de la API o None
        """
        try:
            url = f"{self.base_url}/bot{self.token}/{method}"
            self.logger.debug(f"📤 [TELEGRAM] Envío: {method}")
            if data:
                # Log sin el token por seguridad
                data_sin_token = {k: v for k, v in data.items() if k != 'caption'}
                self.logger.debug(f" • Datos: {data_sin_token}")
            if files:
                self.logger.debug(f" • Archivos: {len(files)} archivo(s) adjunto(s)")

            # CORRECCIÓN: Manejar correctamente archivos vs datos
            if files:
                # Para envío de archivos (sendPhoto, sendDocument, etc.)
                response = self.session.post(url, data=data, files=files, timeout=30)
            else:
                # Para mensajes de texto normales
                response = self.session.post(url, json=data, timeout=30)
            
            self.logger.debug(f"📊 [TELEGRAM] Status code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    self.logger.debug(f"✅ [TELEGRAM] {method} exitoso")
                    return result
                else:
                    self.logger.error(f"❌ [TELEGRAM] {method} falló: {result.get('description', 'Error desconocido')}")
                    return None
            else:
                self.logger.error(f"❌ [TELEGRAM] Error HTTP {response.status_code}: {response.text}")
                return None
        except Exception as e:
            self.logger.error(f"❌ [TELEGRAM] Error en request {method}: {e}")
            return None

    def enviar_mensaje(self, chat_ids: List[str], mensaje: str, parse_mode: str = 'Markdown') -> bool:
        """
        Envía mensaje por Telegram con logs detallados
        Args:
            chat_ids: Lista de chat IDs
            mensaje: Mensaje a enviar
            parse_mode: Modo de parseo
        Returns:
            True si el envío es exitoso
        """
        try:
            if not self.token:
                self.logger.warning("⚠️ [TELEGRAM] Token no configurado, omitiendo envío")
                return False

            timestamp_inicio = datetime.now()
            exitosos = 0
            fallidos = 0

            self.logger.debug(f"📨 [TELEGRAM] Enviando mensaje a {len(chat_ids)} chats")
            self.logger.debug(f" • Mensaje: {len(mensaje)} caracteres")

            for chat_id in chat_ids:
                try:
                    self.stats['chat_ids_atendidos'].add(chat_id)
                    data = {
                        'chat_id': chat_id,
                        'text': mensaje,
                        'parse_mode': parse_mode
                    }

                    result = self._hacer_request('sendMessage', data)

                    if result:
                        exitosos += 1
                        self.logger.debug(f" ✅ Chat {chat_id}: Mensaje enviado")
                    else:
                        fallidos += 1
                        self.logger.error(f" ❌ Chat {chat_id}: Error enviando mensaje")

                    # Pausa para evitar rate limits
                    time.sleep(0.2)

                except Exception as e:
                    fallidos += 1
                    self.logger.error(f" ❌ Chat {chat_id}: Error inesperado: {e}")

            timestamp_fin = datetime.now()
            tiempo_envio = (timestamp_fin - timestamp_inicio).total_seconds()
            self.stats['tiempo_total_envio'] += tiempo_envio

            if exitosos > 0:
                self.stats['mensajes_enviados'] += exitosos
                self.logger.info(f"📤 [TELEGRAM] Mensaje enviado: {exitosos}/{len(chat_ids)} chats")
                self.logger.info(f" • Tiempo total: {tiempo_envio:.3f}s")
                return True
            else:
                self.stats['errores_envio'] += fallidos
                self.logger.error(f"❌ [TELEGRAM] Fallo al enviar mensaje a todos los chats")
                return False

        except Exception as e:
            self.logger.error(f"❌ [TELEGRAM] Error enviando mensaje: {e}")
            self.stats['errores_envio'] += 1
            return False

    def enviar_grafico(self, chat_ids: List[str], imagen: Union[BytesIO, bytes], caption: str = None) -> bool:
        """
        Envía gráfico por Telegram con logs detallados - CORREGIDO
        Args:
            chat_ids: Lista de chat IDs
            imagen: Imagen a enviar
            caption: Leyenda de la imagen
        Returns:
            True si el envío es exitoso
        """
        try:
            if not self.token:
                self.logger.warning("⚠️ [TELEGRAM] Token no configurado, omitiendo envío de gráfico")
                return False

            timestamp_inicio = datetime.now()
            exitosos = 0
            fallidos = 0

            self.logger.debug(f"🖼️ [TELEGRAM] Enviando gráfico a {len(chat_ids)} chats")
            if caption:
                self.logger.debug(f" • Caption: {len(caption)} caracteres")

            # Preparar archivo
            if isinstance(imagen, BytesIO):
                imagen.seek(0)
                files = {'photo': ('chart.png', imagen, 'image/png')}
            else:
                files = {'photo': ('chart.png', imagen, 'image/png')}

            for chat_id in chat_ids:
                try:
                    self.stats['chat_ids_atendidos'].add(chat_id)
                    data = {'chat_id': chat_id}
                    if caption:
                        data['caption'] = caption
                    data['parse_mode'] = 'Markdown'

                    # CORRECCIÓN: Pasar archivos por separado, no mezclados con data
                    result = self._hacer_request('sendPhoto', data, files)

                    if result:
                        exitosos += 1
                        self.logger.debug(f" ✅ Chat {chat_id}: Gráfico enviado exitosamente")
                        # NUEVO LOG DE CONFIRMACIÓN DETALLADO
                        self.logger.info(f"🎯 [TELEGRAM] ✅ GRÁFICO ENVIADO EXITOSAMENTE - Chat: {chat_id}")
                        if caption:
                            self.logger.info(f" 📝 [TELEGRAM] Caption enviado: {caption[:50]}{'...' if len(caption) > 50 else ''}")
                    else:
                        fallidos += 1
                        self.logger.error(f" ❌ Chat {chat_id}: Error enviando gráfico")

                    # Pausa para evitar rate limits
                    time.sleep(0.5)

                except Exception as e:
                    fallidos += 1
                    self.logger.error(f" ❌ Chat {chat_id}: Error inesperado: {e}")

            timestamp_fin = datetime.now()
            tiempo_envio = (timestamp_fin - timestamp_inicio).total_seconds()
            self.stats['tiempo_total_envio'] += tiempo_envio

            if exitosos > 0:
                self.stats['graficos_enviados'] += exitosos
                # LOG DE CONFIRMACIÓN MEJORADO
                self.logger.info(f"📤 [TELEGRAM] ✅ GRÁFICO ENVIADO EXITOSAMENTE: {exitosos}/{len(chat_ids)} chats")
                self.logger.info(f" ⏱️ [TELEGRAM] Tiempo total de envío: {tiempo_envio:.3f}s")
                self.logger.info(f" 📊 [TELEGRAM] Total gráficos enviados hoy: {self.stats['graficos_enviados']}")
                return True
            else:
                self.stats['errores_envio'] += fallidos
                self.logger.error(f"❌ [TELEGRAM] FALLO TOTAL: No se pudo enviar gráfico a ningún chat")
                return False

        except Exception as e:
            self.logger.error(f"❌ [TELEGRAM] Error general enviando gráfico: {e}")
            self.stats['errores_envio'] += 1
            return False

    def configurar_webhook(self, webhook_url: str) -> bool:
        """
        Configura el webhook de Telegram
        Args:
            webhook_url: URL del webhook
        Returns:
            True si la configuración fue exitosa
        """
        try:
            if not self.token:
                self.logger.warning("⚠️ [TELEGRAM] Token no configurado, omitiendo configuración de webhook")
                return False

            data = {
                'url': webhook_url,
                'drop_pending_updates': True
            }

            result = self._hacer_request('setWebhook', data)
            return result is not None

        except Exception as e:
            self.logger.error(f"❌ [TELEGRAM] Error configurando webhook: {e}")
            return False

    def obtener_estadisticas_api(self) -> Dict[str, Any]:
        """Obtiene estadísticas de uso de la API"""
        stats = self.stats.copy()
        stats.update({
            'chat_ids_unicos': len(self.stats['chat_ids_atendidos']),
            'total_envios': self.stats['mensajes_enviados'] + self.stats['graficos_enviados'],
            'tasa_exito': ((self.stats['mensajes_enviados'] + self.stats['graficos_enviados']) /
                           max(1, self.stats['mensajes_enviados'] + self.stats['graficos_enviados'] + self.stats['errores_envio']) * 100)
        })
        return stats

    def obtener_estado_conexion(self) -> Dict[str, Any]:
        """
        Obtiene el estado de conexión con Telegram - NUEVA FUNCIÓN
        Returns:
            Estado detallado de la conexión
        """
        try:
            if not self.token:
                return {
                    'conectado': False,
                    'error': 'Token no configurado',
                    'timestamp': datetime.now().isoformat()
                }

            # Verificar conectividad
            data = {'offset': -1, 'limit': 1}
            result = self._hacer_request('getUpdates', data)
            
            return {
                'conectado': result is not None,
                'token_configurado': bool(self.token),
                'graficos_enviados_hoy': self.stats['graficos_enviados'],
                'mensajes_enviados_hoy': self.stats['mensajes_enviados'],
                'errores_hoy': self.stats['errores_envio'],
                'chats_atendidos': len(self.stats['chat_ids_atendidos']),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'conectado': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class BinanceAPIClient:
    """Cliente de Binance API"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = config.binance_api_base
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': config.binance_api_key
        })
        self.logger.info("💹 [BINANCE] Cliente de Binance API inicializado")

    def obtener_datos_klines(self, symbol: str, interval: str, limit: int = 200) -> Optional[List[Dict[str, Any]]]:
        """
        Obtiene datos de klines de Binance
        Args:
            symbol: Par de trading (ej: BTCUSDT)
            interval: Intervalo (ej: 1h, 4h, 1d)
            limit: Número de velas
        Returns:
            Lista de datos de klines
        """
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            self.logger.debug(f"📊 [BINANCE] Obteniendo datos: {symbol} {interval} ({limit} velas)")

            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                self.logger.debug(f"✅ [BINANCE] Datos obtenidos: {len(data)} velas")
                return data
            else:
                self.logger.error(f"❌ [BINANCE] Error HTTP {response.status_code}: {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"❌ [BINANCE] Error obteniendo datos: {e}")
            return None


class RenderHealthClient:
    """Cliente para verificar health de Render"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def verificar_health(self) -> Dict[str, Any]:
        """Verifica el estado de health del sistema"""
        try:
            return {
                'status': 'ok',
                'timestamp': datetime.now().isoformat(),
                'servicio': 'bot_trading'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Instancias globales de clientes
telegram_client = TelegramAPIClient()
binance_client = BinanceAPIClient()
render_health_client = RenderHealthClient()
