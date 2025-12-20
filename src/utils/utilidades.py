"""
Utilidades y funciones auxiliares.
Contiene funciones comunes, manejo de reportes y utilidades del sistema.
"""

import csv
import os
import logging
import statistics
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..config.settings import config

class UtilidadesBot:
    """Clase con utilidades y funciones auxiliares"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def filtrar_operaciones_ultima_semana(self) -> List[Dict[str, Any]]:
        """
        Filtra operaciones de los últimos 7 días
        
        Returns:
            Lista de operaciones recientes
        """
        try:
            if not os.path.exists(config.log_path):
                return []
                
            ops_recientes = []
            fecha_limite = datetime.now() - timedelta(days=Constants.DIAS_REPORTE_SEMANAL)
            
            with open(config.log_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        timestamp = datetime.fromisoformat(row['timestamp'])
                        if timestamp >= fecha_limite:
                            ops_recientes.append({
                                'timestamp': timestamp,
                                'symbol': row['symbol'],
                                'resultado': row['resultado'],
                                'pnl_percent': float(row['pnl_percent']),
                                'tipo': row['tipo'],
                                'breakout_usado': row.get('breakout_usado', 'False') == 'True'
                            })
                    except (ValueError, KeyError) as e:
                        self.logger.debug(f"Ignorando fila inválida: {e}")
                        continue
                        
            return ops_recientes
            
        except Exception as e:
            self.logger.error(f"Error filtrando operaciones: {e}")
            return []
    
    def contar_breakouts_semana(self) -> int:
        """
        Cuenta breakouts detectados en la última semana
        
        Returns:
            Número de breakouts
        """
        try:
            ops = self.filtrar_operaciones_ultima_semana()
            breakouts = sum(1 for op in ops if op.get('breakout_usado', False))
            return breakouts
            
        except Exception as e:
            self.logger.error(f"Error contando breakouts: {e}")
            return 0
    
    def generar_reporte_semanal(self) -> Optional[str]:
        """
        Genera reporte semanal automático
        
        Returns:
            Mensaje del reporte o None
        """
        try:
            ops_ultima_semana = self.filtrar_operaciones_ultima_semana()
            
            if not ops_ultima_semana:
                return None
                
            # Calcular métricas
            total_ops = len(ops_ultima_semana)
            wins = sum(1 for op in ops_ultima_semana if op['resultado'] == Constants.OPERACION_TP)
            losses = sum(1 for op in ops_ultima_semana if op['resultado'] == Constants.OPERACION_SL)
            winrate = (wins/total_ops*100) if total_ops > 0 else 0
            pnl_total = sum(op['pnl_percent'] for op in ops_ultima_semana)
            
            # Encontrar mejor y peor operación
            mejor_op = max(ops_ultima_semana, key=lambda x: x['pnl_percent'])
            peor_op = min(ops_ultima_semana, key=lambda x: x['pnl_percent'])
            
            # Calcular promedios
            ganancias = [op['pnl_percent'] for op in ops_ultima_semana if op['pnl_percent'] > 0]
            perdidas = [abs(op['pnl_percent']) for op in ops_ultima_semana if op['pnl_percent'] < 0]
            avg_ganancia = sum(ganancias)/len(ganancias) if ganancias else 0
            avg_perdida = sum(perdidas)/len(perdidas) if perdidas else 0
            
            # Calcular racha actual
            racha_actual = 0
            for op in reversed(ops_ultima_semana):
                if op['resultado'] == Constants.OPERACION_TP:
                    racha_actual += 1
                else:
                    break
            
            # Emoji según resultado
            emoji_resultado = "🟢" if pnl_total > 0 else "🔴" if pnl_total < 0 else "⚪"
            
            mensaje = f"""
━━━━━━━━━━━━━━━━━━━━
📊 <b>REPORTE SEMANAL</b>
━━━━━━━━━━━━━━━━━━━━
📅 {datetime.now().strftime('%d/%m/%Y')} | Últimos 7 días
<b>RENDIMIENTO GENERAL</b>
{emoji_resultado} PnL Total: <b>{pnl_total:+.2f}%</b>
📈 Win Rate: <b>{winrate:.1f}%</b>
✅ Ganadas: {wins} | ❌ Perdidas: {losses}
<b>ESTADÍSTICAS</b>
📊 Operaciones: {total_ops}
💰 Ganancia Promedio: +{avg_ganancia:.2f}%
📉 Pérdida Promedio: -{avg_perdida:.2f}%
🔥 Racha actual: {racha_actual} wins
<b>DESTACADOS</b>
🏆 Mejor: {mejor_op['symbol']} ({mejor_op['tipo']})
   → {mejor_op['pnl_percent']:+.2f}%
⚠️ Peor: {peor_op['symbol']} ({peor_op['tipo']})
   → {peor_op['pnl_percent']:+.2f}%
━━━━━━━━━━━━━━━━━━━━
🤖 Bot automático 24/7
⚡ Estrategia: Breakout + Reentry
💎 Acceso Premium: @TuUsuario
            """
            
            return mensaje
            
        except Exception as e:
            self.logger.error(f"Error generando reporte semanal: {e}")
            return None
    
    def enviar_reporte_semanal(self, bot_instance) -> bool:
        """
        Envía el reporte semanal por Telegram
        
        Args:
            bot_instance: Instancia del bot para acceso a configuraciones
            
        Returns:
            True si se envió exitosamente
        """
        try:
            mensaje = self.generar_reporte_semanal()
            if not mensaje:
                self.logger.info("ℹ️ No hay datos suficientes para generar reporte")
                return False
            
            if config.telegram_token and config.telegram_chat_ids:
                from ..api.clients import telegram_client
                resultado = telegram_client.enviar_mensaje(config.telegram_chat_ids, mensaje)
                
                if resultado:
                    self.logger.info("✅ Reporte semanal enviado correctamente")
                    return True
                else:
                    self.logger.error("❌ Error enviando reporte semanal")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error enviando reporte semanal: {e}")
            return False
    
    def verificar_envio_reporte_automatico(self, bot_instance) -> bool:
        """
        Verifica si debe enviar el reporte semanal (cada lunes a las 9:00)
        
        Args:
            bot_instance: Instancia del bot
            
        Returns:
            True si se envió el reporte
        """
        try:
            ahora = datetime.now()
            
            # Verificar si es lunes entre las 9:00 y 10:00
            if ahora.weekday() == 0 and Constants.HORA_REPORTE_SEMANAL <= ahora.hour < Constants.HORA_REPORTE_SEMANAL + 1:
                archivo_control = config.ultimo_reporte_file
                
                try:
                    # Verificar si ya se envió hoy
                    if os.path.exists(archivo_control):
                        with open(archivo_control, 'r') as f:
                            ultima_fecha = f.read().strip()
                            if ultima_fecha == ahora.strftime('%Y-%m-%d'):
                                return False
                    
                    # Enviar reporte
                    if self.enviar_reporte_semanal(bot_instance):
                        with open(archivo_control, 'w') as f:
                            f.write(ahora.strftime('%Y-%m-%d'))
                        return True
                        
                except Exception as e:
                    self.logger.error(f"Error en envío automático: {e}")
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error verificando envío automático: {e}")
            return False
    
    def validar_configuracion(self) -> Dict[str, Any]:
        """
        Valida la configuración del bot
        
        Returns:
            Diccionario con resultados de validación
        """
        try:
            validacion = {
                'valida': True,
                'errores': [],
                'advertencias': [],
                'configuracion': {}
            }
            
            # Verificar tokens y APIs
            if not config.telegram_token:
                validacion['errores'].append("Token de Telegram no configurado")
                validacion['valida'] = False
                
            if not config.telegram_chat_ids:
                validacion['advertencias'].append("Chat IDs de Telegram no configurados")
            
            # Verificar símbolos
            if not config.symbols:
                validacion['errores'].append("No hay símbolos configurados")
                validacion['valida'] = False
            elif len(config.symbols) > 50:
                validacion['advertencias'].append(f"Muchos símbolos configurados ({len(config.symbols)})")
            
            # Verificar timeframes
            if not config.timeframes:
                validacion['errores'].append("No hay timeframes configurados")
                validacion['valida'] = False
            
            # Verificar parámetros de trading
            if config.min_channel_width_percent <= 0:
                validacion['errores'].append("Ancho mínimo de canal debe ser positivo")
                validacion['valida'] = False
                
            if config.scan_interval_minutes <= 0:
                validacion['errores'].append("Intervalo de escaneo debe ser positivo")
                validacion['valida'] = False
            
            # Compilar configuración actual
            validacion['configuracion'] = {
                'símbolos': len(config.symbols),
                'timeframes': config.timeframes,
                'auto_optimize': config.auto_optimize,
                'scan_interval': config.scan_interval_minutes,
                'min_channel_width': config.min_channel_width_percent,
                'telegram_configurado': bool(config.telegram_token)
            }
            
            return validacion
            
        except Exception as e:
            self.logger.error(f"Error validando configuración: {e}")
            return {
                'valida': False,
                'errores': [f"Error validando: {e}"],
                'advertencias': [],
                'configuracion': {}
            }
    
    def generar_estadisticas_sistema(self) -> Dict[str, Any]:
        """
        Genera estadísticas del sistema y rendimiento
        
        Returns:
            Diccionario con estadísticas
        """
        try:
            estadisticas = {
                'timestamp': datetime.now().isoformat(),
                'archivos': {},
                'conexiones': {},
                'rendimiento': {}
            }
            
            # Estadísticas de archivos
            if os.path.exists(config.log_path):
                try:
                    with open(config.log_path, 'r', encoding='utf-8') as f:
                        lines = sum(1 for _ in f) - 1  # -1 por header
                    estadisticas['archivos']['operaciones_log'] = lines
                except:
                    estadisticas['archivos']['operaciones_log'] = 'error'
            
            if os.path.exists(config.estado_file):
                estadisticas['archivos']['estado_bot'] = 'ok'
            else:
                estadisticas['archivos']['estado_bot'] = 'no existe'
            
            if os.path.exists(config.mejores_parametros_file):
                estadisticas['archivos']['parametros_optimizados'] = 'ok'
            else:
                estadisticas['archivos']['parametros_optimizados'] = 'no existe'
            
            # Estadísticas de conexiones
            from ..api.clients import binance_client, telegram_client
            
            estadisticas['conexiones']['binance'] = binance_client.verificar_conexion()
            estadisticas['conexiones']['telegram'] = bool(config.telegram_token)
            
            # Estadísticas de rendimiento
            ops_recientes = self.filtrar_operaciones_ultima_semana()
            if ops_recientes:
                pnls = [op['pnl_percent'] for op in ops_recientes]
                estadisticas['rendimiento'] = {
                    'operaciones_semana': len(ops_recientes),
                    'winrate': sum(1 for op in ops_recientes if op['resultado'] == Constants.OPERACION_TP) / len(ops_recientes) * 100,
                    'pnl_promedio': statistics.mean(pnls),
                    'mejor_pnl': max(pnls),
                    'peor_pnl': min(pnls)
                }
            
            return estadisticas
            
        except Exception as e:
            self.logger.error(f"Error generando estadísticas: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def limpiar_datos_antiguos(self, dias_retener: int = 30) -> Dict[str, int]:
        """
        Limpia datos antiguos para mantener el rendimiento
        
        Args:
            dias_retener: Días de datos a mantener
            
        Returns:
            Diccionario con conteos de limpieza
        """
        try:
            limpieza = {
                'operaciones_eliminadas': 0,
                'breakouts_limpiados': 0,
                'archivos_procesados': 0
            }
            
            fecha_limite = datetime.now() - timedelta(days=dias_retener)
            
            # Limpiar archivo de log
            if os.path.exists(config.log_path):
                try:
                    temp_file = config.log_path + '.temp'
                    with open(config.log_path, 'r', encoding='utf-8') as infile, \
                         open(temp_file, 'w', newline='', encoding='utf-8') as outfile:
                        
                        reader = csv.DictReader(infile)
                        writer = csv.writer(outfile)
                        
                        # Escribir header
                        writer.writerow(reader.fieldnames)
                        
                        # Filtrar datos recientes
                        for row in reader:
                            try:
                                timestamp = datetime.fromisoformat(row['timestamp'])
                                if timestamp >= fecha_limite:
                                    writer.writerow(row.values())
                                else:
                                    limpieza['operaciones_eliminadas'] += 1
                            except:
                                # Mantener filas problemáticas
                                writer.writerow(row.values())
                    
                    # Reemplazar archivo
                    os.replace(temp_file, config.log_path)
                    limpieza['archivos_procesados'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error limpiando log: {e}")
            
            # Limpiar estado de breakouts antiguos
            try:
                # Esta función sería llamada desde el bot principal
                # para limpiar breakouts_detectados antiguos
                pass
            except Exception as e:
                self.logger.error(f"Error limpiando breakouts: {e}")
            
            self.logger.info(f"Limpieza completada: {limpieza}")
            return limpieza
            
        except Exception as e:
            self.logger.error(f"Error en limpieza de datos: {e}")
            return {'error': str(e)}
    
    def crear_backup_estado(self) -> bool:
        """
        Crea backup del estado actual del bot
        
        Returns:
            True si el backup fue exitoso
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Backup del archivo de estado
            if os.path.exists(config.estado_file):
                backup_file = f"{config.estado_file}.backup_{timestamp}"
                try:
                    import shutil
                    shutil.copy2(config.estado_file, backup_file)
                    self.logger.info(f"✅ Backup creado: {backup_file}")
                except Exception as e:
                    self.logger.error(f"Error creando backup de estado: {e}")
                    return False
            
            # Backup del log de operaciones
            if os.path.exists(config.log_path):
                backup_log = f"{config.log_path}.backup_{timestamp}"
                try:
                    import shutil
                    shutil.copy2(config.log_path, backup_log)
                    self.logger.info(f"✅ Backup de log creado: {backup_log}")
                except Exception as e:
                    self.logger.error(f"Error creando backup de log: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creando backup: {e}")
            return False
    
    def verificar_integridad_datos(self) -> Dict[str, Any]:
        """
        Verifica la integridad de los archivos de datos
        
        Returns:
            Diccionario con resultados de verificación
        """
        try:
            verificacion = {
                'timestamp': datetime.now().isoformat(),
                'archivos': {},
                'errores': [],
                'advertencias': []
            }
            
            # Verificar archivo de estado
            if os.path.exists(config.estado_file):
                try:
                    import json
                    with open(config.estado_file, 'r', encoding='utf-8') as f:
                        estado = json.load(f)
                    verificacion['archivos']['estado'] = 'ok'
                except json.JSONDecodeError:
                    verificacion['archivos']['estado'] = 'json_corrupto'
                    verificacion['errores'].append("Archivo de estado con JSON corrupto")
                except Exception as e:
                    verificacion['archivos']['estado'] = 'error'
                    verificacion['errores'].append(f"Error leyendo estado: {e}")
            else:
                verificacion['archivos']['estado'] = 'no_existe'
                verificacion['advertencias'].append("Archivo de estado no existe")
            
            # Verificar archivo de log
            if os.path.exists(config.log_path):
                try:
                    with open(config.log_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                    verificacion['archivos']['log'] = f'ok ({len(rows)} operaciones)'
                except Exception as e:
                    verificacion['archivos']['log'] = 'error'
                    verificacion['errores'].append(f"Error leyendo log: {e}")
            else:
                verificacion['archivos']['log'] = 'no_existe'
                verificacion['advertencias'].append("Archivo de log no existe")
            
            # Verificar parámetros optimizados
            if os.path.exists(config.mejores_parametros_file):
                try:
                    with open(config.mejores_parametros_file, 'r', encoding='utf-8') as f:
                        import json
                        params = json.load(f)
                    verificacion['archivos']['parametros'] = 'ok'
                except Exception as e:
                    verificacion['archivos']['parametros'] = 'error'
                    verificacion['errores'].append(f"Error leyendo parámetros: {e}")
            else:
                verificacion['archivos']['parametros'] = 'no_existe'
            
            return verificacion
            
        except Exception as e:
            self.logger.error(f"Error verificando integridad: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

# Instancia global de utilidades
utilidades_bot = UtilidadesBot()

print("🔧 Utilidades del bot cargadas correctamente")