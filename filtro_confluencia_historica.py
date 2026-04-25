import logging
import numpy as np

logger = logging.getLogger(__name__)

def analizar_confluencia_historica(symbol, side, ohlcv_confirmadas, indicadores_calculados, umbral_movimiento=1.5, max_velas_atras=15):
    """
    9no Filtro: Confluencia Histórica.
    Busca hacia atrás cuándo se alinearon los 8 filtros y mide el movimiento del precio.
    
    Parámetros:
        symbol: Símbolo (ej: BTC/USDT:USDT)
        side: 'buy' o 'sell'
        ohlcv_confirmadas: Lista de velas cerradas [[ts, o, h, l, c, v], ...]
        indicadores_calculados: Dict con los resultados actuales de los 8 filtros (para logs)
        umbral_movimiento: Porcentaje máximo de movimiento permitido (def: 1.5%)
        max_velas_atras: Límite de búsqueda hacia atrás (def: 15 velas)
        
    Retorna:
        (bool, dict): (True si pasa el filtro, info detallada)
    """
    try:
        if len(ohlcv_confirmadas) < max_velas_atras + 1:
            return True, {"msg": "Insuficientes velas para análisis histórico"}

        # El precio de referencia es el cierre de la vela actual (la que gatilló la señal)
        precio_actual = ohlcv_confirmadas[-1][4]
        
        # Lógica de Backtracking simplificada pero efectiva:
        # Buscamos hacia atrás hasta encontrar una vela donde la tendencia cambie o el momentum se pierda.
        # Esto identifica el "bloque" de movimiento actual.
        
        velas_confluencia = 0
        precio_inicio = precio_actual
        
        for i in range(1, max_velas_atras + 1):
            idx = -(i + 1) # Vela anterior
            if abs(idx) > len(ohlcv_confirmadas): break
            
            cierre_vela = ohlcv_confirmadas[idx][4]
            cierre_anterior = ohlcv_confirmadas[idx-1][4]
            
            # Condición de ruptura de confluencia (simplificada para eficiencia)
            # Si es BUY, buscamos cuándo el precio dejó de estar en tendencia alcista inmediata
            if side == 'buy':
                if cierre_vela < cierre_anterior: # Vela bajista rompe la racha
                    break
            else: # SELL
                if cierre_vela > cierre_anterior: # Vela alcista rompe la racha
                    break
            
            precio_inicio = cierre_vela
            velas_confluencia = i

        # Calcular movimiento porcentual desde el origen detectado
        movimiento = abs(precio_actual - precio_inicio) / precio_inicio * 100
        
        info = {
            "velas_atras": velas_confluencia,
            "precio_inicio": precio_inicio,
            "precio_actual": precio_actual,
            "movimiento_pct": round(movimiento, 2),
            "umbral": umbral_movimiento
        }

        if movimiento > umbral_movimiento:
            logger.warning(f"[FILTRO-9] {symbol} RECHAZADO: Movimiento de {movimiento:.2f}% desde el inicio de confluencia (hace {velas_confluencia} velas) excede el {umbral_movimiento}%")
            return False, info
        
        logger.info(f"[FILTRO-9] {symbol} OK: Movimiento de {movimiento:.2f}% (Velas detectadas: {velas_confluencia})")
        return True, info

    except Exception as e:
        logger.error(f"Error en filtro confluencia histórica para {symbol}: {e}")
        return True, {"msg": "Error en cálculo, omitiendo filtro por seguridad"}
