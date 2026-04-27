import logging

logger = logging.getLogger(__name__)

def analizar_confluencia_historica(symbol, side, ohlcv_confirmadas, indicadores_calculados, umbral_movimiento=1.5, max_velas_atras=15):
    """
    9no Filtro: Confluencia Histórica.
    Calcula cuánto se ha movido el precio desde el inicio real de la tendencia actual,
    evitando entrar "tarde" en movimientos ya extendidos.
    """
    try:
        if len(ohlcv_confirmadas) < max_velas_atras + 1:
            return True, {"msg": "Insuficientes velas para análisis histórico", "movimiento_pct": 0, "velas_atras": 0}

        precio_actual = ohlcv_confirmadas[-1][4]
        
        # 1. Determinar el inicio de la tendencia usando TDF (si está disponible)
        tdf_detalle = indicadores_calculados.get("tdf", {}).get("detalle", {})
        trend_count = tdf_detalle.get("trend_count", 0)
        
        # Si TDF nos dice que la tendencia lleva X velas, usamos ese rango (limitado a max_velas_atras)
        if trend_count > 0:
            velas_evaluar = min(max(trend_count, 5), max_velas_atras)
        else:
            velas_evaluar = max_velas_atras
            
        # 2. Extraer el segmento histórico a evaluar (sin incluir la vela actual)
        segmento = ohlcv_confirmadas[-(velas_evaluar + 1):-1]
        
        # 3. Encontrar el precio de origen real (Low para compras, High para ventas)
        # Esto evita que una pequeña vela de retroceso "rompa" el conteo.
        if side == 'buy':
            # Buscamos el precio más bajo (Low) del segmento como origen del impulso
            precio_origen = min([vela[3] for vela in segmento])
            movimiento = ((precio_actual - precio_origen) / precio_origen) * 100
        else: # sell
            # Buscamos el precio más alto (High) del segmento como origen del impulso
            precio_origen = max([vela[2] for vela in segmento])
            movimiento = ((precio_origen - precio_actual) / precio_origen) * 100
            
        info = {
            "velas_atras": velas_evaluar,
            "precio_origen": precio_origen,
            "precio_actual": precio_actual,
            "movimiento_pct": round(movimiento, 2),
            "umbral": umbral_movimiento
        }

        if movimiento > umbral_movimiento:
            logger.warning(f"[FILTRO-9] {symbol} RECHAZADO: Movimiento de {movimiento:.2f}% desde el origen de la tendencia (hace {velas_evaluar} velas) excede el {umbral_movimiento}%")
            return False, info
        
        logger.info(f"[FILTRO-9] {symbol} OK: Movimiento de {movimiento:.2f}% (Origen detectado hace {velas_evaluar} velas)")
        return True, info

    except Exception as e:
        logger.error(f"Error en filtro confluencia histórica para {symbol}: {e}")
        return True, {"msg": f"Error: {str(e)}", "movimiento_pct": 0, "velas_atras": 0}
