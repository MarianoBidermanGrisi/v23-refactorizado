"""
Indicador ADX/DI (Average Directional Index + Directional Movement Index)
para confirmación de señales de trading en el bot Breakout + Reentry.

Este módulo replica la lógica del ADX/DI para confirmar tendencias
antes de ejecutar operaciones LONG o SHORT.
"""

import pandas as pd
import numpy as np


def calculate_adx_di(data: pd.DataFrame, length: int = 14, threshold: int = 20) -> pd.DataFrame:
    """
    Calcula el ADX (Average Directional Index) y los indicadores DI+ y DI- 
    a partir de datos OHLC, replicando la lógica del script de Pine Script.
    
    Args:
        data: DataFrame con columnas 'high', 'low', 'close'
        length: Período para el cálculo (default: 14)
        threshold: Umbral del ADX para considerar tendencia fuerte (default: 20)
    
    Returns:
        DataFrame con columnas: DIPlus, DIMinus, ADX, Threshold, Signal
    """
    # Verificar columnas necesarias
    required_cols = {'high', 'low', 'close'}
    if not required_cols.issubset(data.columns):
        missing = required_cols - set(data.columns)
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    
    df = data.copy()
    
    # --- Paso 1: Calcular True Range (TR) ---
    prev_close = df['close'].shift(1)
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - prev_close)
    low_close_prev = abs(df['low'] - prev_close)
    df['tr'] = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    
    # --- Paso 2: Directional Movement (DM) ---
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    
    up_move = df['high'] - prev_high
    down_move = prev_low - df['low']
    
    condition_plus = (up_move > down_move) & (up_move > 0)
    condition_minus = (down_move > up_move) & (down_move > 0)
    
    df['dm_plus'] = np.where(condition_plus, up_move, 0)
    df['dm_minus'] = np.where(condition_minus, down_move, 0)
    
    # --- Paso 3: Wilder's Smoothing ---
    # Inicializar con SMA para los primeros valores
    df['smoothed_tr'] = df['tr'].rolling(window=length).mean()
    df['smoothed_dm_plus'] = df['dm_plus'].rolling(window=length).mean()
    df['smoothed_dm_minus'] = df['dm_minus'].rolling(window=length).mean()
    
    # Aplicar suavizado Wilder recursivamente
    for i in range(length, len(df)):
        df.loc[df.index[i], 'smoothed_tr'] = (
            df.loc[df.index[i-1], 'smoothed_tr'] * (length - 1) + df.loc[df.index[i], 'tr']
        ) / length
        df.loc[df.index[i], 'smoothed_dm_plus'] = (
            df.loc[df.index[i-1], 'smoothed_dm_plus'] * (length - 1) + df.loc[df.index[i], 'dm_plus']
        ) / length
        df.loc[df.index[i], 'smoothed_dm_minus'] = (
            df.loc[df.index[i-1], 'smoothed_dm_minus'] * (length - 1) + df.loc[df.index[i], 'dm_minus']
        ) / length
    
    # --- Paso 4: Calcular DI+ y DI- ---
    # Evitar división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        df['DIPlus'] = 100 * df['smoothed_dm_plus'] / df['smoothed_tr']
        df['DIMinus'] = 100 * df['smoothed_dm_minus'] / df['smoothed_tr']
    
    df['DIPlus'] = df['DIPlus'].fillna(0).replace([np.inf, -np.inf], 0)
    df['DIMinus'] = df['DIMinus'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # --- Paso 5: Calcular DX y ADX ---
    di_sum = df['DIPlus'] + df['DIMinus']
    with np.errstate(divide='ignore', invalid='ignore'):
        df['DX'] = 100 * abs(df['DIPlus'] - df['DIMinus']) / di_sum
    
    df['DX'] = df['DX'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # ADX como SMA de DX
    df['ADX'] = df['DX'].rolling(window=length).mean()
    df['Threshold'] = threshold
    
    # --- Paso 6: Generar señales ---
    # LONG: DI+ > DI- Y ADX > threshold (tendencia alcista fuerte)
    # SHORT: DI- > DI+ Y ADX > threshold (tendencia bajista fuerte)
    df['Signal'] = np.where(
        (df['ADX'] > threshold) & (df['DIPlus'] > df['DIMinus']), 'LONG',
        np.where(
            (df['ADX'] > threshold) & (df['DIMinus'] > df['DIPlus']), 'SHORT',
            'NEUTRAL'
        )
    )
    
    # Devolver solo las columnas de resultado
    return df[['DIPlus', 'DIMinus', 'ADX', 'Threshold', 'Signal']]


def get_adx_confirmation(di_plus: float, di_minus: float, adx: float, threshold: float = 20) -> dict:
    """
    Función simple para obtener confirmación ADX/DI con valores individuales.
    
    Args:
        di_plus: Valor actual de DI+
        di_minus: Valor actual de DI-
        adx: Valor actual del ADX
        threshold: Umbral para considerar tendencia fuerte
    
    Returns:
        Dict con: 'confirmacion', 'direccion', 'fuerza', 'mensaje'
    """
    if adx < threshold:
        return {
            'confirmacion': False,
            'direccion': 'NEUTRAL',
            'fuerza': 'DEBIL',
            'mensaje': f'ADX={adx:.1f} < threshold={threshold}: tendencia débil'
        }
    
    if di_plus > di_minus:
        diferencia = di_plus - di_minus
        fuerza = 'FUERTE' if diferencia > 10 else 'MODERADA' if diferencia > 5 else 'DEBIL'
        return {
            'confirmacion': True,
            'direccion': 'LONG',
            'fuerza': fuerza,
            'mensaje': f'ADX={adx:.1f} >= {threshold}, DI+={di_plus:.1f} > DI-={di_minus:.1f}: tendencia ALCISTA'
        }
    elif di_minus > di_plus:
        diferencia = di_minus - di_plus
        fuerza = 'FUERTE' if diferencia > 10 else 'MODERADA' if diferencia > 5 else 'DEBIL'
        return {
            'confirmacion': True,
            'direccion': 'SHORT',
            'fuerza': fuerza,
            'mensaje': f'ADX={adx:.1f} >= {threshold}, DI-={di_minus:.1f} > DI+={di_plus:.1f}: tendencia BAJISTA'
        }
    else:
        return {
            'confirmacion': False,
            'direccion': 'NEUTRAL',
            'fuerza': 'DEBIL',
            'mensaje': f'ADX={adx:.1f} >= {threshold}, pero DI+ = DI-: sin dirección clara'
        }


def preparar_datos_adx(datos_mercado: dict) -> pd.DataFrame:
    """
    Convierte los datos del mercado (formato del bot) a DataFrame para ADX.
    
    Args:
        datos_mercado: Dict con listas de 'maximos', 'minimos', 'cierres'
    
    Returns:
        DataFrame con columnas 'high', 'low', 'close'
    """
    if not datos_mercado or len(datos_mercado.get('cierres', [])) < 14:
        return None
    
    df = pd.DataFrame({
        'high': datos_mercado.get('maximos', []),
        'low': datos_mercado.get('minimos', []),
        'close': datos_mercado.get('cierres', [])
    })
    
    return df
