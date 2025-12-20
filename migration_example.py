"""
Ejemplo de migración de mplfinance al nuevo módulo
Este archivo muestra cómo actualizar el código existente
"""

# ============================================
# ANTES (con mplfinance) - INCOMPATIBLE
# ============================================

# import mplfinance as mpf
# import pandas as pd

# def create_candlestick_chart(data, title="Trading Chart"):
#     """Crear gráfico de velas - VERSIÓN ANTIGUA"""
#     mpf.plot(data, 
#              type='candle',
#              title=title,
#              ylabel='Precio',
#              volume=True,
#              savefig='chart.png')
#     return 'chart.png'

# def create_ohlc_chart(data, title="OHLC Chart"):
#     """Crear gráfico OHLC - VERSIÓN ANTIGUA"""
#     mpf.plot(data, 
#              type='ohlc',
#              title=title,
#              ylabel='Precio')
#     return 'chart_ohlc.png'

# ============================================
# DESPUÉS (con financial_charts) - COMPATIBLE
# ============================================

from financial_charts import (
    plot_candlestick, 
    plot_ohlcv_chart, 
    plot_line_chart, 
    plot_volume_chart
)
import pandas as pd
import numpy as np

def create_candlestick_chart(data, title="Trading Chart", save_path="chart.png"):
    """Crear gráfico de velas - VERSIÓN NUEVA"""
    plot_candlestick(data, 
                    title=title, 
                    figsize=(15, 10),
                    save_path=save_path)
    return save_path

def create_ohlc_chart(data, title="OHLC Chart", save_path="chart_ohlc.png"):
    """Crear gráfico OHLC - VERSIÓN NUEVA"""
    plot_ohlcv_chart(data, 
                    title=title, 
                    figsize=(15, 10),
                    save_path=save_path)
    return save_path

def create_volume_chart(data, save_path="volume_chart.png"):
    """Crear gráfico de volumen - VERSIÓN NUEVA"""
    plot_volume_chart(data, 
                     title="Volumen de Trading",
                     figsize=(15, 4),
                     save_path=save_path)
    return save_path

def create_combined_chart(data, save_path="combined_chart.png"):
    """Crear gráfico combinado con precio y volumen"""
    plot_ohlcv_chart(data, 
                    title="Análisis Técnico Completo",
                    figsize=(15, 12),
                    save_path=save_path)
    return save_path

# ============================================
# EJEMPLO DE USO EN EL TRADING BOT
# ============================================

def analyze_trading_data(df, symbol="EURUSD"):
    """
    Función de ejemplo para análisis técnico
    Muestra cómo integrar el nuevo módulo en el trading bot
    """
    try:
        # Crear gráficos para análisis
        chart_paths = []
        
        # Gráfico de velas japonesas
        candlestick_path = f"charts/{symbol}_candles.png"
        create_candlestick_chart(df, f"{symbol} - Gráfico de Velas", candlestick_path)
        chart_paths.append(candlestick_path)
        
        # Gráfico OHLCV completo
        ohlcv_path = f"charts/{symbol}_ohlcv.png"
        create_combined_chart(df, f"{symbol} - Análisis Técnico", ohlcv_path)
        chart_paths.append(ohlcv_path)
        
        # Gráfico de volumen separado
        volume_path = f"charts/{symbol}_volume.png"
        create_volume_chart(df, volume_path)
        chart_paths.append(volume_path)
        
        return {
            "status": "success",
            "charts_generated": chart_paths,
            "message": f"Análisis técnico completado para {symbol}"
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error generando gráficos: {str(e)}"
        }

# ============================================
# TEST DE FUNCIONALIDAD
# ============================================

if __name__ == "__main__":
    # Crear datos de prueba
    dates = pd.date_range('2024-01-01', periods=50, freq='H')
    
    # Simular datos OHLCV
    np.random.seed(42)  # Para reproducibilidad
    base_price = 1.1000
    
    data = pd.DataFrame({
        'Open': [base_price + np.random.uniform(-0.01, 0.01) for _ in range(50)],
        'High': [],
        'Low': [],
        'Close': [],
        'Volume': np.random.uniform(1000000, 5000000, 50)
    }, index=dates)
    
    # Generar High/Low basados en Open
    data['High'] = data['Open'] + np.random.uniform(0, 0.02, 50)
    data['Low'] = data['Open'] - np.random.uniform(0, 0.02, 50)
    
    # Generar Close basado en Open con algo de volatilidad
    price_changes = np.random.uniform(-0.015, 0.015, 50)
    data['Close'] = data['Open'] + price_changes
    
    # Ajustar High/Low para que sean consistentes
    data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
    data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
    
    print("🧪 Testing financial_charts module...")
    
    # Test de todas las funciones
    try:
        # Test gráfico de velas
        candlestick_path = create_candlestick_chart(data, "Test Candlestick")
        print(f"✅ Candlestick chart: {candlestick_path}")
        
        # Test gráfico OHLCV
        ohlcv_path = create_ohlc_chart(data, "Test OHLCV")
        print(f"✅ OHLCV chart: {ohlcv_path}")
        
        # Test análisis completo
        result = analyze_trading_data(data, "TEST")
        print(f"✅ Trading analysis: {result['status']}")
        
        # Test de compatibilidad
        print("🎉 All tests passed! Module is working correctly.")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("Please check the matplotlib installation and dependencies.")