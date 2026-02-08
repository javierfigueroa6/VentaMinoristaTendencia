import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from data_loader import load_and_clean_data
import os

# Set style for plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

OUTPUT_DIR = 'retail_sales_forecast/output'
# Updated to point to the Excel file
DATA_PATH = 'retail_sales_forecast/data/ventas_chile_dummy.xlsx'

def generate_plots(df):
    """Generates and saves EDA plots."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Historical Trend (Daily)
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df['y'], label='Ventas Diarias', color='#bdc3c7', alpha=0.5)
    
    # Add Monthly Trend line
    monthly_trend = df.set_index('ds').resample('ME')['y'].mean()
    plt.plot(monthly_trend.index, monthly_trend.values, label='Tendencia Mensual (Promedio)', color='#2c3e50', linewidth=2)
    
    plt.title('Evolución Histórica de Ventas Minoristas (Chile)', fontsize=16)
    plt.ylabel('Índice de Ventas', fontsize=12)
    plt.xlabel('Fecha', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/historical_trend.png')
    plt.close()

    # 2. Seasonality (Boxplot by Month)
    df['month'] = df['ds'].dt.month_name()
    # Order months correctly
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='month', y='y', data=df, order=months_order, palette='viridis')
    plt.title('Distribución Mensual de Ventas (Estacionalidad)', fontsize=16)
    plt.ylabel('Índice de Ventas', fontsize=12)
    plt.xlabel('Mes', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/seasonality.png')
    plt.close()

    # 3. Weekly Seasonality (Day of Week)
    df['weekday'] = df['ds'].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='weekday', y='y', data=df, order=days_order, palette='magma')
    plt.title('Distribución Semanal de Ventas (Día de la Semana)', fontsize=14)
    plt.ylabel('Índice de Ventas', fontsize=12)
    plt.xlabel('Día', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/weekly_seasonality.png')
    plt.close()

def train_monthly_model(df):
    """Trains Holt-Winters model on monthly aggregated data."""
    # Resample to Monthly Average for smoother trend prediction
    ts_data = df.set_index('ds')['y'].resample('ME').mean()
    ts_data = ts_data.interpolate()
    if ts_data.index.freq is None: ts_data.index.freq = 'ME'

    # Validating on last 6 months
    train = ts_data[:-6]

    try:
        model = ExponentialSmoothing(
            train,
            seasonal_periods=12,
            trend='add',
            seasonal='add',
            damped_trend=True
        ).fit()
    except:
        model = ExponentialSmoothing(train).fit()

    # Final Model on Full Data - Ultra conservative volume level
    try:
        final_model = ExponentialSmoothing(
            ts_data,
            seasonal_periods=12,
            trend='add',
            seasonal='mul',
            damped_trend=True
        ).fit(smoothing_level=0.05, smoothing_trend=0.0, damping_trend=0.6, smoothing_seasonal=0.05) 
    except Exception as e:
        print(f"Error fitting Monthly Holt-Winters: {e}")
        final_model = ExponentialSmoothing(ts_data).fit()
    
    forecast = final_model.forecast(6)
    
    # Plotting Monthly Forecast
    plt.figure(figsize=(12, 6))
    plt.plot(ts_data.index, ts_data.values, label='Datos Históricos (Promedio Mensual)', color='#2c3e50')
    plt.plot(forecast.index, forecast.values, label='Pronóstico (Próximos 6 meses)', color='#e74c3c', linestyle='--', linewidth=2)
    
    next_month_val = forecast.iloc[0]
    next_month_date = forecast.index[0]
    plt.scatter([next_month_date], [next_month_val], color='red', zorder=5)
    plt.annotate(f"{next_month_val:.2f}", (next_month_date, next_month_val), 
                 xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold')

    plt.title('Pronóstico de Ventas Minoristas - Próximos 6 Meses', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/forecast_monthly.png')
    plt.close()
    
    return final_model, forecast, ts_data

def train_daily_model(df):
    """Trains Holt-Winters model on daily data for short-term forecast."""
    ts_data_daily = df.set_index('ds')['y']
    ts_data_daily = ts_data_daily.asfreq('D').interpolate() 

    try:
        # Weekly seasonality = 7. Ultra-conservative volume tracking.
        daily_model = ExponentialSmoothing(
            ts_data_daily,
            seasonal_periods=7,
            trend='add',
            seasonal='mul',
            damped_trend=True,
            use_boxcox=True,
            initialization_method="estimated"
        ).fit(
            smoothing_level=0.01, 
            smoothing_trend=0.0, 
            smoothing_seasonal=0.01, 
            damping_trend=0.8
        ) 
    except Exception as e:
        print(f"Error fitting Daily Holt-Winters: {e}. Trying simple.")
        daily_model = ExponentialSmoothing(ts_data_daily).fit()

    # Forecast next 30 days
    daily_forecast = daily_model.forecast(30)
    
    # --- Advanced Correction Heuristics ---
    def apply_advanced_corrections(series):
        new_series = series.copy()
        for date in new_series.index:
            # 1. Holiday Correction (Enero 1st) - Mandatory Holiday
            if date.month == 1 and date.day == 1:
                new_series.loc[date] = new_series.loc[date] * 0.15 
            
            # 2. Payday Boost (Fin de mes - 25th to 31st) - DAMPENED (max +5%)
            if date.day >= 25:
                # Slower boost
                boost_factor = 1.02 + (0.03 * ((date.day - 25) / 6.0))
                new_series.loc[date] = new_series.loc[date] * boost_factor

            # 3. Mid-month slight dip (10th to 20th) - SUBTLE
            if 10 <= date.day <= 20:
                new_series.loc[date] = new_series.loc[date] * 0.99

        return new_series

    daily_forecast = apply_advanced_corrections(daily_forecast)
    
    # Plotting Daily Forecast
    last_30_days = ts_data_daily[-30:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(last_30_days.index, last_30_days.values, label='Últimos 30 días', color='#34495e', alpha=0.7)
    plt.plot(daily_forecast.index, daily_forecast.values, label='Pronóstico (Enero 2026)', color='#27ae60', marker='o', linewidth=2)
    
    plt.title('Pronóstico Ventas Enero 2026 - Con Efectos de Feriados y Quincena', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/forecast_daily.png')
    plt.close()
    
    return daily_model, daily_forecast

def create_report(ts_data_monthly, monthly_forecast, daily_forecast):
    """Generates the markdown report with Contextual Analysis."""
    next_month_date = monthly_forecast.index[0].strftime('%B %Y')
    next_month_value = monthly_forecast.iloc[0]
    
    # Calculate trend (YoY)
    last_year_date = monthly_forecast.index[0] - pd.DateOffset(years=1)
    try:
        closest_idx = ts_data_monthly.index.get_indexer([last_year_date], method='nearest')[0]
        last_year_val = ts_data_monthly.iloc[closest_idx]
        yoy_growth = ((next_month_value - last_year_val) / last_year_val) * 100
        trend_str = f"{yoy_growth:+.2f}% vs año anterior"
    except:
        trend_str = "N/A"
        
    # --- Contextual Analysis Logic ---
    # Determine the latest full year in the data
    last_data_date = ts_data_monthly.index[-1]
    current_reporting_year = last_data_date.year
    
    # Logic:
    # If we are in December, we look at full current year vs last year.
    # If we are mid-year, we compare YTD (Jan to Last Month) vs Same Period Last Year.
    
    # Get all data for current reporting year
    cur_year_mask = ts_data_monthly.index.year == current_reporting_year
    cur_year_data = ts_data_monthly[cur_year_mask]
    cur_year_avg = cur_year_data.mean()
    
    # Get previous year data
    prev_year_mask = ts_data_monthly.index.year == (current_reporting_year - 1)
    prev_year_data = ts_data_monthly[prev_year_mask]
    prev_year_avg = prev_year_data.mean()
    
    # Calculate growth
    if prev_year_avg > 0:
        annual_growth = ((cur_year_avg - prev_year_avg) / prev_year_avg) * 100
    else:
        annual_growth = 0
        
    # Heuristic interpretation for Chile Context
    if annual_growth > 3.0:
        context_sentiment = "Crecimiento Robusto"
        context_desc = "El sector muestra un desempeño superior al promedio histórico reciente. Esto sugiere una reactivación del consumo privado en retail, apoyado posiblemente por una estabilización inflacionaria y mejora en la liquidez de los hogares tras los periodos de ajuste económico (2022-2023)."
    elif annual_growth > 0.5:
        context_sentiment = "Tendencia de Recuperación/Estabilización"
        context_desc = "Se observa una estabilización positiva en las ventas (+1-3%). Considerando el contexto país, esto indica una 'normalización' de la demanda. Tras la contracción post-pandemia, el mercado ha encontrado un nuevo piso de crecimiento moderado, alineado con las proyecciones de crecimiento del PIB tendencial."
    elif annual_growth > -2.0:
        context_sentiment = "Estancamiento / Ajuste Leve"
        context_desc = "El mercado lateraliza sin una dirección agresiva. Este comportamiento es coherente con una economía en fase de ajuste, donde la incertidumbre local o tasas de interés aún restrictivas podrían estar frenando un mayor impulso en el gasto de supermercados (bienes no durables)."
    else:
        context_sentiment = "Contracción del Consumo"
        context_desc = "Se evidencia una caída sistemática en el índice real de ventas. Esto alerta sobre un debilitamiento del poder adquisitivo o un cambio en los patrones de consumo hacia canastas más básicas, reflejando probablemente debilidad en el mercado laboral o persistencia inflacionaria en alimentos."

    report_content = f"""# Reporte de Pronóstico de Ventas Minoristas (Chile)

**Fecha de Generación:** {pd.Timestamp.now().strftime('%d-%m-%Y')}

## 1. Resumen Ejecutivo
El modelo de pronóstico indica que para el próximo mes (**{next_month_date}**), se espera un índice promedio de ventas de **{next_month_value:.2f}**.
Esto representa una variación de **{trend_str}**.

## 2. Análisis del Contexto Económico (Novedad)
**Diagnóstico: {context_sentiment}**

Al analizar el desempeño acumulado del año **{current_reporting_year}** en comparación con **{current_reporting_year-1}**, observamos una variación promedio del **{annual_growth:+.2f}%**.

**Interpretación:**
{context_desc}

*Nota: Este análisis se basa en el Índice de Ventas a Precios Constantes (descontando inflación), por lo que refleja el volumen real de ventas y no solo el aumento de precios.*

## 3. Análisis Detallado (Próximos 7 Días)
A continuación se presenta la proyección de venta diaria para la próxima semana, considerando el comportamiento semanal reciente.

![Pronóstico Diario](forecast_daily.png)

| Fecha | Día | Predicción de Venta |
|-------|-----|-------------------|
"""
    for date, value in daily_forecast.items():
        day_name = date.day_name()
        report_content += f"| {date.strftime('%d-%m-%Y')} | {day_name} | **{value:.2f}** |\n"

    report_content += f"""
### Análisis de Variación Semanal
Se ha detectado un patrón semanal en las ventas:
![Patrón Semanal](weekly_seasonality.png)

## 4. Análisis de Tendencia (Largo Plazo)
La serie temporal promediada mensualmente muestra la siguiente evolución:
![Tendencia Mensual](forecast_monthly.png)

### Histórico vs Estacionalidad Anual
![Tendencia Histórica](historical_trend.png)
![Estacionalidad Mensual](seasonality.png)

## 5. Proyección Mensual (Próximos 6 Meses)
| Fecha | Índice Predicho (Promedio Mensual) |
|-------|-----------------|
"""
    for date, value in monthly_forecast.items():
        report_content += f"| {date.strftime('%Y-%m-%d')} | {value:.2f} |\n"
    
    report_content += """
## 6. Ficha Técnica y Metodología

### Arquitectura del Modelo
El pronóstico se genera mediante un modelo de **Suavizamiento Exponencial Triple (Holt-Winters)** con las siguientes características:
- **Estacionalidad Multiplicativa**: Los picos de venta se calculan de forma proporcional al nivel del mes, evitando distorsiones por volumen.
- **Tendencia Amortiguada (Damped Trend)**: Se aplica un factor de amortiguación para evitar proyecciones de crecimiento irreal a largo plazo.
- **Granularidad Dual**: El sistema entrena un modelo mensual para captar la tendencia macro y un modelo diario para captar el patrón semanal (Lunes-Domingo).

### Heurísticas de Realismo Retail
Para asegurar que el modelo refleje la realidad del mercado chileno, se aplican las siguientes reglas de negocio post-modelo:
1. **Efecto "Fin de Mes" (Payday)**: Incremento progresivo entre los días 25 y 31 (pago de remuneraciones).
2. **Feriados Irrenunciables**: Corrección manual del 1 de Enero (caída del 85% por cierre legal).
3. **Precios Constantes**: Todos los cálculos se realizan sobre el índice real, eliminando el ruido provocado por la inflación.

---
*Este reporte fue generado automáticamente por el sistema de predicción Antigravity.*
"""
    
    with open(f'{OUTPUT_DIR}/reporte_ventas.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Report generated at {OUTPUT_DIR}/reporte_ventas.md")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Loading data...")
    df = load_and_clean_data(DATA_PATH)
    print(f"Loaded {len(df)} rows.")
    
    print("Generating EDA plots...")
    generate_plots(df)
    
    print("Training Monthly Model...")
    _, monthly_forecast, ts_monthly = train_monthly_model(df)
    
    print("Training Daily Model...")
    _, daily_forecast = train_daily_model(df)
    
    print("Creating report...")
    create_report(ts_monthly, monthly_forecast, daily_forecast)
    print("Done!")
