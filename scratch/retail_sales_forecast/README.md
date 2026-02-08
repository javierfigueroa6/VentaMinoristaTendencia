# 🛒 Retail Sales Forecast Chile

Este proyecto es una herramienta profesional de análisis y pronóstico de ventas minoristas para supermercados en Chile, utilizando el **Índice de Ventas a Precios Constantes**.

## 🚀 Cómo Empezar

### 1. Requisitos Previos
Asegúrate de tener Python 3.8+ instalado.

### 2. Instalación de Dependencias
Ejecuta el siguiente comando para instalar las librerías necesarias:
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el Dashboard
Para iniciar la interfaz interactiva de Streamlit:
```bash
streamlit run src/app.py
```

### 4. Generar Reporte Estático
Si solo deseas entrenar los modelos y generar el reporte en Markdown y los gráficos base:
```bash
python src/model_and_report.py
```

## 📁 Estructura del Proyecto
- `data/`: Contiene el archivo Excel con los datos históricos.
- `src/`: Código fuente del dashboard (`app.py`), carga de datos (`data_loader.py`) y modelamiento (`model_and_report.py`).
- `output/`: Reportes generados y gráficos exportados.
- `requirements.txt`: Lista de librerías de Python necesarias.

## ⚙️ Metodología
El sistema utiliza una arquitectura **Holt-Winters (Triple Smoothing)** con tendencia amortiguada y estacionalidad multiplicativa, ajustada con heurísticas específicas para el retail chileno (Feriados, Efecto Quincena, etc.).

---
*Desarrollado con Antigravity.*
