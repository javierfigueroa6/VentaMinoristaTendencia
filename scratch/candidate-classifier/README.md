# Sistema de Clasificación de Postulantes 🎯

Proyecto profesional de Machine Learning para clasificar candidatos laborales usando CatBoost.

## 📋 Descripción del Proyecto

Este proyecto demuestra capacidades avanzadas en data science, desarrollando un modelo de clasificación inteligente que evalúa postulantes de trabajo basándose en información de sus CVs. Utiliza el algoritmo CatBoost, optimizado para manejar variables categóricas de forma eficiente.

## 🎯 Objetivo

Clasificar automáticamente a postulantes como **Aceptado** o **Rechazado** utilizando características extraídas de sus CVs, incluyendo:
- Información demográfica
- Formación académica
- Competencias técnicas
- Disponibilidad laboral

## 🗂️ Estructura del Proyecto

```
candidate-classifier/
├── data/
│   ├── raw/              # Datos sintéticos generados
│   └── processed/        # Datos preprocesados para entrenamiento
├── src/
│   ├── data_generation.py    # Generación de dataset sintético
│   ├── preprocessing.py      # Preprocesamiento y feature engineering
│   ├── model.py             # Entrenamiento del modelo CatBoost
│   └── evaluation.py        # Evaluación y métricas del modelo
├── models/              # Modelos entrenados guardados
├── results/             # Visualizaciones y reportes
├── notebooks/           # Notebooks de exploración (opcional)
├── requirements.txt     # Dependencias del proyecto
├── README.md           # Este archivo
└── main.py             # Pipeline completo de ejecución
```

## 📊 Dataset

El dataset sintético contiene las siguientes características:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| Nombre | Texto | Nombre completo del postulante |
| Edad | Numérico | Edad del candidato (22-65 años) |
| Título Profesional | Categórico | Título académico obtenido |
| Universidad/Instituto | Categórico | Institución educativa |
| Palabras Clave | Numérico | Número de keywords relevantes al puesto (0-20) |
| Comuna | Categórico | Ubicación geográfica |
| Presencial | Binario | Disponibilidad para trabajo presencial (Si/No) |
| Magíster | Binario | Posee estudios de postgrado (Si/No) |
| **Target** | Binario | **Clasificación: Aceptado/Rechazado** |

## 🚀 Instalación

### Requisitos previos
- Python 3.8 o superior
- pip instalado

### Pasos de instalación

```bash
# Clonar o descargar el proyecto
cd candidate-classifier

# Instalar dependencias
pip install -r requirements.txt
```

## 💻 Uso

### Ejecución completa del pipeline

```bash
python main.py
```

Este comando ejecutará:
1. ✅ Generación de datos sintéticos
2. ✅ Preprocesamiento de datos
3. ✅ Entrenamiento del modelo CatBoost
4. ✅ Evaluación del modelo
5. ✅ Generación de visualizaciones y reportes

### Ejecución por módulos

```python
# Generar datos
from src.data_generation import generate_candidate_data
generate_candidate_data(n_samples=1500)

# Entrenar modelo
from src.model import train_model
model = train_model()

# Evaluar modelo
from src.evaluation import evaluate_model
evaluate_model(model, X_test, y_test)
```

## 📈 Modelo: CatBoost

**CatBoost** (Categorical Boosting) es un algoritmo de gradient boosting desarrollado por Yandex, optimizado para:
- ✅ Manejo nativo de variables categóricas
- ✅ Alta precisión con configuración por defecto
- ✅ Prevención de overfitting
- ✅ Velocidad de entrenamiento

### Ventajas para este proyecto
- No requiere encoding manual de todas las variables categóricas
- Excelente performance en datasets pequeños/medianos
- Interpretabilidad mediante feature importance

## 📊 Métricas de Evaluación

El modelo es evaluado usando:
- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión en predicciones positivas
- **Recall**: Cobertura de casos positivos
- **F1-Score**: Balance entre precision y recall
- **ROC-AUC**: Área bajo la curva ROC
- **Matriz de Confusión**: Visualización de predicciones

## 📁 Resultados

Después de ejecutar el pipeline, encontrarás en `results/`:
- `confusion_matrix.png`: Matriz de confusión
- `feature_importance.png`: Importancia de características
- `roc_curve.png`: Curva ROC
- `classification_report.txt`: Reporte detallado de métricas

## 🔧 Tecnologías Utilizadas

- **Python 3.8+**: Lenguaje principal
- **Pandas**: Manipulación de datos
- **NumPy**: Computación numérica
- **CatBoost**: Modelo de clasificación
- **Scikit-learn**: Preprocesamiento y métricas
- **Matplotlib/Seaborn**: Visualizaciones
- **Faker**: Generación de datos sintéticos

## 👥 Equipo

Desarrollado por expertos en Data Science y Machine Learning, especializados en:
- Modelos predictivos de clasificación y regresión
- Feature engineering y preprocesamiento
- Evaluación y optimización de modelos
- Despliegue de soluciones ML en producción

## 📝 Notas

- Los datos son completamente sintéticos para propósitos de demostración
- El modelo puede ser reentrenado con datos reales
- La estructura del proyecto sigue mejores prácticas de la industria

## 🔮 Próximos Pasos

- [ ] Optimización de hiperparámetros con GridSearch/Optuna
- [ ] Implementación de validación cruzada estratificada
- [ ] API REST para predicciones en tiempo real
- [ ] Dashboard interactivo con Streamlit/Dash
- [ ] Despliegue en la nube (AWS/Azure/GCP)

---

**¿Preguntas o sugerencias?** Este proyecto está diseñado para demostrar capacidades profesionales en data science y puede ser adaptado a casos de uso reales.
