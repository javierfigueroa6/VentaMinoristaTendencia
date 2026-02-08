"""
Pipeline Principal - Sistema de Clasificación de Postulantes

Este script ejecuta el pipeline completo de Machine Learning:
1. Generación de datos sintéticos
2. Preprocesamiento de datos
3. Entrenamiento del modelo CatBoost
4. Evaluación y visualización de resultados

Autor: Expertos en Data Science
"""

import sys
import os

# Agregar src al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generation import generate_candidate_data
from src.preprocessing import preprocess_pipeline
from src.model import train_model
from src.evaluation import evaluate_model
from src.prediction import generate_predictions
from src.interpretability_alternative import analyze_interpretability_alternative
import pandas as pd


def print_header(text):
    """Imprime un encabezado formateado."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def main():
    """Ejecuta el pipeline completo del proyecto."""
    
    print("\n" + "🎯"*35)
    print("  SISTEMA DE CLASIFICACIÓN DE POSTULANTES - MACHINE LEARNING")
    print("  Proyecto de Demostración de Capacidades en Data Science")
    print("🎯"*35 + "\n")
    
    try:
        # ====================================================================
        # PASO 1: GENERACIÓN DE DATOS SINTÉTICOS
        # ====================================================================
        print_header("PASO 1/4: Generación de Datos Sintéticos")
        
        df = generate_candidate_data(
            n_samples=1500,
            output_path='data/raw/candidates.csv'
        )
        
        print("\n✅ Datos generados exitosamente\n")
        
        # ====================================================================
        # PASO 2: PREPROCESAMIENTO DE DATOS
        # ====================================================================
        print_header("PASO 2/4: Preprocesamiento de Datos")
        
        # Cargar datos antes de preprocesar para obtener nombres
        df_full_data = pd.read_csv('data/raw/candidates.csv')
        
        X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
            data_path='data/raw/candidates.csv'
        )
        
        print("✅ Preprocesamiento completado exitosamente\n")
        
        # ====================================================================
        # PASO 3: ENTRENAMIENTO DEL MODELO
        # ====================================================================
        print_header("PASO 3/4: Entrenamiento del Modelo CatBoost")
        
        classifier = train_model(
            X_train, y_train,
            X_test, y_test,
            preprocessor.get_categorical_features_indices()
        )
        
        print("✅ Modelo entrenado y guardado exitosamente\n")
        
        # ====================================================================
        # PASO 4: EVALUACIÓN DEL MODELO
        # ====================================================================
        print_header("PASO 4/4: Evaluación del Modelo")
        
        evaluator = evaluate_model(
            classifier.model,
            X_test,
            y_test,
            feature_names=preprocessor.feature_names,
            preprocessor=preprocessor
        )
        
        print("✅ Evaluación completada exitosamente\n")
        
        # ====================================================================
        # PASO 5: GENERACIÓN DE SCORES DE PREDICCIÓN
        # ====================================================================
        print_header("PASO 5/6: Generación de Scores de Predicción")
        
        # Obtener nombres del conjunto de test
        # Los datos se dividen 80-20, entonces los últimos 20% son test
        total_rows = len(df_full_data)
        test_size = int(total_rows * 0.2)
        # Usar seed 42 para replicar el split
        from sklearn.model_selection import train_test_split
        train_indices, test_indices = train_test_split(
            range(total_rows), test_size=0.2, random_state=42,
            stratify=df_full_data['Target']
        )
        test_names = df_full_data.iloc[test_indices]['Nombre'].values
        
        # Generar predicciones con scores
        predictor, predictions_df = generate_predictions(
            classifier.model,
            X_test,
            preprocessor=preprocessor,
            candidate_names=test_names,
            output_path='results/candidate_predictions.csv'
        )
        
        print("✅ Scores de predicción generados y exportados exitosamente\n")
        
        # ====================================================================
        # PASO 6: ANÁLISIS DE INTERPRETABILIDAD (SIN SHAP)
        # ====================================================================
        print_header("PASO 6/6: Análisis de Interpretabilidad del Modelo")
        
        try:
            interpreter = analyze_interpretability_alternative(
                classifier.model,
                X_train,
                X_test,
                y_test=y_test,
                feature_names=preprocessor.feature_names
            )
            
            print("✅ Análisis de interpretabilidad completado exitosamente\n")
        except Exception as e:
            print(f"⚠️ Error durante análisis de interpretabilidad: {str(e)}")
            print(f"   Continuando con el resto del pipeline...\n")
        
        # ====================================================================
        # RESUMEN FINAL
        # ====================================================================
        print("\n" + "🎉"*35)
        print("  PIPELINE COMPLETADO EXITOSAMENTE")
        print("🎉"*35 + "\n")
        
        print("📁 Archivos generados:")
        print("   ├── data/raw/candidates.csv                          - Dataset sintético")
        print("   ├── data/processed/preprocessor.pkl                  - Preprocesador guardado")
        print("   ├── models/catboost_model.cbm                        - Modelo entrenado")
        print("   ├── models/catboost_model_metadata.pkl               - Metadata del modelo")
        print("   ├── results/confusion_matrix.png                     - Matriz de confusión")
        print("   ├── results/roc_curve.png                            - Curva ROC")
        print("   ├── results/feature_importance.png                   - Importancia de features")
        print("   ├── results/classification_report.txt                - Reporte detallado")
        print("   ├── results/candidate_predictions.csv                - 🆕 Scores por candidato")
        print("   ├── results/interpretability_feature_importance.png  - 🆕 Comparación de importancia")
        print("   ├── results/interpretability_partial_dependence.png  - 🆕 Partial Dependence Plots")
        print("   ├── results/interpretability_individual_*.png        - 🆕 Análisis individuales")
        print("   └── results/interpretability_report.txt              - 🆕 Reporte de interpretabilidad")
        
        print("\n💡 Próximos pasos sugeridos:")
        print("   1. Revisar las visualizaciones en la carpeta 'results/'")
        print("   2. Analizar el CSV de scores de predicción para cada candidato")
        print("   3. Estudiar los gráficos de interpretabilidad para entender decisiones")
        print("   4. Compartir resultados con la clienta")
        print("   5. Ajustar hiperparámetros si es necesario")
        print("   6. Probar con datos reales cuando estén disponibles")
        
        print("\n" + "="*70)
        print("  ¡Proyecto listo para demostración!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución del pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
