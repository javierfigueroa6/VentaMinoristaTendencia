"""
Módulo de predicción para generar scores de candidatos.

Este módulo proporciona funcionalidad para generar predicciones con scores
de probabilidad para cada candidato y exportarlos en formato CSV.
"""

import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier


class CandidatePredictor:
    """Clase para generar predicciones con scores de candidatos."""
    
    def __init__(self, model, preprocessor=None):
        """
        Inicializa el predictor.
        
        Args:
            model: Modelo CatBoost entrenado
            preprocessor: Preprocesador para decodificar labels
        """
        self.model = model
        self.preprocessor = preprocessor
    
    def predict_with_scores(self, X, candidate_names=None):
        """
        Genera predicciones con scores de probabilidad.
        
        Args:
            X: Features de los candidatos
            candidate_names: Lista con nombres de candidatos (opcional)
            
        Returns:
            DataFrame con scores de predicción
        """
        # Obtener probabilidades
        probabilities = self.model.predict_proba(X)
        
        # Obtener predicciones
        predictions = self.model.predict(X)
        
        # Decodificar predicciones si hay preprocessor
        if self.preprocessor and hasattr(self.preprocessor, 'label_encoders'):
            if 'Target' in self.preprocessor.label_encoders:
                predicted_labels = self.preprocessor.label_encoders['Target'].inverse_transform(predictions)
            else:
                predicted_labels = predictions
        else:
            predicted_labels = predictions
        
        # Calcular confianza (probabilidad máxima)
        confidence = np.max(probabilities, axis=1)
        
        # Crear DataFrame con resultados
        results = pd.DataFrame({
            'Probabilidad_Rechazado': probabilities[:, 1],
            'Probabilidad_Aceptado': probabilities[:, 0],
            'Predicción': predicted_labels,
            'Confianza': confidence * 100  # Convertir a porcentaje
        })
        
        # Agregar nombres si están disponibles
        if candidate_names is not None:
            results.insert(0, 'Nombre', candidate_names)
        else:
            results.insert(0, 'ID_Candidato', range(1, len(results) + 1))
        
        return results
    
    def export_predictions(self, X, output_path='results/candidate_predictions.csv', 
                          candidate_names=None, include_features=False):
        """
        Exporta predicciones con scores a archivo CSV.
        
        Args:
            X: Features de los candidatos
            output_path (str): Ruta donde guardar el CSV
            candidate_names: Lista con nombres de candidatos (opcional)
            include_features (bool): Si incluir features originales en el export
            
        Returns:
            DataFrame con las predicciones
        """
        # Generar predicciones
        predictions_df = self.predict_with_scores(X, candidate_names)
        
        # Si se solicita, agregar features originales
        if include_features:
            if isinstance(X, pd.DataFrame):
                features_df = X.reset_index(drop=True)
            else:
                features_df = pd.DataFrame(X)
            
            predictions_df = pd.concat([predictions_df, features_df], axis=1)
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Exportar a CSV
        predictions_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ Predicciones exportadas a: {output_path}")
        print(f"   - Total de candidatos: {len(predictions_df)}")
        print(f"   - Columnas: {', '.join(predictions_df.columns[:5])}")
        
        return predictions_df
    
    def get_prediction_summary(self, X, candidate_names=None):
        """
        Genera un resumen estadístico de las predicciones.
        
        Args:
            X: Features de los candidatos
            candidate_names: Lista con nombres de candidatos (opcional)
            
        Returns:
            dict: Diccionario con estadísticas
        """
        predictions_df = self.predict_with_scores(X, candidate_names)
        
        summary = {
            'total_candidatos': len(predictions_df),
            'aceptados': int((predictions_df['Predicción'] == 'Aceptado').sum()),
            'rechazados': int((predictions_df['Predicción'] == 'Rechazado').sum()),
            'confianza_promedio': float(predictions_df['Confianza'].mean()),
            'confianza_min': float(predictions_df['Confianza'].min()),
            'confianza_max': float(predictions_df['Confianza'].max()),
            'prob_aceptado_promedio': float(predictions_df['Probabilidad_Aceptado'].mean()),
        }
        
        return summary
    
    def print_summary(self, X, candidate_names=None):
        """
        Imprime un resumen de las predicciones en consola.
        
        Args:
            X: Features de los candidatos
            candidate_names: Lista con nombres de candidatos (opcional)
        """
        summary = self.get_prediction_summary(X, candidate_names)
        
        print("\n" + "="*60)
        print("📊 RESUMEN DE PREDICCIONES")
        print("="*60)
        print(f"Total de candidatos evaluados: {summary['total_candidatos']}")
        print(f"  ✅ Aceptados:  {summary['aceptados']} ({summary['aceptados']/summary['total_candidatos']*100:.1f}%)")
        print(f"  ❌ Rechazados: {summary['rechazados']} ({summary['rechazados']/summary['total_candidatos']*100:.1f}%)")
        print(f"\nNivel de Confianza:")
        print(f"  Promedio: {summary['confianza_promedio']:.2f}%")
        print(f"  Mínimo:   {summary['confianza_min']:.2f}%")
        print(f"  Máximo:   {summary['confianza_max']:.2f}%")
        print(f"\nProbabilidad promedio de aceptación: {summary['prob_aceptado_promedio']:.2%}")
        print("="*60 + "\n")


def generate_predictions(model, X_test, preprocessor=None, 
                         candidate_names=None, output_path='results/candidate_predictions.csv'):
    """
    Pipeline para generar y exportar predicciones con scores.
    
    Args:
        model: Modelo entrenado
        X_test: Features de test
        preprocessor: Preprocesador (opcional)
        candidate_names: Nombres de candidatos (opcional)
        output_path: Ruta donde guardar CSV
        
    Returns:
        tuple: (predictor, predictions_df)
    """
    print("🔄 Generando scores de predicción para candidatos...\n")
    
    # Crear predictor
    predictor = CandidatePredictor(model, preprocessor)
    
    # Mostrar resumen
    predictor.print_summary(X_test, candidate_names)
    
    # Exportar predicciones
    predictions_df = predictor.export_predictions(X_test, output_path, candidate_names)
    
    # Mostrar ejemplos
    print("\n📝 Primeros 5 candidatos:")
    print(predictions_df.head().to_string(index=False))
    print(f"\n... y {len(predictions_df) - 5} candidatos más.\n")
    
    return predictor, predictions_df


if __name__ == "__main__":
    print("Este módulo debe ser importado. Ejecuta main.py para el pipeline completo.")
