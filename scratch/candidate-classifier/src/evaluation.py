"""
Módulo de evaluación del modelo de clasificación de postulantes.

Implementa métricas, visualizaciones y reportes de performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
)
import os


# Configurar estilo de visualizaciones
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """Clase para evaluar el modelo de clasificación."""
    
    def __init__(self, model, X_test, y_test, preprocessor=None):
        """
        Inicializa el evaluador.
        
        Args:
            model: Modelo entrenado
            X_test: Features de prueba
            y_test: Target de prueba
            preprocessor: Preprocesador para obtener nombres de clases
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.preprocessor = preprocessor
        self.y_pred = None
        self.y_pred_proba = None
        self.results_dir = 'results'
        
        # Crear directorio de resultados si no existe
        os.makedirs(self.results_dir, exist_ok=True)
    
    def predict(self):
        """Genera predicciones sobre el conjunto de prueba."""
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)
        print("✅ Predicciones generadas")
    
    def calculate_metrics(self):
        """
        Calcula métricas de evaluación.
        
        Returns:
            dict: Diccionario con las métricas
        """
        if self.y_pred is None:
            self.predict()
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred, average='binary'),
            'recall': recall_score(self.y_test, self.y_pred, average='binary'),
            'f1_score': f1_score(self.y_test, self.y_pred, average='binary'),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba[:, 1])
        }
        
        return metrics
    
    def print_metrics(self):
        """Imprime las métricas de evaluación en consola."""
        if self.y_pred is None:
            self.predict()
        
        metrics = self.calculate_metrics()
        
        print("\n" + "="*60)
        print("📊 MÉTRICAS DE EVALUACIÓN DEL MODELO")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print("="*60 + "\n")
        
        return metrics
    
    def plot_confusion_matrix(self, save=True):
        """
        Genera y muestra la matriz de confusión.
        
        Args:
            save (bool): Si guardar la figura
        """
        if self.y_pred is None:
            self.predict()
        
        # Calcular matriz de confusión
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Obtener nombres de clases
        if self.preprocessor:
            class_names = self.preprocessor.label_encoders['Target'].classes_
        else:
            class_names = ['Clase 0', 'Clase 1']
        
        # Crear visualización
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Cantidad'})
        plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
        plt.ylabel('Valor Real', fontsize=12)
        plt.xlabel('Predicción', fontsize=12)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_dir, 'confusion_matrix.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Matriz de confusión guardada en: {filepath}")
        
        plt.close()
    
    def plot_roc_curve(self, save=True):
        """
        Genera la curva ROC.
        
        Args:
            save (bool): Si guardar la figura
        """
        if self.y_pred_proba is None:
            self.predict()
        
        # Calcular curva ROC
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Crear visualización
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Curva ROC (Receiver Operating Characteristic)', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_dir, 'roc_curve.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Curva ROC guardada en: {filepath}")
        
        plt.close()
    
    def plot_feature_importance(self, feature_names=None, top_n=10, save=True):
        """
        Visualiza la importancia de las features.
        
        Args:
            feature_names (list): Nombres de las features
            top_n (int): Número de features más importantes a mostrar
            save (bool): Si guardar la figura
        """
        # Obtener importancia de features
        if hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
        else:
            print("⚠️ El modelo no soporta feature importance")
            return
        
        # Preparar datos
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        # Crear DataFrame para ordenar
        import pandas as pd
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(top_n)
        
        # Crear visualización
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("viridis", len(df_importance))
        bars = plt.barh(range(len(df_importance)), df_importance['Importance'], color=colors)
        plt.yticks(range(len(df_importance)), df_importance['Feature'])
        plt.xlabel('Importancia', fontsize=12)
        plt.title(f'Top {top_n} Features Más Importantes', 
                 fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_dir, 'feature_importance.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Importancia de features guardada en: {filepath}")
        
        plt.close()
    
    def generate_classification_report(self, save=True):
        """
        Genera un reporte de clasificación detallado.
        
        Args:
            save (bool): Si guardar el reporte en archivo
        """
        if self.y_pred is None:
            self.predict()
        
        # Obtener nombres de clases
        if self.preprocessor:
            target_names = self.preprocessor.label_encoders['Target'].classes_
        else:
            target_names = None
        
        # Generar reporte
        report = classification_report(self.y_test, self.y_pred, 
                                       target_names=target_names, digits=4)
        
        print("\n" + "="*60)
        print("📋 REPORTE DE CLASIFICACIÓN DETALLADO")
        print("="*60)
        print(report)
        print("="*60 + "\n")
        
        if save:
            filepath = os.path.join(self.results_dir, 'classification_report.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("REPORTE DE CLASIFICACIÓN - Modelo de Postulantes\n")
                f.write("="*60 + "\n\n")
                f.write(report)
                f.write("\n" + "="*60 + "\n")
            print(f"✅ Reporte guardado en: {filepath}")
    
    def evaluate_full(self, feature_names=None):
        """
        Ejecuta evaluación completa del modelo.
        
        Args:
            feature_names (list): Nombres de las features
        """
        print("\n🔍 Iniciando evaluación completa del modelo...")
        
        # Generar predicciones
        self.predict()
        
        # Mostrar métricas
        self.print_metrics()
        
        # Generar visualizaciones
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_feature_importance(feature_names)
        
        # Generar reporte
        self.generate_classification_report()
        
        print(f"\n✅ Evaluación completa finalizada. Resultados en: {self.results_dir}/\n")


def evaluate_model(model, X_test, y_test, feature_names=None, preprocessor=None):
    """
    Pipeline de evaluación del modelo.
    
    Args:
        model: Modelo entrenado
        X_test: Features de prueba
        y_test: Target de prueba
        feature_names (list): Nombres de las features
        preprocessor: Preprocesador con encoders
        
    Returns:
        ModelEvaluator: Evaluador con resultados
    """
    evaluator = ModelEvaluator(model, X_test, y_test, preprocessor)
    evaluator.evaluate_full(feature_names)
    
    return evaluator


if __name__ == "__main__":
    print("Este módulo debe ser importado. Ejecuta main.py para el pipeline completo.")
