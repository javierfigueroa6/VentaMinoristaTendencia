"""
Módulo de interpretabilidad del modelo SIN dependencias de SHAP.

Usa solo scikit-learn y CatBoost para análisis de interpretabilidad.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from typing import Optional, List


# Configurar estilo de visualizaciones
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ModelInterpreterAlternative:
    """Clase para análisis de interpretabilidad usando métodos nativos."""
    
    def __init__(self, model, X_train, X_test=None, y_test=None, feature_names=None):
        """
        Inicializa el intérprete del modelo.
        
        Args:
            model: Modelo CatBoost entrenado
            X_train: Datos de entrenamiento
            X_test: Datos de test para análisis (opcional)
            y_test: Target de test (opcional)
            feature_names: Nombres de las features
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test if X_test is not None else X_train
        self.y_test = y_test
        self.feature_names = feature_names if feature_names else [f'Feature_{i}' for i in range(X_train.shape[1])]
        self.results_dir = 'results'
        self.perm_importance = None
        
        # Crear directorio de resultados si no existe
        os.makedirs(self.results_dir, exist_ok=True)
    
    def calculate_permutation_importance(self, n_repeats=10):
        """
        Calcula Permutation Importance (más robusta que feature importance estándar).
        
        Args:
            n_repeats (int): Número de repeticiones para permutación
        """
        print("🔄 Calculando Permutation Importance...")
        print(f"   (Esto puede tomar unos minutos)\n")
        
        if self.y_test is None:
            print("⚠️ No hay y_test disponible, usando conjunto de entrenamiento")
            X_eval = self.X_train
            y_eval = self.model.predict(self.X_train)
        else:
            X_eval = self.X_test
            y_eval = self.y_test
        
        # Calcular permutation importance
        self.perm_importance = permutation_importance(
            self.model, X_eval, y_eval,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=1
        )
        
        print("✅ Permutation Importance calculada exitosamente\n")
        
        return self.perm_importance
    
    def plot_feature_importance_comparison(self, save=True):
        """
        Compara Feature Importance nativa vs Permutation Importance.
        
        Args:
            save (bool): Si guardar la figura
        """
        print("📊 Generando comparación de importancia de features...")
        
        # Obtener importancia nativa de CatBoost
        native_importance = self.model.get_feature_importance()
        
        # Calcular permutation importance si no existe
        if self.perm_importance is None:
            self.calculate_permutation_importance()
        
        # Crear DataFrame para comparación
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'CatBoost_Importance': native_importance,
            'Permutation_Importance': self.perm_importance.importances_mean
        }).sort_values('Permutation_Importance', ascending=False)
        
        # Normalizar para comparación visual
        importance_df['CatBoost_Normalized'] = (
            importance_df['CatBoost_Importance'] / importance_df['CatBoost_Importance'].max()
        )
        importance_df['Permutation_Normalized'] = (
            importance_df['Permutation_Importance'] / importance_df['Permutation_Importance'].max()
        )
        
        # Crear visualización
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: CatBoost Importance
        colors1 = sns.color_palette("viridis", len(importance_df))
        ax1.barh(range(len(importance_df)), importance_df['CatBoost_Normalized'], color=colors1)
        ax1.set_yticks(range(len(importance_df)))
        ax1.set_yticklabels(importance_df['Feature'])
        ax1.set_xlabel('Importancia Normalizada', fontsize=12)
        ax1.set_title('Feature Importance (CatBoost Nativo)', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Permutation Importance con barras de error
        colors2 = sns.color_palette("rocket", len(importance_df))
        ax2.barh(range(len(importance_df)), importance_df['Permutation_Normalized'], 
                xerr=self.perm_importance.importances_std / self.perm_importance.importances_mean.max(),
                color=colors2, alpha=0.8)
        ax2.set_yticks(range(len(importance_df)))
        ax2.set_yticklabels(importance_df['Feature'])
        ax2.set_xlabel('Importancia Normalizada', fontsize=12)
        ax2.set_title('Permutation Importance (Más Robusta)', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_dir, 'interpretability_feature_importance.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Comparación de importancia guardada en: {filepath}\n")
        
        plt.close()
        
        return importance_df
    
    def plot_partial_dependence(self, top_n=4, save=True):
        """
        Genera Partial Dependence Plots para las features más importantes.
        
        Args:
            top_n (int): Número de top features a analizar
            save (bool): Si guardar la figura
        """
        print(f"📊 Generando Partial Dependence Plots para top {top_n} features...")
        
        # Calcular permutation importance si no existe
        if self.perm_importance is None:
            self.calculate_permutation_importance()
        
        # Obtener top N features
        top_indices = np.argsort(self.perm_importance.importances_mean)[-top_n:][::-1]
        top_features = [self.feature_names[i] for i in top_indices]
        
        print(f"   Analizando: {', '.join(top_features)}")
        
        # Crear figura
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, (feature_idx, feature_name) in enumerate(zip(top_indices, top_features)):
            if idx >= 4:  # Máximo 4 plots
                break
            
            try:
                # Crear partial dependence plot
                display = PartialDependenceDisplay.from_estimator(
                    self.model,
                    self.X_test,
                    [feature_idx],
                    feature_names=self.feature_names,
                    ax=axes[idx],
                    grid_resolution=50
                )
                
                axes[idx].set_title(f'Dependencia Parcial: {feature_name}', 
                                   fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(feature_name, fontsize=11)
                axes[idx].set_ylabel('Efecto en Predicción', fontsize=11)
                axes[idx].grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"   ⚠️ No se pudo generar plot para {feature_name}: {str(e)}")
                axes[idx].text(0.5, 0.5, f'No disponible\n({feature_name})', 
                             ha='center', va='center', fontsize=10)
                axes[idx].set_title(f'{feature_name} (No disponible)', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_dir, 'interpretability_partial_dependence.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Partial Dependence Plots guardados en: {filepath}\n")
        
        plt.close()
    
    def analyze_individual_predictions(self, n_samples=3, save=True):
        """
        Analiza contribuciones de features para predicciones individuales.
        
        Args:
            n_samples (int): Número de muestras a analizar
            save (bool): Si guardar las figuras
        """
        print(f"📊 Analizando {n_samples} predicciones individuales...")
        
        # Seleccionar muestras aleatorias
        sample_indices = np.random.choice(len(self.X_test), min(n_samples, len(self.X_test)), replace=False)
        
        for idx, sample_idx in enumerate(sample_indices):
            # Obtener la muestra
            X_sample = self.X_test.iloc[[sample_idx]] if isinstance(self.X_test, pd.DataFrame) else self.X_test[[sample_idx]]
            
            # Obtener feature importance para esta predicción individual
            # Usar feature importance como proxy (CatBoost no tiene método exacto sin SHAP)
            feature_importance = self.model.get_feature_importance()
            
            # Obtener valores de features
            feature_values = X_sample.values[0] if isinstance(X_sample, pd.DataFrame) else X_sample[0]
            
            # Crear DataFrame con feature: valor, importancia
            individual_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Valor': feature_values,
                'Importancia_Global': feature_importance
            }).sort_values('Importancia_Global', ascending=False).head(10)
            
            # Obtener predicción
            prediction = self.model.predict(X_sample)[0]
            prediction_proba = self.model.predict_proba(X_sample)[0]
            
            # Crear visualización
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Valores de features
            colors = sns.color_palette("coolwarm", len(individual_df))
            ax1.barh(range(len(individual_df)), individual_df['Importancia_Global'], color=colors)
            ax1.set_yticks(range(len(individual_df)))
            ax1.set_yticklabels([f"{feat}" for feat in individual_df['Feature']])
            ax1.set_xlabel('Importancia Global', fontsize=11)
            ax1.set_title(f'Top 10 Features Importantes\nCandidato #{sample_idx}', 
                         fontsize=13, fontweight='bold')
            ax1.invert_yaxis()
            ax1.grid(axis='x', alpha=0.3)
            
            # Plot 2: Tabla con valores
            ax2.axis('off')
            table_data = []
            for _, row in individual_df.iterrows():
                val = row['Valor']
                # Formatear valor según tipo
                if isinstance(val, (int, np.integer)):
                    val_str = str(val)
                elif isinstance(val, (float, np.floating)):
                    val_str = f"{val:.2f}"
                else:
                    val_str = str(val)[:20]  # Truncar strings largos
                
                table_data.append([row['Feature'], val_str, f"{row['Importancia_Global']:.1f}"])
            
            table = ax2.table(
                cellText=table_data,
                colLabels=['Feature', 'Valor', 'Importancia'],
                cellLoc='left',
                loc='center',
                colWidths=[0.4, 0.3, 0.3]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Estilo de tabla
            for i in range(len(individual_df) + 1):
                if i == 0:
                    table[(i, 0)].set_facecolor('#40466e')
                    table[(i, 1)].set_facecolor('#40466e')
                    table[(i, 2)].set_facecolor('#40466e')
                    table[(i, 0)].set_text_props(weight='bold', color='white')
                    table[(i, 1)].set_text_props(weight='bold', color='white')
                    table[(i, 2)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, 0)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    table[(i, 1)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    table[(i, 2)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            
            # Título con predicción
            pred_class = "Aceptado" if prediction == 1 else "Rechazado"
            prob = prediction_proba[1] if prediction == 1 else prediction_proba[0]
            ax2.set_title(f'Valores de Features\nPredicción: {pred_class} ({prob*100:.1f}% confianza)', 
                         fontsize=13, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            if save:
                filepath = os.path.join(self.results_dir, f'interpretability_individual_{idx}.png')
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"✅ Análisis individual {idx+1} guardado en: {filepath}")
            
            plt.close()
        
        print()
    
    def generate_interpretation_report(self, save=True):
        """
        Genera un reporte de texto con insights.
        
        Args:
            save (bool): Si guardar el reporte
        """
        if self.perm_importance is None:
            self.calculate_permutation_importance()
        
        # Ordenar features por importancia
        sorted_idx = np.argsort(self.perm_importance.importances_mean)[::-1]
        
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("REPORTE DE INTERPRETABILIDAD DEL MODELO")
        report_lines.append("Métodos: Permutation Importance & Partial Dependence")
        report_lines.append("="*70)
        report_lines.append("")
        report_lines.append("📊 IMPORTANCIA DE FEATURES (Permutation Importance)")
        report_lines.append("-"*70)
        
        for i, idx in enumerate(sorted_idx[:10], 1):
            fname = self.feature_names[idx]
            importance = self.perm_importance.importances_mean[idx]
            std = self.perm_importance.importances_std[idx]
            report_lines.append(
                f"{i:2d}. {fname:30s} | Importancia: {importance:.4f} (±{std:.4f})"
            )
        
        report_lines.append("")
        report_lines.append("="*70)
        report_lines.append("INTERPRETACIÓN:")
        report_lines.append("="*70)
        report_lines.append("")
        report_lines.append("Permutation Importance mide cuánto baja el rendimiento del modelo")
        report_lines.append("cuando se permutan aleatoriamente los valores de cada feature.")
        report_lines.append("")
        report_lines.append("- Valores altos → Feature muy importante para el modelo")
        report_lines.append("- Valores bajos → Feature poco relevante")
        report_lines.append("- Barras de error → Variabilidad/estabilidad de la importancia")
        report_lines.append("")
        report_lines.append("Visualizaciones generadas:")
        report_lines.append("  • interpretability_feature_importance.png - Comparación de métodos")
        report_lines.append("  • interpretability_partial_dependence.png - Efecto de features")
        report_lines.append("  • interpretability_individual_*.png - Análisis de casos")
        report_lines.append("")
        report_lines.append("="*70)
        
        # Imprimir en consola
        report_text = "\n".join(report_lines)
        print("\n" + report_text + "\n")
        
        if save:
            filepath = os.path.join(self.results_dir, 'interpretability_report.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✅ Reporte de interpretabilidad guardado en: {filepath}\n")
    
    def analyze_full(self):
        """Ejecuta análisis completo de interpretabilidad."""
        print("\n🔍 Iniciando análisis completo de interpretabilidad...\n")
        
        # Calcular importancia
        self.calculate_permutation_importance(n_repeats=10)
        
        # Generar todas las visualizaciones
        self.plot_feature_importance_comparison()
        self.plot_partial_dependence(top_n=4)
        self.analyze_individual_predictions(n_samples=3)
        
        # Generar reporte
        self.generate_interpretation_report()
        
        print(f"✅ Análisis de interpretabilidad completo. Resultados en: {self.results_dir}/\n")


def analyze_interpretability_alternative(model, X_train, X_test, y_test=None, feature_names=None):
    """
    Pipeline para análisis de interpretabilidad SIN SHAP.
    
    Args:
        model: Modelo entrenado
        X_train: Datos de entrenamiento
        X_test: Datos de test
        y_test: Target de test (opcional)
        feature_names: Nombres de features
        
    Returns:
        ModelInterpreterAlternative: Intérprete con análisis completo
    """
    interpreter = ModelInterpreterAlternative(model, X_train, X_test, y_test, feature_names)
    interpreter.analyze_full()
    
    return interpreter


if __name__ == "__main__":
    print("Este módulo debe ser importado. Ejecuta main.py para el pipeline completo.")
