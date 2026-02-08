"""
Módulo de interpretabilidad del modelo usando SHAP values.

Este módulo proporciona herramientas para explicar las predicciones del modelo
usando SHAP (SHapley Additive exPlanations) values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional, List

# SHAP will be imported lazily when needed
# This allows the module to be imported even if SHAP is not installed


# Configurar estilo de visualizaciones
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ModelInterpreter:
    """Clase para análisis de interpretabilidad del modelo usando SHAP."""
    
    def __init__(self, model, X_train, X_test=None, feature_names=None):
        """
        Inicializa el intérprete del modelo.
        
        Args:
            model: Modelo CatBoost entrenado
            X_train: Datos de entrenamiento para background dataset
            X_test: Datos de test para análisis (opcional)
            feature_names: Nombres de las features
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test if X_test is not None else X_train
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.results_dir = 'results'
        
        # Crear directorio de resultados si no existe
        os.makedirs(self.results_dir, exist_ok=True)
    
    def calculate_shap_values(self, use_sample=True, sample_size=100):
        """
        Calcula los SHAP values para el dataset de test.
        
        Args:
            use_sample (bool): Si usar muestra del dataset (más rápido)
            sample_size (int): Tamaño de la muestra si use_sample=True
        """
        import shap  # Lazy import
        
        print("🔄 Calculando SHAP values...")
        print(f"   (Esto puede tomar unos minutos para datasets grandes)\n")
        
        # Crear explainer para CatBoost (TreeExplainer es más eficiente)
        self.explainer = shap.TreeExplainer(self.model)
        
        # Seleccionar datos para análisis
        if use_sample and len(self.X_test) > sample_size:
            # Usar muestra aleatoria
            indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_explain = self.X_test.iloc[indices] if isinstance(self.X_test, pd.DataFrame) else self.X_test[indices]
            print(f"   Usando muestra de {sample_size} registros de {len(self.X_test)} totales")
        else:
            X_explain = self.X_test
            print(f"   Analizando todos los {len(self.X_test)} registros")
        
        # Calcular SHAP values
        self.shap_values = self.explainer.shap_values(X_explain)
        
        # Si es clasificación binaria, tomar valores para clase positiva
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        self.X_explain = X_explain
        
        print("✅ SHAP values calculados exitosamente\n")
        
        return self.shap_values
    
    def plot_summary(self, save=True, plot_type='dot'):
        """
        Genera el SHAP summary plot.
        
        Args:
            save (bool): Si guardar la figura
            plot_type (str): Tipo de plot ('dot' o 'bar')
        """
        import shap  # Lazy import
        
        if self.shap_values is None:
            self.calculate_shap_values()
        
        print(f"📊 Generando SHAP summary plot ({plot_type})...")
        
        plt.figure(figsize=(12, 8))
        
        shap.summary_plot(
            self.shap_values,
            self.X_explain,
            feature_names=self.feature_names,
            plot_type=plot_type,
            show=False
        )
        
        if plot_type == 'dot':
            plt.title('Importancia e Impacto de Features (SHAP)', 
                     fontsize=16, fontweight='bold', pad=20)
        else:
            plt.title('Importancia Promedio de Features (SHAP)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save:
            filename = f'shap_summary_{plot_type}.png' if plot_type == 'bar' else 'shap_summary.png'
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Summary plot guardado en: {filepath}\n")
        
        plt.close()
    
    def plot_waterfall(self, sample_indices=None, max_display=10, save=True):
        """
        Genera waterfall plots para explicar predicciones individuales.
        
        Args:
            sample_indices: Lista de índices a visualizar (por defecto usa 3 aleatorios)
            max_display (int): Número máximo de features a mostrar
            save (bool): Si guardar las figuras
        """
        import shap  # Lazy import
        
        if self.shap_values is None:
            self.calculate_shap_values()
        
        if sample_indices is None:
            # Seleccionar 3 ejemplos aleatorios
            sample_indices = np.random.choice(len(self.X_explain), min(3, len(self.X_explain)), replace=False)
        
        print(f"📊 Generando waterfall plots para {len(sample_indices)} ejemplos...")
        
        for idx, sample_idx in enumerate(sample_indices):
            plt.figure(figsize=(10, 6))
            
            # Crear objeto Explanation para waterfall plot
            shap_exp = shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value,
                data=self.X_explain.iloc[sample_idx] if isinstance(self.X_explain, pd.DataFrame) else self.X_explain[sample_idx],
                feature_names=self.feature_names
            )
            
            shap.plots.waterfall(shap_exp, max_display=max_display, show=False)
            
            plt.title(f'Explicación de Predicción - Candidato {sample_idx}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save:
                filepath = os.path.join(self.results_dir, f'shap_waterfall_sample_{idx}.png')
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"✅ Waterfall plot {idx+1} guardado en: {filepath}")
            
            plt.close()
        
        print()
    
    def plot_force(self, sample_index=0, save=True):
        """
        Genera force plot para una predicción específica.
        
        Args:
            sample_index (int): Índice del ejemplo a visualizar
            save (bool): Si guardar como HTML
        """
        import shap  # Lazy import
        
        if self.shap_values is None:
            self.calculate_shap_values()
        
        print(f"📊 Generando force plot para candidato {sample_index}...")
        
        # Crear force plot
        shap_exp = shap.Explanation(
            values=self.shap_values[sample_index],
            base_values=self.explainer.expected_value,
            data=self.X_explain.iloc[sample_index] if isinstance(self.X_explain, pd.DataFrame) else self.X_explain[sample_index],
            feature_names=self.feature_names
        )
        
        force_plot = shap.plots.force(shap_exp, matplotlib=False)
        
        if save:
            filepath = os.path.join(self.results_dir, f'shap_force_plot_sample_{sample_index}.html')
            shap.save_html(filepath, force_plot)
            print(f"✅ Force plot guardado en: {filepath}\n")
    
    def plot_dependence(self, feature_indices=None, top_n=3, save=True):
        """
        Genera dependence plots para las features más importantes.
        
        Args:
            feature_indices: Lista de índices de features (por defecto usa top N)
            top_n (int): Número de top features si no se especifican índices
            save (bool): Si guardar las figuras
        """
        import shap  # Lazy import
        
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # Si no se especifican features, usar las más importantes
        if feature_indices is None:
            # Calcular importancia promedio
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
            feature_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
        
        print(f"📊 Generando dependence plots para {len(feature_indices)} features principales...")
        
        for idx in feature_indices:
            if self.feature_names:
                feature_name = self.feature_names[idx]
            else:
                feature_name = f'Feature_{idx}'
            
            plt.figure(figsize=(10, 6))
            
            shap.dependence_plot(
                idx,
                self.shap_values,
                self.X_explain,
                feature_names=self.feature_names,
                show=False
            )
            
            plt.title(f'Dependence Plot: {feature_name}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save:
                safe_name = feature_name.replace('/', '_').replace(' ', '_')
                filepath = os.path.join(self.results_dir, f'shap_dependence_{safe_name}.png')
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"✅ Dependence plot para '{feature_name}' guardado")
            
            plt.close()
        
        print()
    
    def generate_interpretation_report(self, save=True):
        """
        Genera un reporte de texto con insights de SHAP.
        
        Args:
            save (bool): Si guardar el reporte
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # Calcular importancia promedio
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Ordenar features por importancia
        feature_importance_idx = np.argsort(mean_abs_shap)[::-1]
        
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("REPORTE DE INTERPRETABILIDAD DEL MODELO (SHAP)")
        report_lines.append("="*70)
        report_lines.append("")
        report_lines.append("📊 IMPORTANCIA DE FEATURES (Top 10)")
        report_lines.append("-"*70)
        
        for i, idx in enumerate(feature_importance_idx[:10], 1):
            if self.feature_names:
                fname = self.feature_names[idx]
            else:
                fname = f'Feature_{idx}'
            
            importance = mean_abs_shap[idx]
            report_lines.append(f"{i:2d}. {fname:30s} | Importancia: {importance:.4f}")
        
        report_lines.append("")
        report_lines.append("="*70)
        report_lines.append("INTERPRETACIÓN:")
        report_lines.append("="*70)
        report_lines.append("")
        report_lines.append("Los SHAP values indican la contribución de cada feature a la predicción:")
        report_lines.append("")
        report_lines.append("- Valores positivos → Incrementan la probabilidad de ACEPTACIÓN")
        report_lines.append("- Valores negativos → Incrementan la probabilidad de RECHAZO")
        report_lines.append("- Magnitud → Importancia del impacto")
        report_lines.append("")
        report_lines.append("Visualizaciones generadas:")
        report_lines.append("  • shap_summary.png - Muestra importancia y dirección del impacto")
        report_lines.append("  • shap_summary_bar.png - Importancia promedio absoluta")
        report_lines.append("  • shap_waterfall_*.png - Explicaciones de casos individuales")
        report_lines.append("  • shap_force_plot_*.html - Visualización interactiva")
        report_lines.append("  • shap_dependence_*.png - Relación feature-predicción")
        report_lines.append("")
        report_lines.append("="*70)
        
        # Imprimir en consola
        report_text = "\n".join(report_lines)
        print("\n" + report_text + "\n")
        
        if save:
            filepath = os.path.join(self.results_dir, 'shap_interpretation_report.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✅ Reporte de interpretabilidad guardado en: {filepath}\n")
    
    def analyze_full(self, use_sample=True, sample_size=100):
        """
        Ejecuta análisis completo de interpretabilidad.
        
        Args:
            use_sample (bool): Si usar muestra para acelerar
            sample_size (int): Tamaño de muestra
        """
        print("\n🔍 Iniciando análisis completo de interpretabilidad SHAP...\n")
        
        # Calcular SHAP values
        self.calculate_shap_values(use_sample=use_sample, sample_size=sample_size)
        
        # Generar todas las visualizaciones
        self.plot_summary(plot_type='dot')
        self.plot_summary(plot_type='bar')
        self.plot_waterfall()
        self.plot_force(sample_index=0)
        self.plot_dependence(top_n=3)
        
        # Generar reporte
        self.generate_interpretation_report()
        
        print(f"✅ Análisis de interpretabilidad completo. Resultados en: {self.results_dir}/\n")


def analyze_interpretability(model, X_train, X_test, feature_names=None, 
                             use_sample=True, sample_size=100):
    """
    Pipeline para análisis de interpretabilidad con SHAP.
    
    Args:
        model: Modelo entrenado
        X_train: Datos de entrenamiento
        X_test: Datos de test
        feature_names: Nombres de features
        use_sample (bool): Si usar muestra para acelerar
        sample_size (int): Tamaño de muestra
        
    Returns:
        ModelInterpreter: Intérprete con análisis completo
    """
    interpreter = ModelInterpreter(model, X_train, X_test, feature_names)
    interpreter.analyze_full(use_sample=use_sample, sample_size=sample_size)
    
    return interpreter


if __name__ == "__main__":
    print("Este módulo debe ser importado. Ejecuta main.py para el pipeline completo.")
