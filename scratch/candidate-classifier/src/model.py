"""
Módulo de entrenamiento del modelo CatBoost para clasificación de postulantes.

Implementa entrenamiento, validación y guardado del modelo.
"""

from catboost import CatBoostClassifier, Pool
import pickle
import os
from datetime import datetime


class CandidateClassifierModel:
    """Clase para manejar el modelo de clasificación con CatBoost."""
    
    def __init__(self, categorical_features_indices=None):
        """
        Inicializa el modelo CatBoost.
        
        Args:
            categorical_features_indices (list): Índices de features categóricas
        """
        self.categorical_features_indices = categorical_features_indices
        self.model = None
        self.training_history = {}
        
    def create_model(self, iterations=500, learning_rate=0.1, depth=6, 
                     l2_leaf_reg=3, verbose=100):
        """
        Crea y configura el modelo CatBoost.
        
        Args:
            iterations (int): Número de iteraciones de boosting
            learning_rate (float): Tasa de aprendizaje
            depth (int): Profundidad de los árboles
            l2_leaf_reg (float): Regularización L2
            verbose (int): Frecuencia de logging durante entrenamiento
            
        Returns:
            CatBoostClassifier: Modelo configurado
        """
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            loss_function='Logloss',
            eval_metric='Accuracy',
            random_seed=42,
            verbose=verbose,
            cat_features=self.categorical_features_indices
        )
        
        print("✅ Modelo CatBoost configurado:")
        print(f"   - Iteraciones: {iterations}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Profundidad: {depth}")
        print(f"   - Features categóricas: {len(self.categorical_features_indices) if self.categorical_features_indices else 0}")
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        Entrena el modelo CatBoost.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_val: Features de validación (opcional)
            y_val: Target de validación (opcional)
            plot (bool): Si mostrar gráficos durante entrenamiento
            
        Returns:
            CatBoostClassifier: Modelo entrenado
        """
        if self.model is None:
            self.create_model()
        
        print("\n🚀 Iniciando entrenamiento del modelo CatBoost...\n")
        
        # Crear pool de datos
        train_pool = Pool(X_train, y_train, cat_features=self.categorical_features_indices)
        
        # Si hay datos de validación, crear pool
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(X_val, y_val, cat_features=self.categorical_features_indices)
        
        # Entrenar
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            plot=plot,
            use_best_model=True if eval_set else False
        )
        
        # Guardar información del entrenamiento
        self.training_history = {
            'train_samples': X_train.shape[0],
            'features': X_train.shape[1],
            'iterations': self.model.get_params()['iterations'],
            'best_iteration': self.model.get_best_iteration() if eval_set else None,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print("\n✅ Entrenamiento completado!")
        if eval_set:
            print(f"   - Mejor iteración: {self.model.get_best_iteration()}")
        
        return self.model
    
    def predict(self, X):
        """
        Realiza predicciones con el modelo.
        
        Args:
            X: Features para predecir
            
        Returns:
            array: Predicciones (0 o 1)
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Obtiene probabilidades de predicción.
        
        Args:
            X: Features para predecir
            
        Returns:
            array: Probabilidades para cada clase
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names=None):
        """
        Obtiene la importancia de las features.
        
        Args:
            feature_names (list): Nombres de las features
            
        Returns:
            dict: Diccionario con feature: importancia
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        
        importance = self.model.get_feature_importance()
        
        if feature_names:
            return dict(zip(feature_names, importance))
        return importance
    
    def save_model(self, filepath='models/catboost_model.cbm'):
        """
        Guarda el modelo entrenado.
        
        Args:
            filepath (str): Ruta donde guardar el modelo
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar. Entrena el modelo primero.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar modelo CatBoost
        self.model.save_model(filepath)
        
        # Guardar metadatos adicionales
        metadata_path = filepath.replace('.cbm', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'categorical_features_indices': self.categorical_features_indices,
                'training_history': self.training_history
            }, f)
        
        print(f"✅ Modelo guardado en: {filepath}")
        print(f"✅ Metadata guardada en: {metadata_path}")
    
    def load_model(self, filepath='models/catboost_model.cbm'):
        """
        Carga un modelo previamente guardado.
        
        Args:
            filepath (str): Ruta del modelo guardado
        """
        self.model = CatBoostClassifier()
        self.model.load_model(filepath)
        
        # Cargar metadatos
        metadata_path = filepath.replace('.cbm', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.categorical_features_indices = metadata.get('categorical_features_indices')
                self.training_history = metadata.get('training_history', {})
        
        print(f"✅ Modelo cargado desde: {filepath}")


def train_model(X_train, y_train, X_test, y_test, categorical_features_indices):
    """
    Pipeline de entrenamiento del modelo.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_test: Features de prueba
        y_test: Target de prueba
        categorical_features_indices: Índices de features categóricas
        
    Returns:
        CandidateClassifierModel: Modelo entrenado
    """
    print("🔄 Iniciando pipeline de entrenamiento...")
    
    # Crear instancia del modelo
    classifier = CandidateClassifierModel(categorical_features_indices)
    
    # Configurar modelo
    classifier.create_model(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        verbose=50
    )
    
    # Entrenar con conjunto de validación
    classifier.train(X_train, y_train, X_test, y_test)
    
    # Guardar modelo
    classifier.save_model()
    
    print("✅ Pipeline de entrenamiento completado\n")
    
    return classifier


if __name__ == "__main__":
    print("Este módulo debe ser importado. Ejecuta main.py para el pipeline completo.")
