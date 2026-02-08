"""
Módulo de preprocesamiento de datos para el clasificador de postulantes.

Maneja la carga, limpieza, transformación y división de datos.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle


class DataPreprocessor:
    """Clase para preprocesar datos de candidatos."""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self, filepath='data/raw/candidates.csv'):
        """
        Carga los datos desde un archivo CSV.
        
        Args:
            filepath (str): Ruta al archivo CSV
            
        Returns:
            pd.DataFrame: DataFrame con los datos cargados
        """
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        print(f"✅ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
        return df
    
    def prepare_features(self, df):
        """
        Prepara las features para el modelo.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            tuple: (X, y) - Features y target
        """
        # Crear una copia para no modificar el original
        df_processed = df.copy()
        
        # Separar features y target
        X = df_processed.drop(['Target', 'Nombre'], axis=1)  # Nombre no es predictivo
        y = df_processed['Target']
        
        # Codificar target
        if 'Target' not in self.label_encoders:
            self.label_encoders['Target'] = LabelEncoder()
            y = self.label_encoders['Target'].fit_transform(y)
        else:
            y = self.label_encoders['Target'].transform(y)
        
        # Identificar columnas categóricas y numéricas
        categorical_features = ['Titulo_Profesional', 'Universidad_Instituto', 
                               'Comuna', 'Presencial', 'Magister']
        numerical_features = ['Edad', 'Palabras_Clave']
        
        # CatBoost puede manejar variables categóricas directamente
        # Solo necesitamos asegurarnos de que sean tipo string
        for col in categorical_features:
            X[col] = X[col].astype(str)
        
        self.feature_names = list(X.columns)
        self.categorical_features_indices = [i for i, col in enumerate(X.columns) 
                                             if col in categorical_features]
        
        print(f"✅ Features preparadas: {X.shape[1]} características")
        print(f"   - Categóricas: {len(categorical_features)}")
        print(f"   - Numéricas: {len(numerical_features)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        
        Args:
            X: Features
            y: Target
            test_size (float): Proporción del conjunto de prueba
            random_state (int): Semilla aleatoria
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✅ Datos divididos:")
        print(f"   - Entrenamiento: {X_train.shape[0]} registros")
        print(f"   - Prueba: {X_test.shape[0]} registros")
        print(f"   - Proporción: {test_size*100}% prueba")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, filepath='data/processed/preprocessor.pkl'):
        """
        Guarda el preprocesador para uso posterior.
        
        Args:
            filepath (str): Ruta donde guardar el preprocesador
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'categorical_features_indices': self.categorical_features_indices
            }, f)
        
        print(f"✅ Preprocesador guardado en: {filepath}")
    
    def load_preprocessor(self, filepath='data/processed/preprocessor.pkl'):
        """
        Carga un preprocesador guardado.
        
        Args:
            filepath (str): Ruta del preprocesador guardado
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.label_encoders = data['label_encoders']
            self.feature_names = data['feature_names']
            self.categorical_features_indices = data['categorical_features_indices']
        
        print(f"✅ Preprocesador cargado desde: {filepath}")
    
    def get_categorical_features_indices(self):
        """Retorna los índices de las features categóricas."""
        return self.categorical_features_indices
    
    def inverse_transform_target(self, y_encoded):
        """
        Convierte target codificado de vuelta a etiquetas originales.
        
        Args:
            y_encoded: Target codificado
            
        Returns:
            array: Target con etiquetas originales
        """
        return self.label_encoders['Target'].inverse_transform(y_encoded)


def preprocess_pipeline(data_path='data/raw/candidates.csv'):
    """
    Pipeline completo de preprocesamiento.
    
    Args:
        data_path (str): Ruta a los datos crudos
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    print("🔄 Iniciando pipeline de preprocesamiento...")
    
    # Inicializar preprocesador
    preprocessor = DataPreprocessor()
    
    # Cargar datos
    df = preprocessor.load_data(data_path)
    
    # Preparar features
    X, y = preprocessor.prepare_features(df)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Guardar preprocesador
    preprocessor.save_preprocessor()
    
    print("✅ Pipeline de preprocesamiento completado\n")
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Ejecutar pipeline cuando se corre directamente
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline()
    
    print("📊 Información de los conjuntos:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"\nPrimeras filas de X_train:")
    print(X_train.head())
