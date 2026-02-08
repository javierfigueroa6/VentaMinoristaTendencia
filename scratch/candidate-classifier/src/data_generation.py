"""
Módulo de generación de datos sintéticos para el clasificador de postulantes.

Este módulo crea un dataset ficticio con características realistas de candidatos
para entrenar el modelo de clasificación.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
import os

# Configurar semilla para reproducibilidad
np.random.seed(42)
random.seed(42)
fake = Faker('es_CL')  # Faker en español de Chile


def generate_candidate_data(n_samples=1500, output_path='data/raw/candidates.csv'):
    """
    Genera un dataset sintético de postulantes de trabajo.
    
    Args:
        n_samples (int): Número de registros a generar
        output_path (str): Ruta donde guardar el archivo CSV
        
    Returns:
        pd.DataFrame: DataFrame con los datos generados
    """
    
    # Listas de valores posibles para variables categóricas
    titulos = [
        'Ingeniero Civil', 'Ingeniero Comercial', 'Ingeniero Informático',
        'Licenciado en Ciencias', 'Contador Auditor', 'Psicólogo',
        'Técnico en Administración', 'Técnico en Informática', 'Abogado',
        'Diseñador Gráfico', 'Periodista', 'Arquitecto', 'Médico', 'Enfermero'
    ]
    
    universidades = [
        'Universidad de Chile', 'Pontificia Universidad Católica',
        'Universidad de Santiago', 'Universidad Técnica Federico Santa María',
        'Universidad de Concepción', 'Universidad Adolfo Ibáñez',
        'DUOC UC', 'INACAP', 'Universidad Mayor', 'Universidad Andrés Bello',
        'Universidad Diego Portales', 'Instituto Profesional AIEP'
    ]
    
    comunas = [
        'Santiago', 'Providencia', 'Las Condes', 'Ñuñoa', 'Maipú',
        'La Florida', 'San Miguel', 'Puente Alto', 'Peñalolén', 'Quilicura',
        'Estación Central', 'Recoleta', 'Independencia', 'Viña del Mar',
        'Valparaíso', 'Concepción', 'La Serena', 'Antofagasta'
    ]
    
    data = {
        'Nombre': [],
        'Edad': [],
        'Titulo_Profesional': [],
        'Universidad_Instituto': [],
        'Palabras_Clave': [],
        'Comuna': [],
        'Presencial': [],
        'Magister': [],
        'Target': []
    }
    
    for _ in range(n_samples):
        # Generar características
        nombre = fake.name()
        edad = np.random.randint(22, 66)
        titulo = random.choice(titulos)
        universidad = random.choice(universidades)
        palabras_clave = np.random.randint(0, 21)
        comuna = random.choice(comunas)
        presencial = random.choice(['Si', 'No'])
        magister = random.choice(['Si', 'No'])
        
        # Lógica para generar el target con correlaciones realistas
        # Crear un score basado en diferentes factores
        score = 0
        
        # Mayor peso a palabras clave (más keywords = mejor candidato)
        score += palabras_clave * 3
        
        # Magíster suma puntos
        if magister == 'Si':
            score += 15
        
        # Edad óptima entre 25-45
        if 25 <= edad <= 45:
            score += 10
        elif edad < 25:
            score += 5
        else:
            score += 3
        
        # Disponibilidad presencial suma
        if presencial == 'Si':
            score += 8
        
        # Universidades más prestigiosas suman (simplificación)
        if universidad in ['Universidad de Chile', 'Pontificia Universidad Católica', 
                          'Universidad Técnica Federico Santa María']:
            score += 10
        elif universidad in ['DUOC UC', 'INACAP']:
            score += 5
        
        # Títulos más demandados (simplificación para este ejemplo)
        if 'Ingeniero' in titulo or 'Informático' in titulo:
            score += 8
        elif 'Técnico' in titulo:
            score += 4
        
        # Convertir score a probabilidad de aceptación
        # Añadir ruido aleatorio para no tener correlación perfecta
        ruido = np.random.normal(0, 10)
        score_final = score + ruido
        
        # Threshold para clasificación: Si score > 60, más probable ser aceptado
        probabilidad = 1 / (1 + np.exp(-(score_final - 60) / 10))
        target = 'Aceptado' if np.random.random() < probabilidad else 'Rechazado'
        
        # Agregar al dataset
        data['Nombre'].append(nombre)
        data['Edad'].append(edad)
        data['Titulo_Profesional'].append(titulo)
        data['Universidad_Instituto'].append(universidad)
        data['Palabras_Clave'].append(palabras_clave)
        data['Comuna'].append(comuna)
        data['Presencial'].append(presencial)
        data['Magister'].append(magister)
        data['Target'].append(target)
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar a CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ Dataset generado exitosamente: {n_samples} registros")
    print(f"📁 Guardado en: {output_path}")
    print(f"\n📊 Distribución de la variable target:")
    print(df['Target'].value_counts())
    print(f"\n📈 Proporción:")
    print(df['Target'].value_counts(normalize=True))
    
    return df


if __name__ == "__main__":
    # Generar datos cuando se ejecuta directamente
    df = generate_candidate_data(n_samples=1500)
    print("\n🔍 Primeras 5 filas del dataset:")
    print(df.head())
    print(f"\n📋 Información del dataset:")
    print(df.info())
