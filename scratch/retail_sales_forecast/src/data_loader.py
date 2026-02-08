import pandas as pd
import numpy as np
import os

def parse_spanish_date(date_str):
    """
    Parses a date string in format 'dd.mmm.yyyy' with Spanish month abbreviations.
    Example: '01.ene.2018' -> datetime object
    """
    if isinstance(date_str, pd.Timestamp) or isinstance(date_str, datetime):
        return pd.Timestamp(date_str)
        
    spanish_months = {
        'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12
    }
    
    try:
        # If it's already a string in specific format
        if isinstance(date_str, str):
            day, month_str, year = date_str.split('.')
            month = spanish_months[month_str.lower()]
            return pd.Timestamp(year=int(year), month=month, day=int(day))
    except Exception as e:
        # Fallback to standard parser
        return pd.to_datetime(date_str, errors='coerce')
    return pd.NaT

def load_and_clean_data(filepath):
    """
    Loads data from CSV or Excel, parses dates and numeric values.
    Returns a cleaned DataFrame with 'ds' (date) and 'y' (value) columns.
    """
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
    
    # Rename columns for clarity (Position 0: Date, Position 1: Value)
    df = df.iloc[:, [0, 1]]
    df.columns = ['ds', 'y']
    
    # Parse dates
    # Check if 'ds' is already datetime (common in Excel)
    if not pd.api.types.is_datetime64_any_dtype(df['ds']):
         df['ds'] = df['ds'].apply(parse_spanish_date)
    
    # Parse numeric values (handle comma decimal)
    if df['y'].dtype == 'object':
        df['y'] = df['y'].astype(str).str.replace('.', '', regex=False) # Remove thousands separator
        df['y'] = df['y'].str.replace(',', '.', regex=False)
        df['y'] = pd.to_numeric(df['y'])
        
    return df

if __name__ == "__main__":
    from datetime import datetime
    # Test the loader (try to find the file dynamically)
    base_path = 'retail_sales_forecast/data'
    files = os.listdir(base_path)
    target_file = [f for f in files if f.startswith('ventas_chile')][0]
    
    print(f"Loading {target_file}...")
    data = load_and_clean_data(os.path.join(base_path, target_file))
    print("Data loaded successfully:")
    print(data.head())
    print("\nData types:")
    print(data.dtypes)
