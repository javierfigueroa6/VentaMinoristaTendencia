import pandas as pd
import numpy as np
import locale
from datetime import datetime

# Set seed for reproducibility
np.random.seed(42)

# Generate dates from 2018 to 2024
dates = pd.date_range(start='2018-01-01', end='2024-12-01', freq='MS')

# Generate synthetic sales data with trend and seasonality
# Trend: increasing over time
trend = np.linspace(10, 20, len(dates))
# Seasonality: higher in December (Christmas), lower in February
seasonality = 5 * np.sin(2 * np.pi * dates.month / 12) + np.random.normal(0, 1, len(dates))
sales = trend + seasonality + 50 # Base level

# Format dates as '01.mmm.yyyy' (e.g., '01.ene.2018')
# We need to manually map months because locale setting might vary on the user's machine
spanish_months = {
    1: 'ene', 2: 'feb', 3: 'mar', 4: 'abr', 5: 'may', 6: 'jun',
    7: 'jul', 8: 'ago', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dic'
}

formatted_dates = [f"01.{spanish_months[d.month]}.{d.year}" for d in dates]

# Format sales as string with comma decimal
formatted_sales = [f"{s:.2f}".replace('.', ',') for s in sales]

# Create DataFrame
df = pd.DataFrame({
    'Periodo': formatted_dates,
    '1. Índice de ventas diarias de comercio minorista': formatted_sales
})

# Save to CSV
output_path = 'retail_sales_forecast/data/ventas_chile_dummy.csv'
df.to_csv(output_path, index=False, sep=',')

print(f"Dummy data generated at {output_path}")
print(df.head())
