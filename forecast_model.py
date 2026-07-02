import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# 1. LOAD DATA


df = pd.read_csv("C:/Users/DEll/Desktop/Attention_span_project/student_learning_interaction_dataset.csv")


# 2. CLEAN COLUMN NAMES

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")


# 3. FIX TIMESTAMP FORMAT

df['timestamp'] = df['timestamp'].astype(str)
df['timestamp'] = df['timestamp'].str.replace('.', ':', regex=False)

df['timestamp'] = pd.to_datetime(
    df['timestamp'],
    format="%d-%m-%Y %H:%M",
    errors='coerce'
)

# Drop invalid timestamps
df = df.dropna(subset=['timestamp'])


# 4. SORT DATA

df = df.sort_values(by=["student_id", "timestamp"])


# 5. SELECT ONE STUDENT

student_id = df['student_id'].iloc[0]
student_data = df[df['student_id'] == student_id]


# 6. TIME SERIES PREPARATION

student_data = student_data.set_index('timestamp')
student_data = student_data.sort_index()

# Keep only required column
student_data = student_data[['attention_score']]

# Resample to daily frequency
student_data = student_data.resample('D').mean()

# Fill missing values
student_data['attention_score'] = student_data['attention_score'].interpolate()

attention = student_data['attention_score']


# 7. ARIMA MODEL

model = ARIMA(attention, order=(1, 0, 1))
model_fit = model.fit()


# 8. FORECAST

forecast = model_fit.forecast(steps=10)


# 9. VISUALIZATION

plt.figure(figsize=(10, 5))
plt.plot(attention, label="Historical Attention")
plt.plot(forecast, label="Forecast", linestyle='dashed')
plt.legend()
plt.title(f"Attention Span Forecast for Student {student_id}")
plt.xlabel("Time")
plt.ylabel("Attention Score")
plt.show()