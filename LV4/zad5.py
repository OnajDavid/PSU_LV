import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

df = pd.read_csv('cars_processed.csv') 

df = df.drop(['name', 'fuel', 'seller_type', 'transmission'], axis=1)

for col in ['mileage', 'engine', 'max_power']:
    df[col] = df[col].str.replace('[^0-9.]', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

X = df.drop('selling_price', axis=1)
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("=== TRAIN SET ===")
print("MAE: ", mean_absolute_error(y_train, y_train_pred))
print("MSE: ", mean_squared_error(y_train, y_train_pred))
print("R2: ", r2_score(y_train, y_train_pred))
print("Max Error: ", max_error(y_train, y_train_pred))

print("\n=== TEST SET ===")
print("MAE: ", mean_absolute_error(y_test, y_test_pred))
print("MSE: ", mean_squared_error(y_test, y_test_pred))
print("R2: ", r2_score(y_test, y_test_pred))
print("Max Error: ", max_error(y_test, y_test_pred))
