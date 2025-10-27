# train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv("K:\end to end 2\car data.csv")  # make sure your CSV file name matches

# Encode categorical columns
le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_trans = LabelEncoder()

df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])
df['Seller_Type'] = le_seller.fit_transform(df['Seller_Type'])
df['Transmission'] = le_trans.fit_transform(df['Transmission'])

# Define features and target
X = df[['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
y = df['Selling_Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Model trained successfully!\nR2 Score: {r2:.3f}\nMSE: {mse:.3f}")

# Save model and encoders
with open('car_price_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'le_fuel': le_fuel,
        'le_seller': le_seller,
        'le_trans': le_trans
    }, f)

print("✅ Model and encoders saved to car_price_model.pkl")
