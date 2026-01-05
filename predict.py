import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/model.pkl")

# New house data for prediction
new_data = pd.DataFrame({
    "size": [1200],
    "bedrooms": [3]
})

# Make prediction
prediction = model.predict(new_data)

print("Predicted house price:", prediction[0])
