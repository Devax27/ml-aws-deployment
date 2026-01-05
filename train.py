import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load data
data = pd.read_csv("data/house_data.csv")

X = data.drop("price", axis=1)
y = data["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print("Model trained successfully")
print("Mean Squared Error:", mse)

# Save model
joblib.dump(model, "model/model.pkl")
print("Model saved successfully")
