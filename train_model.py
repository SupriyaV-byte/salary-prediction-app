import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample training data
data = {
    "experience": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
    "education": [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3],
    "salary": [
        180000, 210000, 280000, 330000,
        380000, 460000, 520000, 580000,
        640000, 760000, 950000, 1200000
    ]
}

df = pd.DataFrame(data)

# Input
X = df[["experience", "education"]]

# Output
y = df["salary"]

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X, y)

# Save trained model
joblib.dump(model, "salary_model.pkl")

print("Model trained successfully!")
print("Model R² Score:", round(model.score(X, y), 3))