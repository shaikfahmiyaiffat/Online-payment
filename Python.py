import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Create a synthetic dataset
np.random.seed(42)

# Features: transaction amount, user age, transaction time, etc.
data_size = 1000
data = {
    'transaction_amount': np.random.normal(100, 50, data_size),
    'user_age': np.random.randint(18, 70, data_size),
    'transaction_time': np.random.randint(0, 24, data_size),  # 0-23 hours
    'is_fraud': np.random.choice([0, 1], data_size, p=[0.95, 0.05])  # 5% fraud cases
}

df = pd.DataFrame(data)

# Split the data into features and target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output the results
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
