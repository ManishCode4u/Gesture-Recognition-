import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Loading data...")

# 1. Load the dataset
try:
    data = pd.read_csv('gesture_data.csv')
except FileNotFoundError:
    print("Error: 'gesture_data.csv' not found. Please run your data collector first.")
    exit()

# 2. Separate the Input (Coordinates) from the Output (Labels)
# Force conversion to native numpy arrays to avoid PyArrow compatibility issues in sklearn
X = data.iloc[:, :-1].to_numpy(dtype='float32')
y = data.iloc[:, -1].to_numpy()

# 3. Split data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
# Use K-Nearest Neighbors for classification
print("Training model...")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 5. Check how well it learned
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the trained model to a file
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Success! Model saved as 'gesture_model.pkl'")