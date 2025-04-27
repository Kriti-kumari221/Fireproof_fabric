!pip install git+https://github.com/scikit-fuzzy/scikit-fuzzy.git
import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler.
# Function to simulate sensor data collection
def collect_sensor_data():
    temperature = random.randint(10, 120)  # Temperature in Celsius
    gas_level = random.randint(0, 100)
    oxygen_level = random.randint(5, 20)
    fabric_wear = random.randint(0, 100)
    return [temperature, gas_level, oxygen_level, fabric_wear]
def generate_cnn_input():
    # Generate a random "image" representing sensor data over time or space
    # Just a 2D matrix for the sake of example (you can expand this for your case)
    return np.random.random((10, 10))

# Normalize sensor data before passing to models
scaler = MinMaxScaler(feature_range=(0, 1))

# Example of normalizing the data
def normalize_data(sensor_data):
    return scaler.fit_transform(np.array(sensor_data).reshape(-1, 1)).flatten()

# KNN classification setup
X_train = np.array([
    [25, 10, 19, 10],  # Safe
    [30, 30, 18, 30],  # Warning
    [100, 80, 5, 50],  # Danger
])

y_train = np.array([0, 1, 2])  # Labels: Safe (0), Warning (1), Danger (2)

# Initialize and train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

def knn_predict(sensor_data):
    # Predict state (Safe, Warning, Danger)
    prediction = knn.predict([sensor_data])
    return prediction[0]


# CNN Model Setup (Simple CNN)
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(10, 10, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 classes
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Example CNN data (10x10 grid) for fabric area over time or space
def cnn_predict(sensor_data):
    # Reshape data to fit CNN input requirements
    sensor_data_reshaped = np.array(sensor_data).reshape(1, 10, 10, 1)  # For 10x10 "image"
    prediction = cnn_model.predict(sensor_data_reshaped)
    return np.argmax(prediction)  # Return the class with the highest probability

def genetic_algorithm(cooling_params):
    # Example optimization: Maximize cooling efficiency while minimizing wear and tear
    optimal_cooling = min(cooling_params, key=lambda x: x[0] - x[1])  # Sample fitness function
    return optimal_cooling

def optimize_system():
    cooling_params = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]  # Random cooling settings
    optimal_cooling = genetic_algorithm(cooling_params)
    return optimal_cooling


import skfuzzy as fuzz

def fuzzy_logic(temperature, fabric_wear):
    # Define fuzzy logic rules based on thresholds
    if temperature >= 70 and fabric_wear >= 50:
        cooling_intensity = "High"
    elif temperature >= 40 or fabric_wear >= 30:
        cooling_intensity = "Medium"
    else:
        cooling_intensity = "Low"

    return cooling_intensity


def run_system():
    # Simulate sensor data collection
    sensor_data = collect_sensor_data()
    normalized_data = normalize_data(sensor_data)

    # Step 1: KNN Classification
    state = knn_predict(normalized_data)
    print(f"Predicted State (KNN): {state}")

    # Step 2: CNN Pattern Recognition
    cnn_input = generate_cnn_input()  # Random image data for CNN
    cnn_state = cnn_predict(cnn_input)
    print(f"Predicted State (CNN): {cnn_state}")

    # Step 3: Genetic Algorithm Optimization
    optimal_cooling = optimize_system()
    print(f"Optimal Cooling Settings (Genetic Algorithm): {optimal_cooling}")

    # Step 4: Fuzzy Logic Decision Making
    temperature = sensor_data[0]
    fabric_wear = sensor_data[3]
    cooling_intensity = fuzzy_logic(temperature, fabric_wear)
    print(f"Cooling Intensity (Fuzzy Logic): {cooling_intensity}")

run_system()
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Simulate multiple sensor readings
def collect_multiple_sensor_data(n=50):
    data = {
        'Temperature': [],
        'Gas_Level': [],
        'Oxygen_Level': [],
        'Fabric_Wear': []
    }
    for _ in range(n):
        temp, gas, oxy, wear = collect_sensor_data()
        data['Temperature'].append(temp)
        data['Gas_Level'].append(gas)
        data['Oxygen_Level'].append(oxy)
        data['Fabric_Wear'].append(wear)
    return pd.DataFrame(data)

# Step 2: Collect and visualize
sensor_df = collect_multiple_sensor_data(100)

# Line Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=sensor_df)
plt.title("Sensor Data Over Time")
plt.xlabel("Time (samples)")
plt.ylabel("Sensor Readings")
plt.grid(True)
plt.show()

# Optional: Heatmap of correlation
plt.figure(figsize=(6, 5))
sns.heatmap(sensor_df.corr(), annot=True, cmap='coolwarm')
plt.title("Sensor Data Correlation")
plt.show()









