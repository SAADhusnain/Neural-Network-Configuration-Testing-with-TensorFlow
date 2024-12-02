import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
file_path = 'Your data.xlsx'
dataset = pd.read_excel(file_path)

# Select Features and Target
features = dataset[['Your Parameters', 'Your Parameters', 'Your Parameters', 'Your Parameters']]
target = dataset['Target Parameters']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define the model with variable number of layers, neurons, and dropout rates
def create_model(layers_configs, dropout_rates):
    model = keras.Sequential()
    for i, (neurons, dropout_rate) in enumerate(zip(layers_configs, dropout_rates)):
        if i == 0:
            model.add(keras.layers.Dense(neurons, activation='relu', input_shape=(features.shape[1],)))
        else:
            model.add(keras.layers.Dense(neurons, activation='relu'))
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Lists of configurations to test
# Number of layers to test
num_layers_list = [1, 2, 3, 4]  

# Neurons configurations for each layer
neurons_configs = {
    1: [(16,), (32,), (64,), (128,)],  # 1 Layer
    2: [(16, 8), (32, 16), (64, 32), (128, 64)],  # 2 Layers
    3: [(16, 8, 4), (32, 16, 8), (64, 32, 16), (128, 64, 32)],  # 3 Layers
    4: [(16, 8, 4, 2), (32, 16, 8, 4), (64, 32, 16, 8), (128, 64, 32, 16)]  # 4 Layers
}

# Dropout rates configurations for each layer
dropout_rates_configs = {
    1: [(0.0,), (0.1,), (0.2,), (0.3,)],  # 1 Layer
    2: [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2), (0.3, 0.3)],  # 2 Layers - Same dropout for all
    3: [(0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3)],  # 3 Layers - Same dropout for all
    4: [(0.0, 0.0, 0.0, 0.0), (0.1, 0.1, 0.1, 0.1), (0.2, 0.2, 0.2, 0.2), (0.3, 0.3, 0.3, 0.3)]  # 4 Layers - Same dropout for all
}

# Dictionary to store the results
results = {}

for num_layers in num_layers_list:
    for neurons_config in neurons_configs[num_layers]:
        for dropout_rates in dropout_rates_configs[num_layers]:
            if len(neurons_config) != num_layers or len(dropout_rates) != num_layers:
                continue  # Skip if lengths don't match the number of layers
            
            print(f"Testing {num_layers} layers, Neurons: {neurons_config}, Dropout Rates: {dropout_rates}")
            
            # Create a new model for each configuration
            model = create_model(neurons_config, dropout_rates)
            
            # Train the model
            model.fit(X_train, y_train, epochs=10, verbose=0)
            
            # Make predictions on the test set
            predictions = model.predict(X_test)
            
            # Calculate the Mean Squared Error (MSE)
            mse = mean_squared_error(y_test, predictions)
            # Store the result
            config_key = f"{num_layers} Layers - Neurons:{neurons_config} - Dropout Rates:{dropout_rates}"
            results[config_key] = mse
            print(f"MSE for configuration {config_key}: {mse:.2f}\n")

# Plot the results
plt.figure(figsize=(12,6))

# Group results by number of layers for plotting
for num_layers in num_layers_list:
    filtered_configs = [key for key in results.keys() if key.startswith(f"{num_layers} Layers")]
    filtered_mse = [results[config] for config in filtered_configs]
    
    # Extract neuron and dropout config for labeling
    labels = [f"Neurons:{config.split('-')[1].split(':')[1]} | Dropout Rates:{config.split('-')[2].split(':')[1]}" for config in filtered_configs]
    
    plt.plot(labels, filtered_mse, label=f"{num_layers} Layer(s)", marker='o')

plt.title('Number of Layers, Neurons, & Dropout Rates vs Mean Squared Error (MSE)')
plt.xlabel('Configuration (Neurons | Dropout Rates)')
plt.ylabel('MSE Value')
plt.legend()
plt.grid(True)
plt.show()

# Find and print the best configuration (based on the lowest MSE)
best_config = min(results, key=results.get)
print(f"\nBest Configuration (Number of Layers, Neurons, Dropout Rates) based on lowest MSE: {best_config}")
print(f"Best MSE: {results[best_config]:.2f}")

# Detailed Analysis of the Best Configuration
best_config_parts = best_config.split(' - ')
num_layers_best = int(best_config_parts[0].split(' ')[0])
neurons_best = tuple(map(int, best_config_parts[1].split(':')[1].strip().split(', ')))
dropout_rates_best = tuple(map(float, best_config_parts[2].split(':')[1].strip().split(', ')))

print(f"\n**Detailed Best Configuration Analysis:**")
print(f"- **Number of Layers:** {num_layers_best}")
print(f"- **Neurons per Layer:** {neurons_best}")
print(f"- **Dropout Rates per Layer:** {dropout_rates_best}")

# Example of how to use the best configuration to make a new model
best_model = create_model(neurons_best, dropout_rates_best)
print("\n**Training the Best Model Again for Demonstration:**")
best_model.fit(X_train, y_train, epochs=10, verbose=1)

# Make a prediction with the best model
best_prediction = best_model.predict(X_test)
print("\n**Example Prediction using the Best Model:**")
print(best_prediction[:5])  # Show the first 5 predictions
