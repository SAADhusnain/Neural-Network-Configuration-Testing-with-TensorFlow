# Neural Network Configuration Testing with TensorFlow
 
## Overview
This code tests various configurations of a **fully connected neural network (dense layers)** to identify the optimal structure for minimizing **Mean Squared Error (MSE)** in a regression task. 
The configurations include varying the number of layers, the number of neurons per layer, and dropout rates. 

---

## How the Code Works

### 1. Dataset Loading
- The dataset is loaded from an external file into a pandas DataFrame.
- The dataset should contain:
  - **Features**: Independent variables for training the model.
  - **Target**: Dependent variable the model aims to predict.

### 2. Model Creation Function
- A function, `create_model`, defines the model architecture dynamically based on:
  - Number of layers.
  - Neurons in each layer.
  - Dropout rates for regularization in each layer.
- The final layer always has 1 neuron for regression output.

### 3. Configurations
- **Number of Layers**: Tested from 1 to 4 layers.
- **Neurons per Layer**: Several configurations for the number of neurons in each layer are predefined (e.g., `(16,)`, `(32, 16)`, etc.).
- **Dropout Rates**: Dropout rates corresponding to each configuration are tested (e.g., `0.0`, `0.1`, etc.).

### 4. Model Training and Evaluation
- Each configuration is trained for 10 epochs.
- The model's performance is evaluated on a test set using **Mean Squared Error (MSE)**.
- Results are stored and printed for each configuration.

### 5. Visualization
- Results are visualized in a plot where each configuration's performance (MSE) is compared.
- Configurations are grouped by the number of layers for better comparison.

### 6. Optimal Configuration
- The best configuration, i.e., the one with the lowest MSE, is identified and printed.
- A detailed analysis of the best configuration is provided, including:
  - Number of layers.
  - Neurons per layer.
  - Dropout rates per layer.

---

## Purpose of the Code
- To test and identify the most efficient neural network architecture for a given dataset.
- To experiment with regularization techniques (dropout) for reducing overfitting.
- To create a reusable framework for hyperparameter tuning of dense neural networks.

---

## General Use Cases
- Regression tasks where minimizing prediction error (MSE) is critical.
- Hyperparameter tuning for network architectures in TensorFlow.
- Building a foundation for neural network experimentation.

---

## Requirements
- **Input Dataset**:
  - The dataset must include both features and a target variable.
  - Modify the code to specify appropriate feature and target column names.
- **Libraries**:
  - `numpy`, `pandas`, `tensorflow`, `matplotlib`.
- **Hardware**:
  - GPU acceleration is recommended for faster training, especially for large datasets.

---

## Notes
- The code can be extended to include additional parameters like activation functions, learning rates, or optimizers.
- The range of configurations (layers, neurons, dropout rates) can be adjusted for different datasets or tasks.

---

This modular code allows users to easily adapt the configurations for any regression dataset by changing the dataset file, feature and target column names, or the configuration parameters.