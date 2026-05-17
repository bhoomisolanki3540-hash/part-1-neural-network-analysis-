# Part 1: Neural Network Fundamentals and Training Behavior Analysis

# Project Overview

This project focuses on building and analyzing a simple feed-forward neural network for a supervised learning problem.

The dataset used in this project is a customer churn dataset. The goal is to predict whether a customer will churn or remain with the company.

The project demonstrates:

* Dataset exploration
* Data preprocessing
* Neural network model building
* Model training and evaluation
* Hyperparameter experimentation
* Final reflection on neural network learning behavior

---

# Task 1: Dataset Understanding

## Dataset Name

```text
customer_churn_nn.csv
```

---

## Dataset Shape

* Total Rows: `2000`
* Total Columns: `17`

---

## Target Variable

The target variable is:

```text
churn
```

The `churn` column represents whether a customer left the company or stayed.

### Target Labels

| Value | Meaning           |
| ----- | ----------------- |
| `0`   | Customer retained |
| `1`   | Customer churned  |

---

## Input Feature Types

The dataset contains both categorical and numerical features.

---

## Categorical Features

* `region`
* `plan_type`
* `contract_type`
* `payment_method`

---

## Numerical Features

* `tenure_months`
* `monthly_charges_inr`
* `avg_login_days_per_month`
* `support_tickets_last_90_days`
* `payment_delay_days`
* `data_usage_gb`
* `satisfaction_score`
* `last_complaint_days_ago`
* `discount_percent`
* `autopay_enabled`
* `referral_count`

---

## Identifier Column

The `customer_id` column is only a unique identifier and was not used during model training.

---

# Missing Value Check

Missing values were checked using:

```python
df.isnull().sum()
```

The notebook includes a missing value check for each column before preprocessing.

---

# Basic Statistical Summary

A statistical summary was generated using:

```python
df.describe(include='all')
```

This helped analyze:

* Mean
* Standard deviation
* Minimum and maximum values
* Feature distributions
* Categorical value frequencies

---

# Target Variable Distribution

The target variable distribution is highly imbalanced.

## Class Distribution

| Class | Description       | Records |
| ----- | ----------------- | ------- |
| `0`   | Customer retained | 1969    |
| `1`   | Customer churned  | 31      |

---

## Imbalance Interpretation

Most customers in the dataset did not churn.

Because of this imbalance:

* Accuracy alone can be misleading
* A model may achieve high accuracy by predicting the majority class
* Detecting churned customers becomes more difficult

---

# Task 2: Data Preprocessing

The following preprocessing steps were applied before training the neural network.

---

## Preprocessing Steps

1. Removed the `customer_id` column
2. Separated the `churn` column as the target variable
3. Applied one-hot encoding to categorical columns
4. Scaled numerical features using `StandardScaler`
5. Split the dataset into training and testing sets
6. Used stratified splitting to preserve class distribution

---

## Train-Test Split

| Dataset      | Percentage |
| ------------ | ---------- |
| Training Set | 80%        |
| Testing Set  | 20%        |

---

## Why Scaling Was Important

Scaling ensures numerical features are on similar ranges.

Neural networks train more effectively when features are normalized or standardized.

---

# Task 3: Neural Network Model Building

A feed-forward neural network was built using TensorFlow/Keras.

---

# Baseline Model Architecture

The baseline neural network includes:

1. Input Layer
2. Hidden Layer with 32 neurons and ReLU activation
3. Hidden Layer with 16 neurons and ReLU activation
4. Output Layer with 1 neuron and Sigmoid activation

---

## Output Layer

The sigmoid output layer was used because this is a binary classification problem.

---

## Loss Function

```text
binary_crossentropy
```

This loss function is appropriate for binary classification tasks.

---

## Optimizer

```text
Adam
```

Adam efficiently updates model weights during training.

---

# Task 4: Training and Evaluation

The baseline model was trained for:

* Epochs: `30`
* Batch Size: `32`

The model was trained using training data and evaluated using testing data.

---

# Evaluation Output File

```text
results/evaluation_outputs.png
```

This image contains:

* Training loss curve
* Validation/testing loss curve
* Confusion matrix

---

# Evaluation Metrics

The model generated:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

# Confusion Matrix Purpose

The confusion matrix shows:

* Correct classifications
* Incorrect classifications
* Prediction errors for each class

---

# Result Interpretation

The model achieved high overall accuracy.

However, because the dataset is highly imbalanced:

* Accuracy alone is not enough
* Minority churn cases are harder to detect
* The model may still struggle to identify churned customers correctly

---

# Task 5: Hyperparameter Experimentation

Multiple experiments were performed by changing:

* Hidden layers
* Learning rate
* Batch size
* Epoch count
* Activation function

---

# Comparison Table File

```text
results/model_comparison_table.csv
```

---

# Experiment Results

## Baseline Model Benchmark

| Parameter       | Value      |
| --------------- | ---------- |
| Hidden Layers   | `[32, 16]` |
| Learning Rate   | `0.001`    |
| Epochs          | `30`       |
| Batch Size      | `32`       |
| Activation      | `relu`     |
| Train Accuracy  | `0.9856`   |
| Test Accuracy   | `0.9850`   |
| Final Test Loss | `0.0589`   |

---

## Experiment 1: Deeper Network Archetype

| Parameter       | Value          |
| --------------- | -------------- |
| Hidden Layers   | `[64, 32, 16]` |
| Learning Rate   | `0.001`        |
| Epochs          | `30`           |
| Batch Size      | `32`           |
| Activation      | `relu`         |
| Train Accuracy  | `1.0000`       |
| Test Accuracy   | `0.9750`       |
| Final Test Loss | `0.1367`       |

---

## Experiment 2: Conservative Learning Step

| Parameter       | Value      |
| --------------- | ---------- |
| Hidden Layers   | `[32, 16]` |
| Learning Rate   | `0.0001`   |
| Epochs          | `40`       |
| Batch Size      | `32`       |
| Activation      | `relu`     |
| Train Accuracy  | `0.9844`   |
| Test Accuracy   | `0.9850`   |
| Final Test Loss | `0.0737`   |

---

## Experiment 3: Alternative Tanh Activation Function

| Parameter       | Value      |
| --------------- | ---------- |
| Hidden Layers   | `[32, 16]` |
| Learning Rate   | `0.001`    |
| Epochs          | `30`       |
| Batch Size      | `64`       |
| Activation      | `tanh`     |
| Train Accuracy  | `0.9862`   |
| Test Accuracy   | `0.9850`   |
| Final Test Loss | `0.0518`   |

---

# Interpretation of Experiments

## Experiment 1

Experiment 1 showed signs of overfitting because:

* Training accuracy reached `1.0000`
* Test accuracy decreased
* Test loss increased

---

## Experiment 2

Experiment 2 used a smaller learning rate.

The model trained more conservatively but did not significantly improve testing accuracy.

---

## Experiment 3

Experiment 3 achieved the lowest final test loss, making it the best configuration based on loss performance.

---

# Task 6: Final Reflection

## What Role do Weights and Biases Play?

Weights determine the importance of each input feature.

Higher weights indicate stronger influence on predictions.

Biases help shift neuron outputs and improve model flexibility.

Together, weights and biases are the trainable parameters learned during training.

---

## Why is an Activation Function Required?

Activation functions allow neural networks to learn non-linear patterns.

Without activation functions, deep networks would behave like simple linear models.

### Activation Functions Used

| Layer Type    | Activation |
| ------------- | ---------- |
| Hidden Layers | ReLU       |
| Output Layer  | Sigmoid    |

---

## What Happens When Learning Rate is Too High or Too Low?

### High Learning Rate

* Large parameter updates
* Loss may oscillate
* Model may fail to converge

### Low Learning Rate

* Very slow learning
* Requires more epochs
* Longer training time

A balanced learning rate helps the model converge steadily.

---

## Did the Model Show Underfitting or Overfitting?

The baseline model did not show strong underfitting because it achieved high training and testing accuracy.

Experiment 1 showed signs of overfitting because:

* Training accuracy became perfect
* Test accuracy decreased
* Test loss increased

Class imbalance also affects interpretation because the minority churn class remains difficult to predict.

---

# How Neural Networks Learn

## Forward Pass

During the forward pass:

* Input features move through neural network layers
* Weights and biases are applied
* Activation functions generate outputs
* Final predictions are produced

---

## Loss Calculation

The predicted output is compared with the actual target value using a loss function.

This project used:

```text
binary_crossentropy
```

because the task is binary classification.

---

## Backpropagation

Backpropagation sends prediction error backward through the network.

It calculates how much each parameter contributed to the error.

---

## Parameter Updates

The optimizer updates:

* Weights
* Biases

using gradients calculated during backpropagation.

This process gradually reduces loss and improves predictions over multiple epochs.

---

# Results Files

The project includes the following required result files:

| File                                 | Purpose                              |
| ------------------------------------ | ------------------------------------ |
| `results/evaluation_outputs.png`     | Loss curves and confusion matrix     |
| `results/model_comparison_table.csv` | Hyperparameter experiment comparison |

---

# Tools and Libraries Used

The project uses:

* Python
* TensorFlow/Keras
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

# Repository Structure

```text
part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
├── customer_churn_nn.csv
├── data_dictionary.md
└── results/
    ├── evaluation_outputs.png
    └── model_comparison_table.csv
```

---

# Conclusion

This project demonstrates how a feed-forward neural network can be used for supervised binary classification.

The model predicts customer churn using structured numerical and categorical features.

The project covers:

* Dataset understanding
* Data preprocessing
* Neural network creation
* Model training
* Evaluation
* Hyperparameter experimentation
* Neural network learning behavior

The experiments also show how architecture choices, activation functions, and learning rates influence neural network performance.

