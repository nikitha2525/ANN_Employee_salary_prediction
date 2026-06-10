# ⚡ Day 47 — ANN, Activation Functions & Optimizers: How Neural Networks Learn

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-API-D00000?style=flat-square&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Data-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Preprocessing-150458?style=flat-square&logo=pandas&logoColor=white)
![Challenge](https://img.shields.io/badge/100%20Days%20AI%2FML-Day%2047-blueviolet?style=flat-square)

**A neural network is not powerful because it has many layers — it's powerful because the right activation functions and optimizers help it learn meaningful patterns.**

</div>

---

## 📌 Overview

Adding more layers doesn't make a neural network smarter. Without the right **activation functions**, every layer collapses into a single linear transformation — no matter how deep the network is. Without the right **optimizer**, the network either learns too slowly, overshoots, or gets stuck entirely.

Day 47 breaks down the three building blocks every neural network depends on:
- **ANN architecture** — how information flows from input to output
- **Activation functions** — how networks learn nonlinear patterns
- **Optimizers** — how weights get updated to minimize loss

Applied to a real task: **Employee Salary Prediction using TensorFlow**.

> **Hard truth learned today:** A neural network is not powerful because it has many layers. It's powerful because the right activation functions and optimizers help it learn meaningful patterns.

---

## 🧠 Artificial Neural Networks (ANN)

An ANN is a layered computational graph where each node (neuron) receives weighted inputs, applies an activation function, and passes its output forward.

```
INPUT LAYER          HIDDEN LAYER 1       HIDDEN LAYER 2       OUTPUT LAYER
                                                                
  x₁ ──────────┐    ┌─── h₁₁ ────┐    ┌─── h₂₁ ────┐    
  x₂ ──────────┼───►│    h₁₂     ├───►│    h₂₂     ├───►   ŷ (salary)
  x₃ ──────────┤    │    h₁₃     │    │    h₂₃     │    
  x₄ ──────────┘    │    h₁₄     │    └────────────┘    
                     └────────────┘                      
  
  Features:           Dense(64)            Dense(32)          Dense(1)
  experience,         + ReLU               + ReLU             Linear
  education,
  department,
  age
```

**Forward propagation** at each neuron:

$$z = \sum_{i=1}^{n} w_i x_i + b \qquad a = f(z)$$

Where $w_i$ = weights, $b$ = bias, $f$ = activation function, $a$ = neuron output.

---

## ⚡ Activation Functions

Without activation functions, stacking layers is mathematically equivalent to having a single linear layer. Activation functions introduce **nonlinearity** — allowing the network to learn complex, curved decision boundaries.

### ReLU — Rectified Linear Unit

$$f(x) = \max(0, x)$$

```
Output
  │         /
  │        /
  │       /
  │      /
──┼─────/──────── Input
  │    0
```

| Property | Detail |
|---|---|
| Range | $[0, +\infty)$ |
| Vanishing gradient | ❌ No (for positive values) |
| Dying neurons | ⚠️ Yes (if inputs stay negative) |
| Computation | Very fast |
| Best used in | Hidden layers (most common choice) |

---

### Sigmoid

$$f(x) = \frac{1}{1 + e^{-x}}$$

```
Output
 1 │          ─────────
   │        ╱
0.5│───────╱
   │      ╱
 0 │─────╱
   └──────────────── Input
      -5   0   5
```

| Property | Detail |
|---|---|
| Range | $(0, 1)$ |
| Vanishing gradient | ✅ Yes (saturates at extremes) |
| Output interpretation | Probability |
| Best used in | Output layer for binary classification |

---

### Tanh — Hyperbolic Tangent

$$f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

```
Output
 1 │          ─────────
   │        ╱
 0 │───────╱
   │      ╱
-1 │─────╱
   └──────────────── Input
      -5   0   5
```

| Property | Detail |
|---|---|
| Range | $(-1, 1)$ |
| Zero-centered | ✅ Yes (better gradient flow than Sigmoid) |
| Vanishing gradient | ⚠️ Less severe than Sigmoid |
| Best used in | Hidden layers when zero-centering matters (RNNs) |

---

### Quick Comparison

| Function | Range | Zero-centered | Vanishing Gradient | Typical Use |
|---|---|---|---|---|
| ReLU | $[0, +\infty)$ | ❌ | ❌ (positive side) | Hidden layers |
| Sigmoid | $(0, 1)$ | ❌ | ✅ severe | Binary output |
| Tanh | $(-1, 1)$ | ✅ | ⚠️ moderate | Hidden (RNNs) |
| Softmax | $(0,1)$, sums to 1 | ❌ | — | Multi-class output |
| Leaky ReLU | $(-\infty, +\infty)$ | ❌ | ❌ | Fix for dying ReLU |

---

## 🚀 Optimizers

Optimizers control **how the network updates its weights** after each batch. They all follow the same goal — minimize the loss function — but differ in *how* they adjust the learning rate and momentum.

### Gradient Descent Core Formula

$$w \leftarrow w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}$$

Where $\eta$ = learning rate, $\mathcal{L}$ = loss function.

---

### SGD — Stochastic Gradient Descent

Updates weights using the gradient from **one random sample** (or mini-batch) at a time.

$$w \leftarrow w - \eta \cdot \nabla_w \mathcal{L}(x_i, y_i)$$

| Property | Detail |
|---|---|
| Speed | Fast per step, noisy updates |
| Convergence | Can oscillate around minimum |
| Memory | Very low |
| Best for | Large datasets, simple problems |

---

### Adam — Adaptive Moment Estimation

Combines **momentum** (smoothed gradient) and **RMSProp** (adaptive per-parameter learning rate). Almost always the best default choice.

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \qquad \text{(1st moment — momentum)}$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \qquad \text{(2nd moment — variance)}$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \qquad \text{(bias correction)}$$

$$w \leftarrow w - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Default values: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

| Property | Detail |
|---|---|
| Learning rate | Adaptive per parameter |
| Momentum | ✅ Built-in |
| Convergence | Fast and stable |
| Memory | Higher (stores m and v) |
| Best for | Most deep learning tasks |

---

### Optimizer Comparison

| Optimizer | Adaptive LR | Momentum | Convergence | Memory | Best for |
|---|---|---|---|---|---|
| SGD | ❌ | Optional | Slow, noisy | Low | Large datasets |
| SGD + Momentum | ❌ | ✅ | Better | Low | CV tasks |
| RMSProp | ✅ | ❌ | Good | Medium | RNNs |
| Adam | ✅ | ✅ | Fast, stable | Medium | Default choice |
| AdamW | ✅ | ✅ | Best generalization | Medium | Transformers |

---

## 💼 Mini Project — Employee Salary Prediction

### Dataset Features

| Feature | Type | Description |
|---|---|---|
| `years_experience` | Numerical | Total work experience |
| `education_level` | Categorical | High School / Bachelor / Master / PhD |
| `department` | Categorical | Engineering / HR / Marketing / Finance |
| `age` | Numerical | Employee age |
| `performance_score` | Numerical | Annual performance rating |
| **`salary`** | **Target** | **Annual salary (USD)** |

---

### Model Architecture

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# --- Load & Preprocess ---
df = pd.read_csv('employee_data.csv')

# Encode categoricals
le = LabelEncoder()
df['education_level'] = le.fit_transform(df['education_level'])
df['department']      = le.fit_transform(df['department'])

X = df.drop('salary', axis=1).values
y = df['salary'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --- Build ANN ---
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)                       # Linear output for regression
])

# --- Compile ---
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)

model.summary()

# --- Train ---
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# --- Evaluate ---
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: ${test_mae:,.2f}")

# --- Predict ---
sample = scaler.transform([[5, 2, 0, 28, 4.2]])     # 5 yrs exp, Master's, Engineering, 28, 4.2 score
predicted_salary = model.predict(sample)[0][0]
print(f"Predicted Salary: ${predicted_salary:,.2f}")
```

---

## 📊 Training Pipeline

```
Raw Employee Data
       │
       ▼
┌──────────────────┐
│  Preprocessing   │  Label encode categoricals, StandardScaler on numerics
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   ANN Model      │  Input(5) → Dense(64,ReLU) → Dense(32,ReLU)
│   (TensorFlow)   │          → Dense(16,ReLU) → Dense(1,Linear)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Compile         │  Loss: MSE  |  Optimizer: Adam  |  Metric: MAE
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Train           │  100 epochs, batch_size=32, 20% validation split
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Evaluate        │  Test MAE — mean absolute salary error in USD
└────────┬─────────┘
         │
         ▼
  Salary Predictions
```

---

## 💡 Key Learnings

- **Activation functions enable neural networks to learn complex patterns** — without them, any depth reduces to a single linear equation
- **Optimizers play a crucial role in convergence** — Adam's adaptive learning rate makes it far more reliable than plain SGD for tabular tasks
- **ANNs model nonlinear relationships** better than linear regression when features interact in complex ways
- **TensorFlow/Keras simplifies deep learning** — architecture, compilation, and training in under 20 lines
- **Feature scaling is critical for ANNs** — neurons are sensitive to input magnitude; always normalize before training

---

## 🗂️ Project Structure

```
day-47-ann-salary-prediction/
├── ann_salary.py               # Main ANN training script
├── activation_explorer.py      # Plots all activation functions
├── optimizer_comparison.py     # SGD vs Adam convergence comparison
├── data/
│   └── employee_data.csv
├── outputs/
│   ├── training_loss_curve.png
│   ├── activation_functions.png
│   └── optimizer_comparison.png
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/day-47-ann-salary-prediction
cd day-47-ann-salary-prediction
pip install -r requirements.txt
python ann_salary.py
```

**Requirements:**
```
tensorflow>=2.10
pandas
numpy
scikit-learn
matplotlib
```

---

## 🔗 Part of the 100 Days AI/ML Engineer Challenge

> Day 47 of 100 — Deep Learning Foundations: ANN, Activations & Optimizers

| ← Previous | Current | Next → |
|---|---|---|
| [Day 45 — CNN Fundamentals](#) | **Day 47 — ANN & Optimizers** | [Day 48](#) |


---

<div align="center">
<sub>Built with curiosity · Part of #100DaysOfAIML · #ANN #DeepLearning #TensorFlow #ActivationFunctions #Optimizers</sub>
</div>
