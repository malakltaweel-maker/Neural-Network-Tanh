import numpy as np

# ---------- Activation Function ----------
def tanh(x):
    return np.tanh(x)

# ---------- Inputs ----------
x1 = 0.6
x2 = 0.1

# ---------- Bias ----------
b1 = 0.5
b2 = 0.7

# ---------- Random Weights in [-0.5, 0.5] ----------
np.random.seed(1)  # for reproducibility (optional)

w1 = np.random.uniform(-0.5, 0.5)
w2 = np.random.uniform(-0.5, 0.5)
w3 = np.random.uniform(-0.5, 0.5)
w4 = np.random.uniform(-0.5, 0.5)

v1 = np.random.uniform(-0.5, 0.5)
v2 = np.random.uniform(-0.5, 0.5)

# ---------- Hidden Layer ----------
z1 = (x1 * w1) + (x2 * w2) + b1
z2 = (x1 * w3) + (x2 * w4) + b1

h1 = tanh(z1)
h2 = tanh(z2)

# ---------- Output Layer ----------
z_out = (h1 * v1) + (h2 * v2) + b2
output = tanh(z_out)

# ---------- Print Results ----------
print("Hidden Neuron 1 Output:", h1)
print("Hidden Neuron 2 Output:", h2)
print("Final Network Output:", output)