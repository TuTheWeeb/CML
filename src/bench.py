import numpy as np

hidden_size = 100
out_size = 1

# 1. The Data (A -> B)
# Reshaped to be a column matrix: 3 rows, 1 column
X = np.array([[1.0], [2.0], [3.0]])
Y = np.array([[2.0], [4.0], [6.0]])

# 2. Initialize Weights and Biases (Randomly)
np.random.seed(42)  # For reproducible results

# Layer 1: 1 input -> 10 hidden neurons
W1 = np.random.randn(1, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))

# Layer 2: 10 hidden neurons -> 1 output
W2 = np.random.randn(hidden_size, out_size) * 0.1
b2 = np.zeros((1, out_size))

learning_rate = 0.01


# 3. Activation Functions and their Derivatives
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)  # Returns 1 if x > 0, else 0


# 4. The Training Loop
epochs = 100000
n_samples = X.shape[0]
loss = 0


def forward(A, W, B, activation):
    Z = np.dot(A, W) + B
    return Z, activation(Z)


def calc_loss(A, Y):
    return np.mean((A - Y) ** 2)


def backward_pass(A, Y, num_samples):
    return 2 * (A - Y) / num_samples


def gradient(A, dZ):
    return np.dot(A, dZ), np.sum(dZ, axis=0, keepdims=True)


def update_weight(W, b, dW, db, learning_rate):
    return W - (dW * learning_rate), b - (db * learning_rate)


for epoch in range(epochs):

    # --- FORWARD PASS (The Guess) ---
    # Z1 = np.dot(X, W1) + b1  # Math for hidden layer
    # A1 = relu(Z1)  # Apply non-linearity

    Z1, A1 = forward(X, W1, b1, relu)

    # Z2 = np.dot(A1, W2) + b2  # Math for output layer
    # A2 = Z2  # No activation needed for regression output

    Z2, _ = forward(A1, W2, b2, relu)
    A2 = Z2

    # --- LOSS CALCULATION (Mean Squared Error) ---
    # loss = np.mean((A2 - Y) ** 2)
    loss = calc_loss(A2, Y)

    # --- BACKWARD PASS (The Chain Rule / Learning) ---
    # Derivative of the Loss with respect to the output
    # dZ2 = 2 * (A2 - Y) / n_samples
    dZ2 = backward_pass(A2, Y, n_samples)

    # Gradients for Layer 2
    # dW2 = np.dot(A1.T, dZ2)
    # db2 = np.sum(dZ2, axis=0, keepdims=True)
    dW2, db2 = gradient(A1.T, dZ2)

    # Pass the error backward to Layer 1
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)  # Undo the ReLU activation

    # Gradients for Layer 1
    # dW1 = np.dot(X.T, dZ1)
    # db1 = np.sum(dZ1, axis=0, keepdims=True)
    dW1, db1 = gradient(X.T, dZ1)

    # --- UPDATE WEIGHTS (Gradient Descent) ---
    # W1 -= learning_rate * dW1
    # b1 -= learning_rate * db1
    # W2 -= learning_rate * dW2
    # b2 -= learning_rate * db2
    W1, b1 = update_weight(W1, b1, dW1, db1, learning_rate)
    W2, b2 = update_weight(W2, b2, dW2, db2, learning_rate)

# 5. Test the Network
print(f"Final Loss: {loss:.5f}")

# Let's test it with a number it hasn't seen: 4.0
test_input = np.array([[8.0]])
Z1_test = np.dot(test_input, W1) + b1
A1_test = relu(Z1_test)
prediction = np.dot(A1_test, W2) + b2

print(f"AI Predicts: {prediction[0][0]:.2f}")
print(b2)
