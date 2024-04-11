import numpy as np

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1)  # Random input values
y = 2 * X**3 + 3 * X**2 + 4 * X + 5 + np.random.randn(100, 1)  # Cubic equation with noise

# Initialize parameters (integers)
a = np.random.randint(-10, 10)
b = np.random.randint(-10, 10)
c = np.random.randint(-10, 10)
d = np.random.randint(-10, 10)

# Define model
def cubic_model(X, a, b, c, d):
    return a * X**3 + b * X**2 + c * X + d

# Define loss function (Mean Squared Error)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Training loop with SGD
learning_rate = 0.01
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    y_pred = cubic_model(X, a, b, c, d)
    
    # Compute loss
    loss = mse_loss(y, y_pred)
    
    # Backward pass (compute gradients)
    grad_a = np.mean(3 * X**2 * (y_pred - y))
    grad_b = np.mean(2 * X * (y_pred - y))
    grad_c = np.mean(X * (y_pred - y))
    grad_d = np.mean(y_pred - y)
    
    # Update parameters using SGD and ReLU
    a -= learning_rate * grad_a * (grad_a > 0)
    b -= learning_rate * grad_b * (grad_b > 0)
    c -= learning_rate * grad_c * (grad_c > 0)
    d -= learning_rate * grad_d * (grad_d > 0)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss}")

# Evaluate the model
y_pred_final = cubic_model(X, a, b, c, d)
final_loss = mse_loss(y, y_pred_final)
print("Final Loss:", final_loss)
print("Final Parameters (a, b, c, d):", a, b, c, d)
