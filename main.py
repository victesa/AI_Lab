import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'Nairobi Office Price Ex.csv'
data = pd.read_csv(file_path)

x = data['SIZE'].values
y = data['PRICE'].values

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(x, y, m, c, learning_rate):
    n = len(y)
    y_pred = m * x + c
    dm = (-2 / n) * np.sum(x * (y - y_pred))
    dc = (-2 / n) * np.sum(y - y_pred)

    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c

# Initialize parameters
m = np.random.randn()  # Random initial value for slope
c = np.random.randn()  # Random initial value for y-intercept
learning_rate = 0.0001  # Smaller learning rate to prevent divergence
epochs = 10

# Train the model for a specified number of epochs
mse_history = []

for epoch in range(epochs):
    # Perform gradient descent to update m and c
    m, c = gradient_descent(x, y, m, c, learning_rate)
    # Calculate mean squared error for the current epoch
    y_pred = m * x + c
    mse = mean_squared_error(y, y_pred)
    mse_history.append(mse)
    # Print the error at each epoch
    print(f"Epoch {epoch + 1}, MSE: {mse:.4f}")

# Plot the line of best fit after the final epoch
plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', label='Line of best fit')
plt.xlabel('Office Size (sq. ft)')
plt.ylabel('Office Price')
plt.title('Linear Regression: Office Size vs. Price (Adjusted Learning Rate)')
plt.legend()
plt.show()

# Display final values of m and c
print(f"Final slope (m): {m}")
print(f"Final intercept (c): {c}")

# Predict the office price for a size of 100 sq. ft using the final model parameters
predicted_price = m * 100 + c
print(f"The predicted office price for a 100 sq. ft office is: {predicted_price:.2f}")
