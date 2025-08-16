import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Initial parameters
w = float(input("Enter a initial value for w (parameter 1):"))
b = float(input("Enter a initial value for b (parameter 2):"))
learning_rate = float(input("Enter your learning rate:"))
epochs = 1000
n = len(X)

def StartPrediction(w,b,learning_rate):
    print("starting predictions...")
    for epoch in range(epochs):
        # Predict
        y_pred = w * X + b

        # Compute loss (Mean Squared Error)
        loss = np.mean((y - y_pred) ** 2)

        # Compute gradients
        dw = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Optional: print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

    print(f"\nFinal parameters: w={w:.4f}, b={b:.4f}")
    return w, b

# To visualize
w,b = StartPrediction(w,b,learning_rate)
plt.scatter(X, y, label='Data')
plt.plot(X, w*X + b, color='red', label='Regression Line')
plt.legend()
plt.show()
