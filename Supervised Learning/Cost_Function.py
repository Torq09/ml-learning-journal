import numpy as np
import matplotlib.pyplot as plt

# Sample dataset (x = study hours, y = exam scores)
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])  # Perfect linear relation y = 2x

# Cost function (Mean Squared Error)
def compute_cost(x, y, w, b):
    m = len(x)
    predictions = w * x + b
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

# Test different w values (fix b = 0 for simplicity)
w_values = np.linspace(0, 4, 100)  # weights from 0 to 4
cost_values = [compute_cost(x, y, w, b=0) for w in w_values]

# Plot
plt.plot(w_values, cost_values, color='blue')
plt.xlabel('Weight (w)')
plt.ylabel('Cost J(w,b)')
plt.title('Cost Function vs Weight')
plt.grid(True)
plt.show()
