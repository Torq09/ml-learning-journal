import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([300.0, 700.0, 1150.0, 1550.0, 1890.0])

# m is the number of training examples
# print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]

w = 100
b = 100
print(f"current w: {w}")
print(f"current b: {b}")

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

def plot_model_prediction(tmp_f_wb):
    # Plot our model prediction
    plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

    # Plot the data points
    plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

    # Set the title
    plt.title("Housing Prices")
    # Set the y-axis label
    plt.ylabel('Price (in 1000s of dollars)')
    # Set the x-axis label
    plt.xlabel('Size (1000 sqft)')
    plt.legend()
    plt.show()

print("Choose:\n'1' to use default weight and bias\n'2' to use custom weight and bias")
choice = int(input('Choice:'))

if(choice == 1):
    tmp_f_wb = compute_model_output(x_train, w, b)
    plot_model_prediction(tmp_f_wb)
elif(choice == 2):
    w = int(input("Weight:"))
    b = int(input("Bias:"))
    tmp_f_wb = compute_model_output(x_train, w, b)
    plot_model_prediction(tmp_f_wb)
