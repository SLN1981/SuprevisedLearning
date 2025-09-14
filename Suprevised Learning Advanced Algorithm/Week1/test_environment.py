import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import sklearn
import torch

# Print versions to verify everything is working
print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)
print("PyTorch version:", torch.__version__)
print("pandas version:", pd.__version__)
print("scikit-learn version:", sklearn.__version__)

# Create a simple numpy array and plot it
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.savefig('sine_wave.png')
print("Created sine_wave.png in the current directory")
