import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('loss.txt')

plt.plot(data)

plt.show()
