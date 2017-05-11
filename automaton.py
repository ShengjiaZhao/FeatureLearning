import numpy as np
from matplotlib import pyplot as plt


def sigmoid(val):
    return 1.0 / (1 + np.exp(-val))

img_size = 20
filter_size = 5
x = sigmoid(np.random.randn(img_size, img_size))

class Automaton:
    def __init__(self):
        self.transformation = np.random.randn(2 * filter_size + 1, 2 * filter_size + 1).flatten()

    def dynamics(self, x):
        self.transformation = self.transformation * 0.96 + 0.04 * np.random.randn(2 * filter_size + 1, 2 * filter_size + 1).flatten()
        new_x = np.zeros((img_size, img_size))
        for i in range(img_size):
            for j in range(img_size):
                neighbor = x[max(0, i-filter_size):min(img_size, i+filter_size+1),
                             max(0, j-filter_size):min(img_size, j+filter_size+1)].flatten()
                new_x[i, j] = np.sum(neighbor * self.transformation[0:neighbor.size])
        return sigmoid(new_x)

automaton = Automaton()
plt.ion()
plt.show()
while True:
    plt.imshow(x, interpolation='none', cmap='Greys')
    plt.draw()
    plt.pause(0.5)
    x = automaton.dynamics(x)
