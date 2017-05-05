from matplotlib import pyplot as plt
import numpy as np


# Generate dataset of symmetric images
def generate_symmetric(size):
    x = np.random.random((size, size))
    x = np.tril(x, k=-1) + np.transpose(np.tril(x))
    return x


def generate_double_symmetric(size):
    x = np.random.random((size, size))
    x = np.tril(x, k=-1) + np.transpose(np.tril(x))
    x = np.flipud(x)
    x = np.tril(x, k=-1) + np.transpose(np.tril(x))
    return x


class PatternDataset:
    def __init__(self, size=28, type='symmetric'):
        self.type = type
        self.size = size

    def next_batch(self, batch_size):
        data_batch = np.zeros((batch_size, self.size, self.size))
        for i in range(batch_size):
            data_batch[i, :, :] = generate_symmetric(self.size)
        return data_batch

if __name__ == '__main__':
    x = generate_double_symmetric(6)
    plt.imshow(x, interpolation='none', cmap='Greys')
    plt.show()
