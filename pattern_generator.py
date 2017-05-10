from matplotlib import pyplot as plt
import numpy as np
import math

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


def generate_striped(size):
    x = np.random.random((size, size))
    gx, gy = np.meshgrid(np.linspace(0, 1.0, size), np.linspace(0, 1.0, size))
    z = np.sin(5 * (gx + 2 * gy))
    return x * z


def generate_varied_strips(size):
    x = np.random.random((size, size))
    gx, gy = np.meshgrid(np.linspace(0, 1.0, size), np.linspace(0, 1.0, size))
    frequency = np.random.random() * 4.0 + 5.0
    if np.random.random() > 0.5:
        frequency *= -1.0
    if np.random.random() > 0.5:
        phase = math.pi / 2
    else:
        phase = 0.0
    x_ratio = (np.random.random() * 0.8 + 0.1) / 0.3
    if np.random.random() > 0.5:
        x_ratio *= -1
    z = np.sin(frequency * (gx + x_ratio * gy) + phase)
    return x * z


# GANs may be very bad at counting, construct a dataset with a constant number of random dots
def generate_dots(size):
    x = np.random.random((size, size))


class PatternDataset:
    def __init__(self, size=28, type='symmetric'):
        self.type = type
        self.size = size
        if type == 'symmetric':
            self.gen = generate_symmetric
        elif type == 'stripe':
            self.gen = generate_striped
        elif type == 'vstripe':
            self.gen = generate_varied_strips
        else:
            print("Unknown pattern type")
            exit(-1)

    def next_batch(self, batch_size):
        data_batch = np.zeros((batch_size, self.size, self.size))
        for i in range(batch_size):
            data_batch[i, :, :] = self.gen(self.size)
        return data_batch


if __name__ == '__main__':
    for i in range(16):
        x = generate_varied_strips(20)
        plt.subplot(4, 4, i+1)
        plt.imshow(x, interpolation='none', cmap='Greys')
    plt.show()
