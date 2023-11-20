import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube
import time
import os
from numba import jit, prange
import time


@jit(nopython=True, parallel=True)
def mandelbrot(c, max_iter):
    """
    Calculate the Mandelbrot set iteration count for a given complex number.

    Parameters:
    c (complex): Complex number to calculate Mandelbrot iteration for.
    max_iter (int): Maximum number of iterations.

    Returns:
    int: Mandelbrot set iteration count for the complex number.
    """
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    if n == max_iter:
        return max_iter
    return n + 1 - np.log(np.log2(abs(z)))


@jit(nopython=True, parallel=True)
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Generate a grid representation of the Mandelbrot set.

    Parameters:
    xmin, xmax, ymin, ymax: Coordinates of the viewport.
    width, height (int): Dimensions of the grid.
    max_iter (int): Maximum number of iterations for Mandelbrot calculation.

    Returns:
    tuple: (x-coordinates, y-coordinates, Mandelbrot set grid).
    """
    # Generate linearly spaced ranges for real and imaginary parts
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    # Calculate Mandelbrot set for each point in the grid
    for i in range(width):
        for j in range(height):
            n3[i, j] = mandelbrot(r1[i] + 1j*r2[j], max_iter)
    return (r1, r2, n3)


@jit
def calculate_area(xmin, xmax, ymin, ymax, z, max_iter):
    """Calculate the area of the Mandelbrot set."""
    in_set = z == max_iter
    proportion = np.sum(in_set) / z.size
    return proportion * (xmax - xmin) * (ymax - ymin)


@jit
def orthogonal_sampling(xmin, xmax, ymin, ymax, n_samples, max_iter=256):
    """Sample the Mandelbrot set using orthogonal sampling."""
    grid_size = int(np.sqrt(n_samples))
    actual_n_samples = grid_size**2
    # if grid_size**2 != n_samples:
    #     raise ValueError("n_samples must be a perfect square for orthogonal sampling.")

    x_step = (xmax - xmin) / grid_size
    y_step = (ymax - ymin) / grid_size

    x_samples = []
    y_samples = []
    z = []

    x_set = set()
    y_set = set()

    for i in range(grid_size):
        for j in range(grid_size):
            x_rand = xmin + i * x_step + np.random.uniform(0, x_step)
            while x_rand in x_set:
                x_rand = xmin + i * x_step + np.random.uniform(0, x_step)

            y_rand = ymin + j * y_step + np.random.uniform(0, y_step)
            while y_rand in y_set:
                y_rand = ymin + j * y_step + np.random.uniform(0, y_step)

            x_samples.append(x_rand)
            y_samples.append(y_rand)
            z.append(mandelbrot(x_rand + 1j*y_rand, max_iter))

            x_set.add(x_rand)
            y_set.add(y_rand)

    z = np.array(z)
    area = calculate_area(xmin, xmax, ymin, ymax, z, max_iter)
    return np.array(x_samples), np.array(y_samples), z, area


def calculate_statistics(areas):
    """
    Calculate statistical measures including confidence bounds.
    """
    mean = np.mean(areas)
    variance = np.var(areas)
    std_dev = np.sqrt(variance)
    conf_interval = 1.96 * std_dev / np.sqrt(len(areas))  # 95% confidence interval

    lower_bound = mean - conf_interval
    upper_bound = mean + conf_interval

    return {
        'mean': mean,
        'variance': variance,
        'standard_deviation': std_dev,
        'confidence_interval': conf_interval,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
n_samples = 100000000
max_iter = 1000000

time_start = time.time()
x_samples, y_samples, z, area = orthogonal_sampling(xmin, xmax, ymin, ymax, n_samples, max_iter)
stats = calculate_statistics(z == max_iter)
print(f"Area: {area:.20f}")
print("Samples:", n_samples)
print("Iterations:", max_iter)
print("Time:", time.time() - time_start)
print(stats)
 