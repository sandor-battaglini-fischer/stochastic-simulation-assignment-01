import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube
import time
import os
from numba import jit


@jit
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

@jit
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Generate a grid representation of the Mandelbrot set.

    Parameters:
    xmin, xmax, ymin, ymax (float): Coordinates defining the viewport of the Mandelbrot set.
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
def antithetic_random_sampling(xmin, xmax, ymin, ymax, n_samples, max_iter=256):
    rng = random.default_rng()
    x_samples = rng.random(n_samples // 2) * (xmax - xmin) + xmin
    y_samples = rng.random(n_samples // 2) * (ymax - ymin) + ymin
    y_samples_anti = ymax + ymin - y_samples
    x_samples = np.concatenate((x_samples, x_samples))
    y_samples = np.concatenate((y_samples, y_samples_anti))
    z = np.empty(n_samples)
    for i in range(n_samples):
        x = x_samples[i]
        y = y_samples[i]
        z[i] = mandelbrot(x + 1j*y, max_iter)
    area = calculate_area(xmin, xmax, ymin, ymax, z, max_iter)
    return x_samples, y_samples, z, area

@jit
def antithetic_latin_sampling(xmin, xmax, ymin, ymax, n_samples, max_iter=256):
    sampler = LatinHypercube(d=2)
    sample = sampler.random(n=n_samples//2)
    x_sample_nonanti = sample[:, 0] * (xmax - xmin) + xmin
    x_samples = np.concatenate((x_sample_nonanti, x_sample_nonanti))
    y_samples_nonanti = sample[:, 1] * (ymax - ymin) + ymin
    y_samples = np.concatenate((y_samples_nonanti, - y_samples_nonanti))
    z = np.array([mandelbrot(x + 1j*y, max_iter) for x, y in zip(x_samples, y_samples)])
    area = calculate_area(xmin, xmax, ymin, ymax, z, max_iter)
    return x_samples, y_samples, z, area

@jit
def antithetic_orthogonal_sampling(xmin, xmax, ymin, ymax, n_samples, max_iter=256):
    # Calculate grid size for antithetic sampling
    grid_size = int(np.sqrt(n_samples / 2))
    actual_n_samples = grid_size**2 * 2  # Actual number of samples used

    x_step = (xmax - xmin) / grid_size
    y_step = (ymax - ymin) / grid_size
    y_mid = (ymin + ymax) / 2

    x_samples = []
    y_samples = []
    z = []

    for i in range(grid_size):
        for j in range(grid_size):
            # Sample a point
            x_rand = xmin + i * x_step + np.random.uniform(0, x_step)
            y_rand = ymin + j * y_step + np.random.uniform(0, y_step)

            # Store the original point and its evaluation
            x_samples.append(x_rand)
            y_samples.append(y_rand)
            z.append(mandelbrot(x_rand + 1j*y_rand, max_iter))

            # Mirror the point
            y_mirror = y_mid - (y_rand - y_mid)
            x_samples.append(x_rand)
            y_samples.append(y_mirror)
            z.append(mandelbrot(x_rand + 1j*y_mirror, max_iter))

    z = np.array(z)
    area = calculate_area(xmin, xmax, ymin, ymax, z, max_iter)
    return np.array(x_samples), np.array(y_samples), z, area


# Sampling configurations
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
n_samples = 1000000
max_iter = 256
width, height = 800, 800

def main():
    # Create directory to store images
    directory_path = "/Users/sandor/dev/Computational Science/stochastic-simulation/stochastic-simulation-assignment-01/images"
    os.makedirs(directory_path, exist_ok=True)

    methods = {
        'Antithetic Random Sampling': antithetic_random_sampling,
        'Antithetic Latin Sampling': antithetic_latin_sampling,
        'Antithetic Orthogonal Sampling': antithetic_orthogonal_sampling
    }

    # Iterate through each sampling method and visualize the results
    for method_name, method_function in methods.items():
        plt.figure(figsize=(9, 9))
        x, y, z_set = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
        plt.pcolormesh(x, y, z_set.T, cmap='hot', shading='auto')

        # Measure the time taken by each sampling method
        start_time = time.time()
        x_samples, y_samples, z, area = method_function(xmin, xmax, ymin, ymax, n_samples, max_iter)
        elapsed_time = time.time() - start_time
        print(f"Method: {method_name}, Area: {area}, Time: {elapsed_time:.2f} seconds")

        # Scatter plot of samples over the Mandelbrot set
        plt.scatter(x_samples, y_samples, color='white', s=1, alpha=0.5)
        plt.title(f"{method_name} - Mandelbrot Set Sampling\nArea: {area:.4f}, Samples: {n_samples}, Time: {elapsed_time:.2f} seconds")
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.savefig(f"images/{method_name}.png", dpi=300)


if __name__ == '__main__':
    main()