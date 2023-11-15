import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    if n == max_iter:
        return max_iter
    return n + 1 - np.log(np.log2(abs(z)))

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    for i in range(width):
        for j in range(height):
            n3[i, j] = mandelbrot(r1[i] + 1j*r2[j],max_iter)
    return (r1, r2, n3)

def plot_mandelbrot(xmin, xmax, ymin, ymax, width=10, height=10, max_iter=256):
    dpi = 80
    img_width = dpi * width
    img_height = dpi * height
    x, y, z = mandelbrot_set(xmin, xmax, ymin, ymax, img_width, img_height, max_iter)
    
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.pcolormesh(x, y, z.T, cmap='magma', shading='auto')
    
    plt.show()

# Set the properties for the plot
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
width, height = 10, 10
max_iter = 256

# Generate and display the Mandelbrot set
plot_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
