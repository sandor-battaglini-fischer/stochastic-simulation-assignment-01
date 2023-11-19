from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit
def mandelbrot(c, max_iter):
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
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((height, width))  
    for x in range(width):
        for y in range(height):
            n3[y, x] = mandelbrot(r1[x] + 1j*r2[y], max_iter)
    return n3

class StaticMandelbrotViewer:
    """
    A class for generating and viewing static Mandelbrot fractal images.

    Args:
        width (int): The width of the image in pixels. Default is 6000*2.
        height (int): The height of the image in pixels. Default is 5000*2.
        max_iter (int): The maximum number of iterations for determining the fractal. Default is 1000.

    Methods:
        generate_fractal(xmin, xmax, ymin, ymax):
            Generates the Mandelbrot fractal image within the specified bounds.

        display_and_save_figure(filename, dpi=1000):
            Displays and saves the generated fractal image.

        zoom_and_view(center_x, center_y, zoom_factor, filename, no_zoom=True):
            Zooms in or out on the fractal image and generates a new image.

    """

    def __init__(self, width=6000*2, height=5000*2, max_iter=1000):
        self.width, self.height = width, height
        self.max_iter = max_iter

    def generate_fractal(self, xmin, xmax, ymin, ymax):
        """
        Generates the Mandelbrot fractal image within the specified bounds (xmin, xmax, ymin, ymax).
        
        """
        start_time = timer()
        self.image = mandelbrot_set(xmin, xmax, ymin, ymax, self.width, self.height, self.max_iter)
        end_time = timer()
        print("Fractal generated in:", end_time - start_time, "seconds")

    def display_and_save_figure(self, filename, dpi=1000):
        """
        Displays and saves the generated fractal image.

        """
        plt.imshow(self.image, cmap='magma')
        plt.axis('off')
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)

    def zoom_and_view(self, center_x, center_y, zoom_factor, filename, no_zoom=True):
        """
        Zooms in or out on the fractal image and generates a new image.
        One can set no_zoom to false to zoom in on the fractal image.
        
        """
        if no_zoom:
            xmin, xmax, ymin, ymax = -2.0, 1.0, -1.25, 1.25
        else:
            scale = 1.5 / zoom_factor
            xmin = center_x - scale
            xmax = center_x + scale
            ymin = center_y - scale
            ymax = center_y + scale

        self.generate_fractal(xmin, xmax, ymin, ymax)
        self.display_and_save_figure(filename)

viewer = StaticMandelbrotViewer()
viewer.zoom_and_view(-0.7436438885706, 0.1318259043124, 74155, 'zoomed_mandelbrot.png', no_zoom=True)
