import numpy as np
import matplotlib.pyplot as plt
import time
import math

samples = 1000
simulations = 1

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
            n3[i, j] = mandelbrot(r1[i] + 1j*r2[j], max_iter)
    return (r1, r2, n3)

def calculate_area(xmin, xmax, ymin, ymax, z, max_iter):
    in_set = z == max_iter
    proportion = np.sum(in_set) / z.size
    return proportion * (xmax - xmin) * (ymax - ymin)

def orthogonal_sampling(xmax=1.5, xmin=-2.5, ymax=1.5j, ymin=-1.5j, samples=1000, max_iter=100, plot=False):
    x = np.linspace(xmin, xmax, samples+1)
    y = np.linspace((ymin*-1j).real, (ymax*-1j).real, samples+1)
    y = [(i * -1j) for i in y]
    y = y[::-1]

    x_rows = []
    y_rows = []
    for i in range(1, len(x)):
        x_rows.append([x[i - 1], x[i]])
        y_rows.append([y[i - 1], y[i]])

    A = np.zeros(samples ** 2).reshape(samples, samples)
    zx = np.arange(0, samples+math.sqrt(samples), math.sqrt(samples))
    zy = np.arange(0, samples, math.sqrt(samples))

    intervals = []
    for i in range(1, len(zx)):
        intervals.append([int(zx[i-1]), int(zx[i])])

    x_indices = [i for i in range(samples)]
    y_indices = [i for i in range(samples)]
    np.random.shuffle(x_indices)
    np.random.shuffle(y_indices)



    xlist = []
    ylist = []

    in_mandelbrot = 0
    start = time.time()

    for i in intervals:
        for j in intervals:
            x_temporary = [x for x in range(j[0], j[1])]
            y_temporary = [y for y in range(i[0], i[1])]
            cx = 0
            cy = 0
            for k in x_indices:
                cx = k
                x_indices.remove(k)
                break
            for k in y_indices:
                cy = k
                y_indices.remove(k)
                break

        A[cy][cx] = 1

        x_sample_range = x_rows[cx]
        y_sample_range = y_rows[-(cy+1)]

        xval = np.random.uniform(x_sample_range[0], x_sample_range[1])
        yval = np.random.uniform(y_sample_range[0], y_sample_range[1])

        xlist.append(xval)
        ylist.append((yval * -1j).real)


        sample = xval + yval

        print(A[(i[0]):(i[1]), (j[0]):(j[1])])
        print()

        z = np.array([mandelbrot(x + 1*y, max_iter) for x, y in zip(x_rows, y_rows)])
        area = calculate_area(xmin, xmax, ymin, ymax, z, max_iter)
        

    print(A)
    ylist = [y.real for y in ylist]

    if plot == True:
        plt.plot(xlist, ylist, 'o', markersize=6)
        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.title(f'Orthogonal Sampling (s={samples})')
        plt.show()

    return area

area_orth = orthogonal_sampling(-2.0, 1.0, -1.5, 1.5, 1000000, 256, plot=True)

print(f'Time elapsed for Orthogonal Sampling with s = {samples} , i = {max_iter} is {time.time() - start} seconds.')
print(f"Sample area = {sample_area}")

