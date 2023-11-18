import numpy as np
import scipy.stats as st
import math
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from methods import random_sampling, latin_sampling_scipy as LHS, orthogonal_sampling
from antithetic import antithetic_random_sampling as ARS, antithetic_latin_sampling as ALS, antithetic_orthogonal_sampling as AOS
from methods import xmax, xmin, ymax, ymin, n_samples, max_iter

def conf_int(mean, var, n, p=0.95):
    """
    Calculate the confidence interval for a given mean, variance, and sample size.

    Parameters:
    mean (float): Mean value of the data.
    var (float): Variance of the data.
    n (int): Sample size.
    p (float): Confidence level (default is 0.95 for 95%).

    Returns:
    str: Confidence interval represented as a string.
    """
    pnew = (p + 1) / 2
    zval = st.norm.ppf(pnew)
    sigma = math.sqrt(var)
    alambda = (zval * sigma) / math.sqrt(n)
    min_lambda = mean - alambda
    plus_lambda = mean + alambda
    return f"[{min_lambda:.4f}, {plus_lambda:.4f}]"


sns.set(style="whitegrid")
colors = sns.color_palette("pastel")


def plot_convergence_iterations(xmin, xmax, ymin, ymax, n_samples, iteration_steps, simulations=30, conf_level=0.95, colors=None):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Palatino"]})


    if colors is None:
        colors = sns.color_palette("tab10", n_colors=3)

    sampling_methods = {
        'Random Sampling': random_sampling,
        'Latin Hypercube Sampling': LHS,
        'Orthogonal Sampling': orthogonal_sampling
    }

    iteration_counts = iteration_steps

    plt.figure(figsize=(12, 8))

    for i, (method_name, method_function) in enumerate(sampling_methods.items()):
        mean_areas = []
        conf_intervals = []

        for max_iter in iteration_counts:
            areas = []
            for _ in range(simulations):
                _, _, _, area = method_function(xmin, xmax, ymin, ymax, n_samples, max_iter)
                areas.append(area)

            mean_area = np.mean(areas)
            var_area = np.var(areas, ddof=1)
            conf_int_low, conf_int_high = map(float, conf_int(mean_area, var_area, simulations, p=conf_level)[1:-1].split(', '))

            mean_areas.append(mean_area)
            conf_intervals.append((conf_int_low, conf_int_high))

        mean_areas = np.array(mean_areas)
        lower_bounds = np.array([ci[0] for ci in conf_intervals])
        upper_bounds = np.array([ci[1] for ci in conf_intervals])

        plt.plot(iteration_counts, mean_areas, marker='o', color=colors[i % len(colors)], label=f"{method_name}")
        plt.fill_between(iteration_counts, lower_bounds, upper_bounds, color=colors[i % len(colors)], alpha=0.2)

    plt.xlabel('Number of Iterations', fontsize=14)
    plt.ylabel('Estimated Area of Mandelbrot Set', fontsize=14)
    plt.title('Convergence of Estimated Area with Increasing Iterations', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence_plot_with_iterations.png', dpi=300)
    plt.show()

iteration_steps = np.linspace(50, 10000, num=100, dtype=int) 
n_samples = 1000000
plot_convergence_iterations(xmin, xmax, ymin, ymax, n_samples, iteration_steps)