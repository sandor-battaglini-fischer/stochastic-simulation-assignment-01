import numpy as np
import scipy.stats as st
import math
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from methods import random_sampling, latin_sampling_scipy as LHS, orthogonal_sampling
from antithetic import antithetic_random_sampling as ARS, antithetic_latin_sampling as ALS, antithetic_orthogonal_sampling as AOS
from methods import xmax, xmin, ymax, ymin, n_samples, max_iter
import time

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


def plot_convergence(xmin, xmax, ymin, ymax, max_iter, max_samples, simulations=10, p_value=0.95, colors=None):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Palatino"]})


    if colors is None:
        colors = sns.color_palette("tab10", n_colors=3)

    sampling_methods = {
        'Random Sampling': random_sampling,
        'Latin Hypercube Sampling': LHS,
        'Orthogonal Sampling': orthogonal_sampling
    }

    sample_sizes = np.logspace(2, 6, num=10, base=10, dtype=int)

    plt.figure(figsize=(12, 8))

    for i, (method_name, method_function) in enumerate(sampling_methods.items()):
        mean_areas = []
        conf_intervals = []

        for n_samples in sample_sizes:
            areas = []
            for _ in range(simulations):
                _, _, _, area = method_function(xmin, xmax, ymin, ymax, n_samples, max_iter)
                areas.append(area)

            mean_area = np.mean(areas)
            var_area = np.var(areas, ddof=1)
            conf_int_low, conf_int_high = map(float, conf_int(mean_area, var_area, simulations, p=p_value)[1:-1].split(', '))

            mean_areas.append(mean_area)
            conf_intervals.append((conf_int_low, conf_int_high))

        mean_areas = np.array(mean_areas)
        lower_bounds = np.array([ci[0] for ci in conf_intervals])
        upper_bounds = np.array([ci[1] for ci in conf_intervals])

        plt.plot(sample_sizes, mean_areas, marker='o', color=colors[i % len(colors)], label=f"{method_name} (Convergence: {mean_areas[-1]:.6f})")
        plt.fill_between(sample_sizes, lower_bounds, upper_bounds, color=colors[i % len(colors)], alpha=0.2)

    plt.xscale('log')
    plt.xlabel('Number of Samples (log scale)', fontsize=14)
    plt.ylabel('Estimated Area of Mandelbrot Set', fontsize=14)
    plt.title('Convergence of Estimated Area with Increasing Samples', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence_plot_with_convergence_in_legend.png', dpi=300)
    plt.show()

def plot_time_per_simulation(xmin, xmax, ymin, ymax, max_iter, simulations=10, colors=None):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Palatino"]})

    if colors is None:
        colors = sns.color_palette("tab10", n_colors=3)

    sampling_methods = {
        'Random Sampling': random_sampling,
        'Latin Hypercube Sampling': LHS,
        'Orthogonal Sampling': orthogonal_sampling
    }

    sample_sizes = np.logspace(2, 6, num=10, base=10, dtype=int)

    plt.figure(figsize=(12, 8))

    for i, (method_name, method_function) in enumerate(sampling_methods.items()):
        avg_times = []

        for n_samples in sample_sizes:
            start_time = time.time()
            for _ in range(simulations):
                method_function(xmin, xmax, ymin, ymax, n_samples, max_iter)
            end_time = time.time()
            avg_time = (end_time - start_time) / simulations
            avg_times.append(avg_time)

        plt.plot(sample_sizes, avg_times, marker='o', color=colors[i % len(colors)], label=method_name)

    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Number of Samples', fontsize=14)
    plt.ylabel('Average Time per Simulation', fontsize=14)
    plt.title('Average Time per Simulation vs. Number of Samples', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig('time_per_simulation_plot.png', dpi=300)
    plt.show()

xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
max_iter = 256
max_samples = 10000
# plot_convergence(xmin, xmax, ymin, ymax, max_iter, max_samples)
plot_time_per_simulation(xmin, xmax, ymin, ymax, max_iter)
