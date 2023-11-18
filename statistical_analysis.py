import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import f, levene
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import math
import csv
import time

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

# Simulation parameters
simulations = 50
conf_level = 0.95

results = {
    "Random Sampling": [],
    "Antithetic Random Sampling": [],
    "LHS": [],
    "Antithetic Latin Sampling": [],
    "Orthogonal Sampling": [],
    "Antithetic Orthogonal Sampling": []
}
timings = {
    "Random Sampling": [],
    "Antithetic Random Sampling": [],
    "LHS": [],
    "Antithetic Latin Sampling": [],
    "Orthogonal Sampling": [],
    "Antithetic Orthogonal Sampling": []
}

# Prepare CSV file to record results
with open('sampling_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Simulation', 'Method', 'Area', 'Time'])

    # Running simulations for each method
    for n in range(1, simulations + 1):
        for method_name, method_function in [
            ("Random Sampling", random_sampling),
            ("Antithetic Random Sampling", ARS),
            ("LHS", LHS),
            ("Antithetic Latin Sampling", ALS),
            ("Orthogonal Sampling", orthogonal_sampling),
            ("Antithetic Orthogonal Sampling", AOS)
        ]:
            start_time = time.time()
            *_, sampled_area = method_function(xmin, xmax, ymin, ymax, n_samples, max_iter)
            elapsed_time = time.time() - start_time

            results[method_name].append(sampled_area)
            timings[method_name].append(elapsed_time)

            # Write individual simulation results to CSV
            writer.writerow([n, method_name, sampled_area, elapsed_time])
            print(f"Simulation {n}, Method: {method_name}, Area: {sampled_area:.4f}, Time: {elapsed_time:.4f} seconds")

    # Writing aggregated statistics to CSV
    writer.writerow([])
    writer.writerow(['Method', 'Mean Area', 'Variance', 'STD', '95% Confidence Interval', 'Average Time'])

    for method, values in results.items():
        mean_val = np.mean(values)
        var_val = np.var(values)
        std_val = np.std(values)
        avg_time = np.mean(timings[method])
        conf_interval = conf_int(mean_val, var_val, simulations, p=conf_level)

        # Write summary statistics to CSV
        writer.writerow([method, mean_val, var_val, std_val, conf_interval, avg_time])
        print(f"{method}: Mean Area = {mean_val:.4f}, Variance = {var_val:.4f}, STD = {std_val:.4f}, 95% CI = {conf_interval}, Average Time = {avg_time:.4f} seconds")

    # Perform Levene's test for homogeneity of variance
    areas_test_random = results['Random Sampling']
    areas_test_lhs = results['LHS']
    areas_test_orthogonal = results['Orthogonal Sampling']
    levene_value, levene_p = levene(areas_test_random, areas_test_lhs, areas_test_orthogonal)
    writer.writerow([])
    writer.writerow(['Levene\'s test for homogeneity of variance:'])
    writer.writerow(['Levene\'s value', levene_value, 'p value:', levene_p])

    #Perform Welsh's ANOVA for difference of means
    data = {
    'Area': areas_test_random + areas_test_lhs + areas_test_orthogonal,
    'Method': ['Random Sampling']*len(areas_test_random) + ['LHS']*len(areas_test_lhs) + ['Orthogonal Sampling']*len(areas_test_orthogonal)
    }
    data = pd.DataFrame(data)
    model = ols('Area ~ C(Method)', data=data).fit()
    ANOVA_results = sm.stats.anova_lm(model, typ=2)

    writer.writerow([])
    writer.writerow(['Welsh\'s ANOVA:'])
    writer.writerow([ANOVA_results.to_string()])

    #Perform F-test for Orthogonal Sampling with Antithetic Variates
    var_ortho = np.var(areas_test_orthogonal)
    var_ortho_anti = np.var(results['Antithetic Orthogonal Sampling'])
    F = var_ortho / var_ortho_anti
    dfn = len(areas_test_orthogonal) - 1
    dfd = len(results['Antithetic Orthogonal Sampling']) - 1
    p_value = f.sf(F, dfn, dfd)

    writer.writerow([])
    writer.writerow(['F-test for Orthogonal Sampling with Antithetic Variates:'])
    writer.writerow(['F value:', F, 'p value:', p_value])