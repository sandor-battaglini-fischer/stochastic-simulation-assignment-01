# Stochastic Simulation Assignment 01 - Calculating the area of the Mandelbrot set

### License:
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains Python scripts developed for a the first assignment of Stochastic Simulation at the University of Amsterdam, 2023. The project focuses on the calculation of the area of the Mandelbrot set using Monte Carlo methods.

## Files Description

### `methods.py`
This file includes the core methodss used in this project. It contains the implementation of the basic Mandelbrot iteration, functions to calculate the area and the three basic sampling techniques. It serves as the backbone for the simulations in other scripts and run by itself it creates plots of the fractals and where each sample lands for each sampling method.

### `antithetic.py`
`antithetic.py` implements the antithetic equivalents of the sampling techniques, which is a variance reduction method in stochastic simulation. Otherwise, it has the same functionality as `methods.py`.

### `statistical_analysis.py`
This script is dedicated to performing statistical analysis on the data generated by the simulations. It imports the methods from `methods.py` and `antithetic.py` and includes functions for calculating key statistical measures, such as mean, variance, and confidence intervals. It outputs the data in .csv format.

### `sample_convergence.py`
In `sample_convergence.py`, the focus is on analyzing the convergence behavior of simulations as the sample number increases while keeping the iteration number constant. It outputs a plot of the convergence behaviour of the sampling methods chosen and usilises the statistics from `statistical_analysis.py` for the confidence intervals.

### `iteration_convergence.py`
Analogous to `sample_convergence.py` This file examines the convergence properties in relation to the number of iterations, keeping the sample size constant.

###  `viewer.py`
This file has the ability to visualise the fractal in different colormaps and can zoom in on specific areas. It's purely for visualisation purposes and has no use for the calculations.

### `orthogonal.py`
`orthogonal.py` is a file to play around with and was created to run the simulation at the highest settings.

## Usage
These scripts were run with Python 3.11.0 on MacOS Ventura. 

### Requirements:
matplotlib==3.7.1
numba==0.57.0
numpy==1.24.3
pandas==1.5.3
scipy==1.10.1
seaborn==0.13.0
statsmodels==0.14.0

## Contact
joana.costaesilva@student.uva.nl
balint.szarvas@student.uva.nl
sandor.battaglini-fischer@student.uva.nl

---

Developed by Joana Costa e Silva, Bálint Szarvas and Sándor Battaglini-Fischer.
