#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:54:34 2023

@author: gregory
"""

import numpy as np
import copy
import random
import tqdm

from Process import Solver
from Process import dice_score, surface_dice_score

class SolverOptimizationContourSimilarity(Solver):
    def __init__(self, source=None, boundary=None, tensor=None, domain=None):
      super().__init__(source=source, boundary=boundary, tensor=tensor, domain=domain)


    def getDistance_metric(self, mask, distance3D, metric='volume', method='Nelder-Mead', x0=0, tolerance=0.01, divergence_threshold=1e3):

        def f_vol(x):
            relative_vol_diff = abs((distance3D <= x).sum() * self.source.spacing.prod() - mask.getVolume()) / mask.getVolume()
            return relative_vol_diff

        def f_dice(x):
            neg_dice = - dice_score(distance3D <= x, mask.imageArray)
            return neg_dice

        def f_surface_dice(x):
            neg_sdice = - surface_dice_score(distance3D <= x, mask.imageArray, 2, voxel_spacing=self.spacing)
            return neg_sdice

        if metric == 'volume':
            f = f_vol
        elif metric == 'dice':
            f = f_dice
        elif metric == 'surface_dice':
            f = f_surface_dice

        max_iter = 100
        if method == 'Secant':

            current_iter = copy.copy(max_iter)  # initialize
            max_tries = 100
            current_try = 0
            min_f = np.inf
            while current_iter == max_iter and current_try < max_tries:

                x1 = x0 + random.uniform(-x0 / 2, x0 / 2)  # starting point
                current_isodistance, current_f, current_iter = self.secant_method(f, x0, x1, tolerance, maxiter=max_iter, divergence_threshold=divergence_threshold)  # lower bound, upper bound, tolerance, iterations

                # Check if the current f(x1) is the smallest
                if current_f < min_f:
                    min_f = current_f
                    best_isodistance = current_isodistance  # Update the best x corresponding to min f(x1)

                if current_iter == max_iter:
                    print(f'WARNING: no solution found for try {current_try} within tolerance. Try with new starting point')
                else:
                    print('Optimal solution found in ' + str(current_iter) + ' iterations')

                current_try += 1

        elif method == 'Nelder-Mead' or method == 'Powell':

            from scipy.optimize import minimize

            # run optimization
            result = minimize(f, x0, method='Nelder-Mead', bounds=[(0, 1e3)], options={'maxiter': max_iter, 'disp': True})

            # store results
            best_isodistance = result.x[0]
            min_f = result.fun

        else:
            print('Provide valid method')

    @staticmethod
    def secant_method(f, x0, x1, tolerance, maxiter=100, divergence_threshold=1e3):
        """Return the root calculated using the secant method,
        the smallest f(x1), and the corresponding x1. Stop if x2 diverges."""
        with tqdm(total=maxiter, desc="Secant Method Progress", unit="step") as pbar:
            i = 0
            f_x0 = f(x0)
            f_x1 = f(x1)
            min_fx1 = abs(f_x1)  # Track the smallest f(x1)
            best_x = x1  # Track x1 corresponding to the smallest f(x1)

            while abs(f_x1) > tolerance and i < maxiter:
                # Update x2 using cached values of f(x0) and f(x1)
                try:
                    x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
                except ZeroDivisionError:
                    print("Encountered division by zero in the secant method.")
                    i = maxiter  # set i as maxiter
                    break

                f_x2 = f(x2)  # Evaluate f(x2)

                # Stop if x2 diverges
                if abs(x2) > divergence_threshold or x2 < 0:
                    print(f"Terminating early: x2={x2} (divergence detected)")
                    i = maxiter  # set i as maxiter
                    break

                # Update the smallest f(x1) if needed
                if abs(f_x2) < min_fx1:
                    min_fx1 = abs(f_x2)
                    best_x = x2

                # Shift variables for the next iteration
                x0, x1 = x1, x2
                f_x0, f_x1 = f_x1, f_x2

                i += 1
                # Update progress bar with current values
                pbar.set_postfix({"f(x1)": f_x1, "x1": x1})
                pbar.update(1)

        return best_x, min_fx1, i