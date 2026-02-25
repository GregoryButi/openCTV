#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:39:36 2023

@author: gregory
"""

import numpy as np
import random
import copy
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize, minimize_scalar

from Process.Solvers import SolverFMM, SolverDijkstra
from Process.Tensors import TensorMetric
from Process.CTVs import CTV
from Process.Analysis.contourComparison import dice_score, surface_dice_score

class CTVGeometric(CTV):

    def __init__(self, rts=None, tensor=None, model=None, imageArray=None, origin=(0, 0, 0), spacing=(1, 1, 1)):
        super().__init__(rts=rts, tensor=tensor, imageArray=imageArray, origin=origin, spacing=spacing)

        self._compute_distance_done = False
        self._distance3D = None
        self._isodistance = None
        self._volume = None

        self._moving = {'masks': [], 'movements': []}
        if rts is not None:
            self._moving['masks'] = rts.getMaskByType('Barrier_moving')
            self._moving['movements'] = [0. for _ in self._moving['masks']]

        self._model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, object):
        self._model = object

    @property
    def compute_distance_done(self):
        return self._compute_distance_done

    @compute_distance_done.setter
    def compute_distance_done(self, bool):
        self._compute_distance_done = bool

    @property
    def distance3D(self):
        if not self._compute_distance_done:

            if (not hasattr(self, 'solver')):
                if self.model['model'] is not None:
                    self.solver = 'FMM'
                    print("Using solver: FMM")
                else:
                    self.solver = None
                    print("Using default scipy solver")

            if not hasattr(self, 'domain'):
                self.domain = None

            distance3D = self.compute_distance3D(self.gtv,
                                                 self.barriers,
                                                 self.preferred,
                                                 self.tensor,
                                                 self.model,
                                                 self.domain,
                                                 self.solver
                                                 )

            # Set attributes
            self.compute_distance_done = True
            self._distance3D = distance3D

        return self._distance3D

    @distance3D.setter
    def distance3D(self, array):
        self._distance3D = array
        # Reset cached properties when distance3D changes
        self._volume = None

    @property
    def isodistance(self):
        return self._isodistance

    @isodistance.setter
    def isodistance(self, value):
        self._isodistance = value
        # Reset cached properties when isodistance changes
        self._volume = None

    @property
    def moving(self):
        return self._moving

    @moving.setter
    def moving(self, value):
        masks = value['masks']
        movements = value['movements']

        self._moving = {'masks': masks, 'movements': movements}

    @staticmethod
    def compute_distance3D(gtv, barriers, preferred, tensor, model, domain, solver, geodesicsROI=None):

        if model is None:

            # compute 3D Euclidian distance map
            distance3D = distance_transform_edt(~gtv.imageArray, sampling=gtv.spacing)

            return distance3D

        elif model['model'] is None:

            # compute 3D Euclidian distance map
            distance3D = distance_transform_edt(~gtv.imageArray, sampling=gtv.spacing)
            # remove barriers
            distance3D[barriers.imageArray] = np.inf

            return distance3D

        else:
            if model['model'] == 'Anisotropic' or model['model'] == 'Nonuniform':
                if tensor is None:
                    tensor = TensorMetric(imageArray=np.ones(tuple(gtv.gridSize) + (3, 3)), spacing=gtv.spacing, origin=gtv.origin)  # unit metric tensor
                    metric = TensorMetric(imageArray=tensor.getMetricTensorFMM(preferred, model), spacing=gtv.spacing, origin=gtv.origin)

                elif not isinstance(tensor, TensorMetric):
                    tensor_inverse = tensor.getMetricTensor()  # get inverse
                    metric = TensorMetric(imageArray=tensor_inverse.getMetricTensorFMM(preferred, model), spacing=gtv.spacing, origin=gtv.origin)

                else:
                    metric = TensorMetric(imageArray=tensor.getMetricTensorFMM(preferred, model), spacing=gtv.spacing, origin=gtv.origin)
            else:
                print('Defaulting to Uniform model')
                metric = None

            if solver == 'Dijkstra':
                solver = SolverDijkstra(source=gtv.copy(), boundary=barriers.copy(), tensor=metric, domain=domain)
            elif solver == 'FMM':
                solver = SolverFMM(source=gtv.copy(), boundary=barriers.copy(), tensor=metric, domain=domain)

            # run solver
            if geodesicsROI is not None:
                distance3D, flow3D, geodesics_world = solver.getDistance(roi=geodesicsROI)
                return distance3D, flow3D, geodesics_world
            else:
                distance3D = solver.getDistance()
                return distance3D

    def setCTV_isodistance(self, isodistance, solver='FMM', domain=None):

        # set attributes
        self.solver = solver
        self.domain = domain

        self.isodistance = isodistance
        self.imageArray = self.distance3D <= isodistance

    def setCTV_isodistance_movingBarriers(self, isodistance, distances, solver='FMM', domain=None):

        # set attributes
        self.solver = solver
        self.domain = domain
        self.isodistance = isodistance

        # Get sorted indices by decreasing magnitude
        sorted_indices = sorted(range(len(distances)), key=lambda i: abs(distances[i]), reverse=True)

        distances_sorted =  [distances[i] for i in sorted_indices]
        masks_sorted = [self.moving['masks'][i] for i in sorted_indices]

        def compute_sequential_expansion():
            barriers_minus_movingBarriers = self.barriers.copy()
            expansion = self.gtv.copy()
            expansion.imageArray = self.distance3D <= (self.isodistance - distances_sorted[0])
            distance_prev = distances_sorted[0]

            for i, mask in enumerate(masks_sorted):
                barriers_minus_movingBarriers.imageArray[mask.imageArray] = False
                distance3D = self.compute_distance3D(
                    expansion,
                    barriers_minus_movingBarriers,
                    self.preferred,
                    self.tensor,
                    self.model,
                    self.domain,
                    self.solver
                )
                if i + 1 < len(distances_sorted):
                    expansion.imageArray = distance3D <= (distance_prev - distances_sorted[i + 1])
                    distance_prev = distances_sorted[i + 1]

            final_distance3D = self.compute_distance3D(
                expansion,
                barriers_minus_movingBarriers,
                self.preferred,
                self.tensor,
                self.model,
                self.domain,
                self.solver
            )
            return final_distance3D <= distances_sorted[-1]

        self.imageArray = compute_sequential_expansion()
        self.compute_distance_done = False

    def getDistance3D_Euclidean(self, model={'obstacle': True}, domain=None, solver='FMM'):

        distance3D_Euclidean = self.compute_distance3D(self.gtv, self.barriers, None, None, model, domain, solver)

        return distance3D_Euclidean

    def setCTV_metric(self, mask_ref, metric='volume', domain=None, solver='FMM', method='Secant', x0=10, tolerance=0.01, divergence_threshold=1e3):

        # set attributes
        self.solver = solver
        self.domain = domain

        def f_vol(x):
            relative_vol_diff = abs((self.distance3D <= x).sum() * self.spacing.prod() - mask_ref.getVolume()) / mask_ref.getVolume()
            return relative_vol_diff

        def f_dice(x):
            neg_dice = - dice_score(self.distance3D <= x, mask_ref.imageArray)
            return neg_dice

        def f_surface_dice(x):
            neg_sdice = - surface_dice_score(self.distance3D <= x, mask_ref.imageArray, 2, voxel_spacing=self.spacing)
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

                x1 = x0 + random.uniform(-x0/2, x0/2) # starting point
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

            # run optimization
            result = minimize(f, x0, method=method, bounds=[(0, 1e3)], options={'maxiter': max_iter, 'disp': True})

            # store results
            best_isodistance = result.x[0]
            min_f = result.fun

        else:
            print('Provide valid method')

        print(f'Isodistance {best_isodistance:.2f} mm with min metric value {min_f:.2f} {metric}')

        # set attributes
        self.isodistance = best_isodistance
        self.imageArray = self.distance3D <= best_isodistance

    def correctCTV_metric_movingBarriers(self, mask_ref, metric='volume', method='Grid_search', num_grid_points=20):


        def f_dice(x):
            neg_dice = - dice_score(x, mask_ref.imageArray)
            return neg_dice

        def f_surface_dice(x):
            neg_sdice = - surface_dice_score(x, mask_ref.imageArray, 2, voxel_spacing=self.spacing)
            return neg_sdice

        if metric == 'dice':
            f = f_dice
        elif metric == 'surface_dice':
            f = f_surface_dice

        def compute_expansion(distance, mask):
            barriers_minus_movingBarriers = self.barriers.copy()
            barriers_minus_movingBarriers.imageArray[mask.imageArray] = False
            expansion = self.gtv.copy()
            expansion.imageArray = self.distance3D <= (self.isodistance - distance)

            final_distance3D = self.compute_distance3D(
                expansion,
                barriers_minus_movingBarriers,
                self.preferred,
                self.tensor,
                self.model,
                self.domain,
                self.solver
            )
            return final_distance3D <= distance

        distances = [0 for _ in self.moving['masks']]
        for i, mask in enumerate(self.moving['masks']):
            if method == 'Grid_search':

                best_loss = float('inf')
                best_value = 0.0
                grid = np.linspace(0, self.isodistance, num_grid_points)
                no_improvement_count = 0
                max_tries_without_improvement = np.inf  # You can tune this value

                for val in grid:
                    imageArray_candidate = compute_expansion(val, mask)
                    loss = f(imageArray_candidate)
                    if loss < best_loss:
                        best_loss = loss
                        best_value = val
                        no_improvement_count = 0  # Reset on improvement
                        print(f"New best loss found for {mask.name} = {val:.4f}, loss = {loss:.6f}")
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= max_tries_without_improvement:
                        print(f"Stopping early for {mask.name} after {no_improvement_count} non-improving steps.")
                        break

                distances[i] = best_value
                print(f"Grid search best for distance {mask.name} = {best_value}, loss = {best_loss}")

            else:

                print("Provide valid method")

        # Final computation using optimized distances
        self.setCTV_isodistance_movingBarriers(self.isodistance, distances)
        self.moving['movements'] = distances
        self.compute_distance_done = False

    def correctCTV_metric_softBarriers(self, mask_ref, metric='volume', method='Grid_search', x0=1, num_grid_points=20, bounds=(1., 100.)):

        def get_distance(rho, idx):
            preferred = self.preferred.copy()
            preferred['resistances'][idx] = rho
            return self.compute_distance3D(
                        self.gtv,
                        self.barriers,
                        preferred,
                        self.tensor,
                        self.model,
                        self.domain,
                        self.solver
            )

        def f_vol(x, i):
            d = get_distance(x, i)
            vol_pred = (d <= self.isodistance).sum() * np.prod(self.spacing)
            return abs(vol_pred - mask_ref.getVolume()) / mask_ref.getVolume()

        def f_dice(x, i):
            d = get_distance(x, i)
            return -dice_score(d <= self.isodistance, mask_ref.imageArray)

        def f_surface_dice(x, i):
            d = get_distance(x, i)
            return -surface_dice_score(d <= self.isodistance, mask_ref.imageArray, 2, voxel_spacing=self.spacing)

        # Select metric function
        metric_funcs = {
            'volume': f_vol,
            'dice': f_dice,
            'surface_dice': f_surface_dice
        }

        if metric not in metric_funcs:
            raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(metric_funcs)}.")

        f = metric_funcs[metric]
        min_metric = float('inf')

        for i, mask in enumerate(self.preferred['masks']):
            print(f"Optimizing resistance for structure {mask.name}...")

            if method == 'Grid_search':
                best_val = bounds[0]
                grid = np.linspace(*bounds, num_grid_points)
                no_improvement_count = 0
                max_tries_without_improvement = np.inf  # You can adjust this value

                for val in grid:
                    loss = f(val, i)
                    if loss < min_metric:
                        min_metric = loss
                        best_val = val
                        no_improvement_count = 0  # Reset on improvement
                        print(f"  New best loss: resistance={val:.2f}, loss={loss:.6f}")
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= max_tries_without_improvement:
                        print(f"  Stopping early after {no_improvement_count} non-improving steps.")
                        break

            elif method == 'Minimize_scalar':
                res = minimize_scalar(lambda x: f(x, i), bounds=bounds, method='bounded')
                best_val = res.x
                min_metric = res.fun
                print(f"  Optimized resistance={best_val:.4f}, loss={min_metric:.6f}")

            else:
                raise ValueError(f"Unsupported method '{method}'.")

            self.preferred['resistances'][i] = best_val

        print(f"Final resistances: {self.preferred['resistances']} with min {metric} loss = {min_metric:.4f}")

        # Final update
        self.distance3D = self.compute_distance3D(
                    self.gtv,
                    self.barriers,
                    self.preferred,
                    self.tensor,
                    self.model,
                    self.domain,
                    self.solver
        )
        self.imageArray = self.distance3D <= self.isodistance

    def getGeodesicsROI(self, ROI):

        _, flow3D, geodesics_world = self.compute_distance3D(self.gtv, self.barriers, self.preferred, self.tensor, self.model, self.domain, self.solver, geodesicsROI=ROI.copy())

        # Convert surface points to voxel indices (assuming they are already in voxel space) with nearest neighbour approach
        voxel_coords = np.empty((3, 0), int)
        for geo in geodesics_world:
            voxel_coords = np.concatenate((voxel_coords, np.round((geo - self.origin[..., np.newaxis]) / self.spacing[..., np.newaxis]).astype(int)), axis=1)  # Convert to nearest voxel index

        # Ensure indices are within bounds
        valid_mask = (
                (voxel_coords[0, :] >= 0) & (voxel_coords[0, :] < self.gridSize[0]) &
                (voxel_coords[1, :] >= 0) & (voxel_coords[1, :] < self.gridSize[1]) &
                (voxel_coords[2, :] >= 0) & (voxel_coords[2, :] < self.gridSize[2])
        )

        # Assign distances to corresponding voxel locations
        flow3D_ROI = (np.ones(self.gridSize)*np.inf, np.ones(self.gridSize)*np.inf, np.ones(self.gridSize)*np.inf)
        for i in range(3):
            flow3D_ROI[i][voxel_coords[0, valid_mask], voxel_coords[1, valid_mask], voxel_coords[2, valid_mask]] = \
                flow3D[i][voxel_coords[0, valid_mask], voxel_coords[1, valid_mask], voxel_coords[2, valid_mask]]

        return flow3D_ROI, geodesics_world

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
