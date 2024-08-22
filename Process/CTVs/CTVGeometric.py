#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:39:36 2023

@author: gregory
"""

from Process.Solvers import SolverFMM
from Process.Tensors import TensorMetric
from Process.CTVs import CTV

class CTVGeometric(CTV):

    def __init__(self, imageArray=None, origin=(0, 0, 0), spacing=(1, 1, 1)):
        super().__init__(imageArray=imageArray, origin=origin, spacing=spacing)

        self._distance3D = None
        self._isodistance = None
        self._volume = None

    @property
    def distance3D(self, rts=None, tensor=None, model=None, domain=None):
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


    def setCTV_volume(self, volume, rts=None, tensor=None, model=None, domain=None, x0=10):

        import random

        if self.distance3D is None:
            self.distance3D = self.getDistance3D(rts, tensor, model, domain)

        def secant_method(f, x0, x1, tolerance=0.1, maxiter=100):

            """Return the root calculated using the secant method."""
            i = 0
            x2 = x1
            while abs(f(x1)) / volume > tolerance and i < maxiter:
                x2 = x1 - f(x1) * (x1 - x0) / float(f(x1) - f(x0))
                x0, x1 = x1, x2
                i += 1
            return x2, i

        def f(x):
            vol_diff = (self.distance3D <= x).sum() * self.spacing.prod() - volume
            return vol_diff

        maxiter = 100
        iter_final = 100  # initialize
        while iter_final == maxiter:

            x1 = x0 + random.randint(-10, 10)
            margin_final, iter_final = secant_method(f, x0, x1, tolerance=0.01, maxiter=maxiter)  # lower bound, upper bound, tolerance, iterations

            if iter_final == maxiter:
                print('WARNING: no solution found within tolerance. Try with new starting point')
            else:
                print('Optimal solution found in ' + str(iter_final) + ' iterations')

        self.isodistance = margin_final
        self.imageArray = self.distance3D <= margin_final

    def setCTV_isodistance(self, isodistance, rts=None, tensor=None, model=None, domain=None):

        if self.distance3D is None:
            self.distance3D = self.getDistance3D(rts, tensor, model, domain)
        self.isodistance = isodistance
        self.imageArray = self.distance3D <= isodistance

    @staticmethod
    def getDistance3D(rts, tensor, model, domain):

        # load GTV
        source = rts.getMaskByName('GTV')

        if model is None:

            from scipy.ndimage import distance_transform_edt

            # compute 3D Euclidian distance map
            distance3D = distance_transform_edt(~source.imageArray, sampling=source.spacing)

            # set value in barrier structures
            distance3D[rts.getMaskByName('BS').imageArray] = 1e+14


        else:
            if model['obstacle']:
                barriers = rts.getMaskByName('BS')
            else:
                barriers = None
            if model['model'] is not None:
                if not isinstance(tensor, TensorMetric):
                    # get inverse
                    tensor_inverse = tensor.getMetricTensor()
                    metric = TensorMetric(imageArray=tensor_inverse.getMetricTensorFMM(rts.getMaskByName('PS').imageArray, model), spacing=tensor.spacing, origin=tensor.origin)
                else:
                    metric = TensorMetric(imageArray=tensor.getMetricTensorFMM(rts.getMaskByName('PS').imageArray, model), spacing=tensor.spacing, origin=tensor.origin)
            else:
                metric = None

            solver = SolverFMM(source, barriers, metric, domain)

            distance3D = solver.getDistance(model)

        return distance3D
