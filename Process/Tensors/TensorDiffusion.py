#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:50:48 2024

@author: gregory
"""

import numpy as np

from Process.Tensors import Tensor

class TensorDiffusion(Tensor):
    def __init__(self, imageArray=None, origin=(0, 0, 0), spacing=(1, 1, 1)):
        super().__init__(imageArray=imageArray, origin=origin, spacing=spacing)

    def getDiffusionTensorFKPP(self, dict):

        # compute normalization such that determinant of tensor is equal to one
        N = (self.evals.prod(axis=-1)) ** (-1 / 3)

        # scale eigenvalues
        evals = dict['diffusion_magnitude'] * np.multiply(N[..., np.newaxis], self.evals)

        # recompute tensor
        DT = self.reconstruct_tensor(self.evecs, evals)

        return DT

    def setBarrier(self, mask, eps=1e-14):
        self.imageArray[mask] = np.eye(3) * eps

    def getMetricTensor(self):
        from Process.Tensors import TensorMetric
        mt = TensorMetric(imageArray=self.getInverse(), origin=self.origin, spacing=self.spacing)
        return mt
