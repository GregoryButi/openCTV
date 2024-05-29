#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:50:48 2024

@author: gregory
"""

import numpy as np
from Process.Tensors import Tensor
from Process.Tensors.TensorMetric import TensorMetric


class TensorDiffusion(Tensor):
    def __init__(self, imageArray=None, origin=(0, 0, 0), spacing=(1, 1, 1)):
        super().__init__(imageArray=imageArray, origin=origin, spacing=spacing)

    def getDiffusionTensorFKPP(self, domain, diffusion_magnitude=1., eps=1e-14):
        idX, idY, idZ = np.nonzero(domain)

        # compute normalization such that determinant of tensor is equal to one
        N = (self.evals[idX, idY, idZ, 0] * self.evals[idX, idY, idZ, 1] * self.evals[idX, idY, idZ, 2]) ** (-1 / 3)

        # scale eigenvalues
        evals = diffusion_magnitude * np.multiply(N[..., np.newaxis], self.evals[idX, idY, idZ])

        # recompute tensor
        imageTensor = self.reconstruct_tensor_mask(self.evecs, evals, idX, idY, idZ)

        # set values outside of domain to zero
        imageTensor[~domain] = np.eye(3) * eps

        return imageTensor

    def setBarrier(self, mask, eps=1e-14):
        self.imageArray[mask] = np.eye(3) * eps

    def getMetricTensorfromDiffusionTensor(self):
        mt = TensorMetric(imageArray=self.getInverse(), origin=self.origin, spacing=self.spacing)
        return mt
