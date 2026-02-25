#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:39:36 2023

@author: gregory
"""

from Process.Solvers import SolverPDE
from Process.Tensors import TensorDiffusion
from Process.CTVs import CTV


class CTVDiffusion(CTV):

    def __init__(self, rts=None, tensor=None, model=None, imageArray=None, origin=(0, 0, 0), spacing=(1, 1, 1)):
        super().__init__(rts=rts, tensor=tensor, imageArray=imageArray, origin=origin, spacing=spacing)

        self._density3D = None

        self._model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, object):
        self._model = object

    @property
    def density3D(self):
        return self._density3D

    @density3D.setter
    def density3D(self, array):
        self._density3D = array

    def setCTV_isodensity(self, isodensity, domain=None):

        self.domain = domain

        if self.density3D is None:
            self.density3D = self.getDensity3D(self.gtv,
                                                 self.barriers,
                                                 self.preferred,
                                                 self.tensor,
                                                 self.model,
                                                 self.domain,)

        self.imageArray = self.density3D <= isodensity

    @staticmethod
    def getDensity3D(gtv, barriers, preferred, tensor, model, domain):

        from scipy.ndimage import gaussian_filter
        from opentps.core.data.images._image3D import Image3D

        # blur GTV to avoid sharp gradients
        source = Image3D(imageArray=gaussian_filter(gtv.imageArray.astype(float), sigma=1), origin=tensor.origin, spacing=tensor.spacing)

        if model['model'] is not None:
            if not isinstance(tensor, TensorDiffusion):
                tensor_inverse = tensor.getDiffusionTensor()  # get inverse
                diffusion = TensorDiffusion(imageArray=tensor_inverse.getDiffusionTensorFKPP(model), spacing=tensor.spacing, origin=tensor.origin)
            else:
                diffusion = TensorDiffusion(imageArray=tensor.getDiffusionTensorFKPP(model), spacing=tensor.spacing, origin=tensor.origin)
        else:
            diffusion = None

        # initialize solver
        solver = SolverPDE(source, barriers, diffusion, domain)

        # store 3D cell density estimation
        density3D = solver.getDensity(model)[-1]  # take last timepoint in list

        return density3D
