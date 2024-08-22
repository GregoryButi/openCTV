#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:54:34 2023

@author: gregory
"""

import numpy as np
from agd import Eikonal
from agd.Metrics import Riemann

from Process.Solvers import Solver
from Process.Tensors import TensorMetric

class SolverFMM(Solver):
    def __init__(self, source=None, barriers=None, tensor=None, domain=None):
      
      if tensor is None:
          MT = np.zeros(tuple(source.gridSize) + (3, 3))
          MT[..., 0:3, 0:3] = np.eye(3)          
          tensor = TensorMetric(imageArray=MT, spacing=source.spacing, origin=source.origin)
    
      super().__init__(source=source, barriers=barriers, tensor=tensor, domain=domain)

    def getDistance(self, dict):

        # Define target / seeds
    
        target = self.source.imageArray
    
        hfmIn = Eikonal.dictIn({
            'model': 'Riemann3',  # Three-dimensional Riemannian eikonal equation
        })
    
        # Define calculation grid in world space
        
        X_world, Y_world, Z_world = self.source.getMeshGridPositions()
        xvec_world, yvec_world, zvec_world = self.source.getMeshGridAxes()
    
        hfmIn.SetRect(sides=[[xvec_world[0], xvec_world[-1] + self.source.spacing[0]], [yvec_world[0], yvec_world[-1] +
                      self.source.spacing[1]], [zvec_world[0], zvec_world[-1] + self.source.spacing[2]]], gridScales=self.source.spacing)
        hfmIn['order'] = 2
    
        # X,Y,Z = hfmIn.Grid()
    
        target_points = np.concatenate((np.expand_dims(X_world[target], 1), np.expand_dims(Y_world[target], 1), np.expand_dims(Z_world[target], 1)), axis=1)
        ntarget = target_points.shape[0]
    
        # Define hfm model
    
        hfmIn.update({
            'seeds': target_points,  # Introduce seeds
            'seedValues': np.zeros(ntarget), # Boundary conditions imposed at the seeds.
            'exportGeodesicFlow': True,  # Export relevant data
            'exportValues': True
        })
    
        # initialize metric tensor components
    
        MTxx = self.tensor.imageArray[..., 0, 0]
        MTyy = self.tensor.imageArray[..., 1, 1]
        MTzz = self.tensor.imageArray[..., 2, 2]
        MTxy = self.tensor.imageArray[..., 0, 1]
        MTxz = self.tensor.imageArray[..., 0, 2]
        MTyz = self.tensor.imageArray[..., 1, 2]
    
        # define metric tensor
    
        MT = [[MTxx, MTxy, MTxz],
              [MTxy, MTyy, MTyz],
              [MTxz, MTyz, MTzz]]
                            
        # Run Riemann model

        hfmIn['walls'] = self.barriers.imageArray
        hfmIn['metric'] = Riemann(MT)
        #hfmIn['stopAtDistance'] = 20.
        #hfmIn['euclideanScale'] = list(self.source.spacing)
        #hfmIn['stopAtEuclideanLength'] = 20. # stopping criteria
        hfmOut = hfmIn.Run()
    
        # store result
        riemann3D = hfmOut['values']
        # euclidean3D = hfmOut['euclideanLengths'] # Euclidean length of geodesics
        return riemann3D