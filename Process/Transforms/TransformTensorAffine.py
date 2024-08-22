#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:35:48 2024

@author: gregory
"""

import numpy as np
import scipy

from Process.Tensors import Tensor
from Process.Tensors import TensorMetric
from Process.Tensors import TensorDiffusion
from Process.Transforms import TransformTensor

class TransformTensorAffine(TransformTensor):
    def __init__(self, mapping):
        super().__init__(mapping=mapping)
        self._T_mov2static = None
        self._T_static2mov = None
        # Define attributes to be consistent with TransformTensorDeformable
        self.domain_shape = mapping.codomain_shape
        self.codomain_shape = mapping.domain_shape
        self.domain_grid2world = mapping.codomain_grid2world
        self.codomain_grid2world = mapping.domain_grid2world

    @property
    def T_mov2static(self):
        if self._T_mov2static is None:
            self._get_simplified_transform_forward()
        return self._T_mov2static

    @T_mov2static.setter
    def T_mov2static(self, array):
        self._T_mov2static = array

    @property
    def T_static2mov(self):
        if self._T_static2mov is None:
            self.T_static2mov = np.linalg.inv(self.T_mov2static)
        return self._T_static2mov

    @T_static2mov.setter
    def T_static2mov(self, array):
        self._T_static2mov = array

    @property
    def jacobianDomain(self):
        if self.jacDomain is None:
            self.jacDomain = np.tile(np.linalg.inv(self.T_mov2static)[0:3, 0:3], self.domain_shape + (1, 1))
        return self.jacDomain

    def jacobianCodomain(self):
        if self.jacCodomain is None:
            self.jacCodomain = np.tile(np.linalg.inv(self.T_mov2static)[0:3, 0:3], self.codomain_shape + (1, 1))
        return self.jacCodomain
  
    def _get_simplified_transform_forward(self):
        
        codomain_world2grid = np.linalg.inv(self.codomain_grid2world)

        # world to world (affine.affine is static2moving in world space)
        A = np.linalg.inv(self.mapping.affine)

        # transformation in grid space: grid space (moving) --> grid space (static)
        self.T_mov2static = np.dot(np.dot(self.domain_grid2world, A), codomain_world2grid)

    def getTensorMetricTransformed(self, tensor, mask=None):

        imageTensor = np.zeros(tuple(self.domain_shape) + (3, 3))  # initialize
        idX, idY, idZ = self._getIndices_codomain(mask) # use codomain function!
        
        # transform tensor
        imageTensor[idX, idY, idZ, ...] = np.matmul(self.transpose(self.jacobianDomain)[idX, idY, idZ, ...], np.matmul(tensor.imageArray[idX, idY, idZ, ...], self.jacobianDomain[idX, idY, idZ, ...]))
    
        # interpolate
        imageTensor = self._deformTensor(imageTensor)

        
        return TensorMetric(imageArray=imageTensor)
  
    def getTensorDiffusionTransformed(self, tensor, method='ICT', mask=None):
      
      if method == 'ICT':

        imageTensor = np.zeros(tuple(self.domain_shape) + (3, 3))  # initialize
        idX, idY, idZ = self._getIndices_codomain(mask) # use codomain function!
                 
        # transform tensor                     
        imageTensor[idX, idY, idZ, ...] = np.matmul(self.transpose(self.jacobianDomain)[idX, idY, idZ, ...], np.matmul(tensor.imageArray[idX, idY, idZ, ...], self.jacobianDomain[idX, idY, idZ, ...]))
  
        # interpolate
        imageTensor = self._deformTensor(imageTensor)

      elif method == 'PPD':

        imageTensor = np.zeros(tuple(self.codomain_shape) + (3, 3))  # initialize
        idX, idY, idZ = self._getIndices_domain(mask) # use domain function!
        
        # transform tensor and initialize tensor object
        tensorCodomain = Tensor(imageArray=self._deformTensor(tensor.imageArray))

        jacInv = self.T_static2mov[0:3, 0:3]
        for i in range(len(idX)):
            
            e1 = tensorCodomain.evecs[idX[i], idY[i], idZ[i], :, 0]
            e2 = tensorCodomain.evecs[idX[i], idY[i], idZ[i], :, 1]
            
            R = self.rotation_component_affine(jacInv, e1, e2)
  
            imageTensor[idX[i], idY[i], idZ[i], :, :] = np.dot(np.dot(R, tensorCodomain.imageArray[idX[i], idY[i], idZ[i], :, :]), R.T)
  
      elif method == 'FS':

          imageTensor = np.zeros(tuple(self.codomain_shape) + (3, 3))  # initialize
          idX, idY, idZ = self._getIndices_domain(mask) # use domain function!
          
          # transform tensor
          tensorCodomain = self._deformTensor(tensor.imageArray)

          # extract rotation matrix
          R = np.matmul(scipy.linalg.fractional_matrix_power(np.matmul(self.T_static2mov[0:3, 0:3], self.T_static2mov[0:3, 0:3].T), -1/2), self.T_static2mov[0:3, 0:3])

          imageTensor[idX, idY, idZ, ...] = np.matmul(np.matmul(R, tensorCodomain[idX, idY, idZ, ...]), R.T)
              
      return TensorDiffusion(imageArray=imageTensor)
  
    def getJacobianDeterminantCodomain(self):
          
        jacDetCodomain = np.linalg.det(self.T_static2mov[0:3, 0:3])*np.ones(self.codomain_shape)
        print('Jacobian determinant: ' + str(jacDetCodomain[0, 0, 0]))
        return jacDetCodomain

    # imageTensor = self._deformTensor(imageTensor, self.get_simplified_transform_forward(), output_shape=self.mapping.codomain_shape)
    #@staticmethod
    #def _deformTensor(tensor, T_grid2grid, output_shape=None):
        
        #xx = scipy.ndimage.affine_transform(tensor[:, :, :, 0, 0], np.linalg.inv(T_grid2grid), output_shape=output_shape)
        #xy = scipy.ndimage.affine_transform(tensor[:, :, :, 0, 1], np.linalg.inv(T_grid2grid), output_shape=output_shape)
        #xz = scipy.ndimage.affine_transform(tensor[:, :, :, 0, 2], np.linalg.inv(T_grid2grid), output_shape=output_shape)
        #yy = scipy.ndimage.affine_transform(tensor[:, :, :, 1, 1], np.linalg.inv(T_grid2grid), output_shape=output_shape)
        #yz = scipy.ndimage.affine_transform(tensor[:, :, :, 1, 2], np.linalg.inv(T_grid2grid), output_shape=output_shape)
        #zz = scipy.ndimage.affine_transform(tensor[:, :, :, 2, 2], np.linalg.inv(T_grid2grid), output_shape=output_shape)
        
        #tensorDeformed = np.array([[xx, xy, xz],
        #                           [xy, yy, yz],
        #                           [xz, yz, zz]])
        #tensorDeformed = np.transpose(tensorDeformed, [2, 3, 4, 0, 1])
        
        #return tensorDeformed
