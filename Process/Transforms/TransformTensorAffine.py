#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:35:48 2024

@author: gregory
"""

import numpy as np
import scipy

from Process.Tensors import Tensor
from Process.Transforms import TransformTensor

class TransformTensorAffine(TransformTensor):
    def __init__(self, mapping):
        super().__init__(mapping=mapping)

    @property
    def jacAffine(self):
        if self.jac is None:
            self.jac = np.tile(np.linalg.inv(self._compute_transformation_matrix())[0:3, 0:3], self.mapping.codomain_shape + (1, 1))
        return self.jac
  
    def _compute_transformation_matrix(self):
        
        domain_world2grid = np.linalg.inv(self.mapping.domain_grid2world)

        # world to world (affine.affine is static2moving in world space)
        A = np.linalg.inv(self.mapping.affine)

        # transformation in grid space: grid space (moving) --> grid space (static)
        T_mov2static = np.dot(np.dot(domain_world2grid,A),self.mapping.codomain_grid2world)  
        
        return T_mov2static
    
    def getTensorMetricTransformed(self, tensor, mask=[]):
        
        if mask == []:
            idX, idY, idZ = np.nonzero(np.ones(self.mapping.codomain_shape))
        else:
            idX, idY, idZ = np.nonzero(mask)
            
        # initialize
        imageTensor = np.zeros(tensor.gridSize)  
        
        # transform tensor
        imageTensor[idX, idY, idZ, ...] = np.matmul(self.transpose(self.jacAffine)[idX, idY, idZ, ...], np.matmul(tensor.imageArray[idX, idY, idZ, ...], self.jacAffine[idX, idY, idZ, ...]))
    
        # interpolate
        imageTensor = self.deformTensors(imageTensor, self._compute_transformation_matrix(), output_shape=self.mapping.domain_shape)
        
        return Tensor(imageArray=imageTensor)    
  
    def getTensorDiffusionTransformed(self, tensor, method='ICT', mask=[]):
          
      if mask == []:
          idX, idY, idZ = np.nonzero(np.ones(self.mapping.codomain_shape))
      else:
          idX, idY, idZ = np.nonzero(mask)
          
      # initialize
      imageTensor = np.zeros(tensor.gridSize)  
      
      if method == 'ICT':
                 
        # transform tensor                     
        imageTensor[idX, idY, idZ, ...] = np.matmul(self.transpose(self.jacAffine)[idX, idY, idZ, ...], np.matmul(tensor.imageArray[idX, idY, idZ, ...], self.jacAffine[idX, idY, idZ, ...]))
  
        # interpolate
        imageTensor = self.deformTensors(imageTensor, self._compute_transformation_matrix(), output_shape=self.mapping.domain_shape)
        
      elif method == 'PPD':
        
        # transform tensor image to domain grid and initialize tensor object
        tensorDomain = Tensor(imageArray=self.deformTensors(tensor.imageArray, self._compute_transformation_matrix(), output_shape=self.mapping.domain_shape), type='diffusion')                    
        
        jacInv = self.invert(self.jacAffine)
        for i in range(len(idX)):
            
            jac_voxel = jacInv[idX[i], idY[i], idZ[i], :, :]
            e1 = tensorDomain.evecs[idX[i], idY[i], idZ[i], :, 0]
            e2 = tensorDomain.evecs[idX[i], idY[i], idZ[i], :, 1]
            
            R = self.rotation_component_affine(jac_voxel, e1, e2)
  
            imageTensor[idX[i], idY[i], idZ[i], :, :] = np.dot(np.dot(R, tensorDomain.imageArray[idX[i], idY[i], idZ[i], :, :]), R.T)
  
      elif method == 'FS':
          
          # transform tensor image to domain grid
          imageTensorDomain = self.deformTensors(tensor.imageArray, self._compute_transformation_matrix(), output_shape=self.mapping.domain_shape)     
      
          # extract rotation matrix
          R = self.rigid_rotation_component(self.invert(self.jacAffine)[0,0,0,...], self.transpose(self.invert(self.jacAffine))[0,0,0,...])
          
          imageTensor[idX,idY,idZ,...] = np.matmul(np.matmul(R, imageTensorDomain[idX,idY,idZ,...]), R.T) 
              
      return Tensor(imageArray=imageTensor)
  
    def getJacobianDeterminantDomain(self):
          
        # inverse Jacobian in grid space is equal to T_mov2static, so we need to invert for the Jacobian
        T_static2mov = np.linalg.inv(self._compute_transformation_matrix()[0:3,0:3])
          
        jacDetDomain = np.linalg.det(T_static2mov)*np.ones(self.mapping.domain_shape)        
        print('Statistics Jacobian determinant: ' + str(jacDetDomain.mean()) + ' +/- ' + str(jacDetDomain.std()))
        return jacDetDomain
    
    @staticmethod
    def deformTensors(tensor, T_grid2grid, output_shape=None):
        
        xx = scipy.ndimage.affine_transform(tensor[:, :, :, 0, 0], np.linalg.inv(T_grid2grid), output_shape=output_shape)
        xy = scipy.ndimage.affine_transform(tensor[:, :, :, 0, 1], np.linalg.inv(T_grid2grid), output_shape=output_shape)
        xz = scipy.ndimage.affine_transform(tensor[:, :, :, 0, 2], np.linalg.inv(T_grid2grid), output_shape=output_shape)
        yy = scipy.ndimage.affine_transform(tensor[:, :, :, 1, 1], np.linalg.inv(T_grid2grid), output_shape=output_shape)
        yz = scipy.ndimage.affine_transform(tensor[:, :, :, 1, 2], np.linalg.inv(T_grid2grid), output_shape=output_shape)
        zz = scipy.ndimage.affine_transform(tensor[:, :, :, 2, 2], np.linalg.inv(T_grid2grid), output_shape=output_shape)
        
        tensorDeformed = np.array([[xx, xy, xz], 
                                   [xy, yy, yz], 
                                   [xz, yz, zz]])
        tensorDeformed = np.transpose(tensorDeformed, [2, 3, 4, 0, 1])
        
        return tensorDeformed 
    
    @staticmethod    
    def rigid_rotation_component(Jac, Jac_T):
        return np.matmul(scipy.linalg.fractional_matrix_power(np.matmul(Jac, Jac_T), -1/2), Jac)