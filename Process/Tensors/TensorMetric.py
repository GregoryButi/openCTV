#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:50:48 2024

@author: gregory
"""

import numpy as np

from Process.Tensors import Tensor

class TensorMetric(Tensor):
    def __init__(self, imageArray=None, origin=(0, 0, 0), spacing=(1, 1, 1)):
      super().__init__(imageArray=imageArray, origin=origin, spacing=spacing)
      
    def estimateTexture(self, image3D, domain, type='structure_tensor', gradientSigmas = 1, tensorSigmas = 5):
        
        import diplib as dip
        
        if type == 'vesselness':
            I = np.asarray(dip.Hessian(dip.Image(image3D.imageArray), sigmas = gradientSigmas)) #[sigma, sigma, sigma])
        elif type == 'structure_tensor':
            I = np.asarray(dip.StructureTensor(dip.Image(image3D.imageArray), gradientSigmas=gradientSigmas, tensorSigmas=tensorSigmas))
            
        Izz, Iyy, Ixx, Iyz, Ixz, Ixy = I[:, :, :, 0], I[:, :, :, 1], I[:, :, :, 2], I[:, :, :, 3],  I[:, :, :, 4], I[:, :, :, 5]
               
        imageTensor = self.initializeTensorImage(np.append(image3D.gridSize,(3,3)))
        imageTensor[domain] = np.array([[Ixx, Ixy, Ixz],
                                        [Ixy, Iyy, Iyz], 
                                        [Ixz, Iyz, Izz]])[domain]
        imageTensor = np.transpose(imageTensor,[2,3,4,0,1])
                    
        # set attributes
        self.imageArray = imageTensor
        self.spacing = image3D.spacing
        self.origin = image3D.origin
    
    def getMetricTensorFMM(self, domain, dict):

        idX, idY, idZ = np.nonzero(domain)
        
        MT = np.zeros(self.gridSize)
        MT[..., 0:3, 0:3] = np.eye(3)
        if dict['model'] == 'Nonuniform':
            
            MT[domain] = np.eye(3) * dict['resistance']

        elif dict['model'] == 'Anisotropic' and dict['model-DTI'] == 'Clatz':
            
            # compute normalization such that determinant of tensor is 1
            N = (self.evals[idX, idY, idZ, 0]*self.evals[idX, idY, idZ, 1]*self.evals[idX, idY, idZ, 2]) ** (-1/3)
            
            # perform tensor sharpening
            evals = dict['resistance'] * sharpen(np.multiply(N[..., np.newaxis], self.evals[idX, idY, idZ]), dict['anisotropy'])
            
            MT = self.reconstruct_tensor_mask(self.evecs, evals, idX, idY, idZ)  
    
        elif dict['model'] == 'Anisotropic' and dict['model-DTI'] == 'Rekik':
    
            Lambda = np.ones(self.evals.shape)  
            Lambda[idX, idY, idZ, 2] = dict['resistance']
            evals = Lambda[idX, idY, idZ]        
            
            MT = self.reconstruct_tensor_mask(self.evecs, evals, idX, idY, idZ)  
              
        return MT    
    
    def setBarrier(self, mask):
        self.imageArray[mask] = np.eye(3) * 1e+14
        
def sharpen(evals, power):

  # product of eigenvalues
  det = evals.prod(axis=-1)[..., np.newaxis]
  # normalization to preserve original determinant
  N = det ** ((1-power)/3)

  # redefine eigenvalues
  evals = np.multiply(N, evals ** power)

  return evals
