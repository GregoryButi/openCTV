#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:50:48 2024

@author: gregory
"""

import numpy as np

from Process.Tensors import Tensor, TensorDiffusion

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
        imageTensor = np.transpose(imageTensor, [2, 3, 4, 0, 1])
                    
        # set attributes
        self.imageArray = imageTensor
        self.spacing = image3D.spacing
        self.origin = image3D.origin

    def getMetricTensorFMM(self, structureDict, modelDict):
        masks = structureDict['masks']
        resistances = structureDict['resistances']
        model_type = modelDict['model']
        model_dti = modelDict.get('model-DTI', None)

        # Union of all mask domains
        domain = np.logical_or.reduce([m.imageArray for m in masks])
        idX_all, idY_all, idZ_all = np.nonzero(domain)

        if model_type == 'Nonuniform':
            MT = self.initializeTensorImage(self.gridSize)
            for mask, resistance in zip(masks, resistances):
                MT[mask.imageArray] = np.eye(3) * resistance

        elif model_type == 'Anisotropic' and model_dti == 'Clatz':
            evals = np.ones_like(self.evals)
            for mask, resistance in zip(masks, resistances):
                idX, idY, idZ = np.nonzero(mask.imageArray)
                local_evals = self.evals[idX, idY, idZ]

                # Normalize to unit determinant
                N = (local_evals[..., 0] * local_evals[..., 1] * local_evals[..., 2]) ** (-1 / 3)
                sharpened = sharpen(N[..., np.newaxis] * local_evals, modelDict['anisotropy'])

                evals[idX, idY, idZ] = resistance * sharpened

            MT = self.reconstruct_tensor_mask(self.evecs, evals, idX_all, idY_all, idZ_all)

        elif model_type == 'Anisotropic' and model_dti == 'Rekik':
            evals = np.ones_like(self.evals)
            for mask, resistance in zip(masks, resistances):
                idX, idY, idZ = np.nonzero(mask.imageArray)
                evals[idX, idY, idZ, 2] = resistance

            MT = self.reconstruct_tensor_mask(self.evecs, evals, idX_all, idY_all, idZ_all)

        return MT

        # elif modelDict['model'] == 'Anisotropic' and modelDict['model-DTI'] == 'Buti':
        #
        #     Lambda = np.ones(self.evals.shape)
        #     Lambda[idX, idY, idZ, 2] = dict['resistance']  # set smallest eigenvalue within mask
        #
        #     # compute normalization such that determinant of tensor is 1
        #     N = (Lambda[..., 0] * Lambda[..., 1] * Lambda[..., 2]) ** (-1 / 3)
        #
        #     # normalize entire image
        #     evals = np.multiply(N[..., np.newaxis], Lambda)
        #
        #     MT = self.reconstruct_tensor(self.evecs, evals)
        #
        # elif modelDict['model'] == 'Anisotropic' and modelDict['model-DTI'] == 'Greg':
        #
        #     Lambda = np.ones(self.evals.shape)
        #     Lambda[idX, idY, idZ, 2] = 1  # set smallest eigenvalue
        #     Lambda[idX, idY, idZ, 0] = modelDict['resistance']  # set highest eigenvalue
        #     Lambda[idX, idY, idZ, 1] = modelDict['resistance']  # set highest eigenvalue
        #     evals = Lambda[idX, idY, idZ]
        #
        #     MT = self.reconstruct_tensor_mask(self.evecs, evals, idX, idY, idZ)

    def setBarrier(self, mask):
        self.imageArray[mask] = np.eye(3) * 1e+14

    def getDiffusionTensor(self):
        dt = TensorDiffusion(imageArray=self.getInverse(), origin=self.origin, spacing=self.spacing)
        return dt
        
def sharpen(evals, power):

  # product of eigenvalues
  det = evals.prod(axis=-1)[..., np.newaxis]
  # normalization to preserve original determinant
  N = det ** ((1-power)/3)

  # redefine eigenvalues
  evals = np.multiply(N, evals ** power)

  return evals
