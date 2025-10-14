#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:39:36 2023

@author: gregory
"""

import numpy as np
from dipy.reconst.dti import fractional_anisotropy, color_fa, mean_diffusivity
from dipy.align.reslice import reslice
from dipy.io.image import load_nifti
from scipy.ndimage import gaussian_filter

from opentps.core.data.images._image3D import Image3D

class Tensor(Image3D):

  def __init__(self, imageArray=None, origin=(0, 0, 0), spacing=(1, 1, 1)):
    super().__init__(imageArray=imageArray, origin=origin, spacing=spacing)
    self._eig_decomp_done = False
    self._evals = None
    self._evecs = None
    self._affine = None
    
  @property
  def eig_decomp_done(self):
      return self._eig_decomp_done

  @eig_decomp_done.setter
  def eig_decomp_done(self, bool):
      self._eig_decomp_done = bool  
      
  @Image3D.imageArray.setter  # Override the setter method from the parent class
  def imageArray(self, array):
      self._imageArray = array
      # Reset cached properties when imageArray changes
      self.eig_decomp_done = False  
      self._evals = None
      self._evecs = None
      
  @property
  def evals(self):
      if not self.eig_decomp_done:
          self._eig_decomp()
      return self._evals
  
  @evals.setter
  def evals(self, array):
      self._evals = array
      
  @property
  def evecs(self):
      if not self.eig_decomp_done:
          self._eig_decomp()
      return self._evecs   
  
  @evecs.setter
  def evecs(self, array):
      self._evecs = array

  @property
  def affine(self):
      return self._affine

  @affine.setter
  def affine(self, array):
      self._affine = array

  def loadTensor(self, file_path, format=None):

    # load data
    imageTensor, affine, spacing = load_nifti(file_path, return_voxsize=True)

    if format == 'ANTs':

        # ANTs convention: xx, xy, yy, xz, yz, zz
        image_xx = imageTensor[:, :, :, 0, 0]
        image_xy = imageTensor[:, :, :, 0, 1]
        image_yy = imageTensor[:, :, :, 0, 2]
        image_xz = imageTensor[:, :, :, 0, 3]
        image_yz = imageTensor[:, :, :, 0, 4]
        image_zz = imageTensor[:, :, :, 0, 5]

        imageTensor = np.array([[image_xx, image_xy, image_xz],
                          [image_xy, image_yy, image_yz],
                          [image_xz, image_yz, image_zz]])
        imageTensor = np.transpose(imageTensor, [2, 3, 4, 0, 1])

    if format == 'MRItrix3':

        # MRItrix3 convention: xx, yy, zz, xy, xz, yz
        image_xx = imageTensor[:, :, :, 0]
        image_yy = imageTensor[:, :, :, 1]
        image_zz = imageTensor[:, :, :, 2]
        image_xy = imageTensor[:, :, :, 3]
        image_xz = imageTensor[:, :, :, 4]
        image_yz = imageTensor[:, :, :, 5]

        imageTensor = np.array([[image_xx, image_xy, image_xz],
                          [image_xy, image_yy, image_yz],
                          [image_xz, image_yz, image_zz]])
        imageTensor = np.transpose(imageTensor, [2, 3, 4, 0, 1])

    if format == 'FSL':

        # FSL convention: xx xy xz yy yz zz
        image_xx = imageTensor[:, :, :, 0]
        image_xy = imageTensor[:, :, :, 1]
        image_xz = imageTensor[:, :, :, 2]
        image_yy = imageTensor[:, :, :, 3]
        image_yz = imageTensor[:, :, :, 4]
        image_zz = imageTensor[:, :, :, 5]

        imageTensor = np.array([[image_xx, image_xy, image_xz],
                          [image_xy, image_yy, image_yz],
                          [image_xz, image_yz, image_zz]])
        imageTensor = np.transpose(imageTensor, [2, 3, 4, 0, 1])

    # set attributes
    self.imageArray = imageTensor
    self.spacing = spacing
    self.origin = affine[:3, 3]
    self.affine = affine


  def _eig_decomp(self):

      print("Perform eigen-decomposition")
                  
      evals, evecs = np.linalg.eigh(self.imageArray)
      
      # Apply corrections
      corrected_evals = correctLowerBound(evals)
      corrected_evecs = correctInvertibility(evecs)

      # Sort eigenvalues
      sorted_indices = np.argsort(-corrected_evals, axis=-1)
      sorted_evals = np.take_along_axis(corrected_evals, sorted_indices, axis=-1)

      # Use advanced indexing to sort corrected_evecs
      sorted_evecs = np.transpose(np.take_along_axis(np.transpose(corrected_evecs, [0, 1, 2, 4, 3]), sorted_indices[..., None], axis=-2), [0, 1, 2, 4, 3])

      # Set attributes
      self.eig_decomp_done = True
      self.evals = sorted_evals
      self.evecs = sorted_evecs

  def get_FA_MD_RGB(self):
    # Compute scalar images with DIPY
    FA = fractional_anisotropy(self.evals)
    MD = mean_diffusivity(self.evals)
    RGB = color_fa(FA, self.evecs)
    
    return FA, MD, RGB

  def getInverse(self):
     # Invert tensor: A = Q*V*Q^-1 --> A^-1 = Q*V^-1*Q^-1
     imageArray = self.reconstruct_tensor(self.evecs, 1.0 / self.evals)
     
     return imageArray

  def getDeterminant(self):
      return np.linalg.det(self.imageArray)
         
  def reslice(self, affine, new_spacing):

      image00, _ = reslice(self.imageArray[..., 0, 0], affine, self.spacing, new_spacing)

      imageTensor = self.initializeTensorImage(image00.shape+(3, 3))
      for i in [0, 1, 2]:
          for j in [0, 1, 2]:
              imageTensor[..., i, j], _ = reslice(self.imageArray[..., i, j], affine, self.spacing, new_spacing)
      
      # update attributes        
      self.imageArray = imageTensor
      self.spacing = new_spacing
      # self.origin = ? 
      
  def smooth(self, sigma=1):
      
      imageTensor = initializeTensorImage(self.gridSize)
      for i in [0, 1, 2]:
          for j in [0, 1, 2]:
              imageTensor[..., i, j] = gaussian_filter(self.imageArray[..., i, j], sigma=sigma)
      
      # update attributes      
      self.imageArray = imageTensor        

  def reduceGrid_mask(self, mask):  
      super().reduceGrid_mask(mask)

      # update attributes 
      self.eig_decomp_done = False
      # TODO: update affine
      # self.affine =

  def isPositiveDefinite(self, mask=None):
      """
      Check positive definiteness for a field of 3x3 matrices.

      Returns
      -------
      mask : np.ndarray
          Boolean array of shape (N, M, L), True if the corresponding 3x3 matrix is positive definite.
      """

      result = np.zeros(self.gridSize[:-2], dtype=bool)

      idX, idY, idZ = np.where(mask)
      for i in idX:
          for j in idY:
              for k in idZ:
                  if mask is not None and not mask[i, j, k]:
                      continue
                  try:
                      np.linalg.cholesky(self.imageArray[i, j, k])
                      result[i, j, k] = True
                  except np.linalg.LinAlgError:
                      result[i, j, k] = False

      return result

  @staticmethod
  def reconstruct_tensor(evecs, evals):
    Lambda = Tensor.initializeTensorImage(evecs.shape)
    Lambda[..., 0, 0] = evals[..., 0]
    Lambda[..., 1, 1] = evals[..., 1]
    Lambda[..., 2, 2] = evals[..., 2]

    return np.matmul(evecs, np.matmul(Lambda, np.linalg.inv(evecs)))

  @staticmethod
  def reconstruct_tensor_mask(evecs, evals, idX, idY, idZ):
    Lambda = Tensor.initializeTensorImage(evecs.shape)
    Lambda[idX, idY, idZ, 0, 0] = evals[idX, idY, idZ, 0]
    Lambda[idX, idY, idZ, 1, 1] = evals[idX, idY, idZ, 1]
    Lambda[idX, idY, idZ, 2, 2] = evals[idX, idY, idZ, 2]

    return np.matmul(evecs, np.matmul(Lambda, np.linalg.inv(evecs)))

  @staticmethod
  def initializeTensorImage(shape):

    # initialize metric tensor as identity matrix
    tensor = np.zeros(shape)
    tensor[..., 0:3, 0:3] = np.eye(3)

    return tensor

def correctInvertibility(matrix, eps = 1e-9):

  # Check if any of the matrices are singular (not invertible)
  singular_indices = np.where(np.linalg.det(matrix) == 0)[0]

  if len(singular_indices) == 0:
      # Matrix is already invertible
      return matrix

  # Correct the singular matrices
  for index in singular_indices:
      # Add a small value to the diagonal elements to make them non-zero
      corrected_matrix = matrix[index] + np.eye(3) * eps

      # Assign the corrected matrix back to the original position
      matrix[index] = corrected_matrix

  # Reshape the corrected matrix back to the original shape
  corrected_matrix = np.reshape(matrix, matrix.shape)

  return corrected_matrix

def correctLowerBound(matrix, lower_bound = 1e-9):
  mask = matrix < lower_bound
  matrix[mask] = lower_bound
  return matrix

# def correctPositiveDefiniteness(evals, eps = 1e-9):   
#   if (evals < 0).any(): 
#       print('Correcting input tensors for positive definiteness')
#       # replace negative eigenvalues by epsilon
#       evals[evals < 0] = eps
#   return evals