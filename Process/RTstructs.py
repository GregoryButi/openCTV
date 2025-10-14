#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:39:36 2023

@author: gregory
"""

import os
import numpy as np
import scipy
from dipy.io.image import load_nifti
from skimage.measure import label
from scipy.ndimage import label, binary_fill_holes

from opentps.core.data.images._roiMask import ROIMask
from opentps.core.processing.segmentation.segmentationCT import compute3DStructuralElement

class Struct(object):

  def __init__(self):
    self._masks = []
    self._roi_types = []
    
  @property
  def masks(self):
      return self._masks

  @masks.setter
  def masks(self, list):
      self._masks = list

  @property
  def roi_types(self):
      return self._roi_types

  @roi_types.setter
  def roi_types(self, list):
      self._roi_types = list

  def loadContours_folder(self, folder_path, file_names, contour_names=None, contour_types=None):
      file_list = os.listdir(folder_path)

      for file_name in file_list:
          full_path = os.path.join(folder_path, file_name)

          # Optional: skip non-NIfTI files
          if not (file_name.endswith(".nii") or file_name.endswith(".nii.gz")):
              continue

          for i, target in enumerate(file_names):
              # Get base name without extension(s)
              base_name = file_name
              if base_name.endswith(".nii.gz"):
                  base_name = base_name[:-7]
              elif base_name.endswith(".nii"):
                  base_name = base_name[:-4]

              if target == base_name:

                  image_array, grid2world, spacing = load_nifti(full_path, return_voxsize=True)

                  if image_array.dtype != bool:
                      image_array = image_array >= 0.5

                  name = contour_names[i] if contour_names is not None else target
                  roi_type = contour_types[i] if contour_types is not None else None

                  mask = ROIMask(imageArray=image_array, name=name, spacing=spacing, origin=grid2world[0:3, 3], grid2world=grid2world)

                  self.masks.append(mask)
                  self.roi_types.append(roi_type)

                  break  # Stop at first match to prevent multiple matches

  def reduceGrid_mask(self, cropmask):
      for mask in self.masks:
          mask.reduceGrid_mask(cropmask)

  def setMask(self, name, image_array, roi_type=None, spacing=(1, 1, 1), origin=(0, 0, 0), grid2world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]):
      # Ensure boolean image
      if not isinstance(image_array, bool):
          image_array = image_array >= 0.5

      for i, mask in enumerate(self.masks):
          if mask.name == name:
              mask.imageArray = image_array
              return

      # If not found, create new mask
      new_mask = ROIMask(imageArray=image_array, name=name, origin=origin, spacing=spacing, grid2world=grid2world)
      self.masks.append(new_mask)
      self.roi_types.append(roi_type)


  def setTypeByName(self, name: str, roi_type_new: str) -> bool:
      for i, mask in enumerate(self.masks):
          if mask.name == name:
              self.roi_types[i] = roi_type_new
              return True
      return False

  def getMaskByName(self, name: str):
      for mask in self.masks:
          if mask.name == name:
              return mask
      return None

  def getMaskByType(self, roi_type_target: str):
      return [mask for i, mask in enumerate(self.masks) if self.roi_types[i] == roi_type_target]

  def getTypeByName(self, name: str):
      for i, mask in enumerate(self.masks):
          if mask.name == name:
              return self.roi_types[i]
      return None
  
  def transformMasksDIPY(self, mapping, direction='forward'):
      for mask in self.masks:
          if direction == 'forward': mask.imageArray = mapping.transform(mask.imageArray) >= 0.5
          if direction == 'backward': mask.imageArray = mapping.transform_inverse(mask.imageArray) >= 0.5

  def transformMaskAffineSCIPY(self, name: str, output_grid2world, output_shape):
      for i, mask in enumerate(self.masks):
          if mask.name == name:

              # compute transformation matrix
              T_mov2static = np.linalg.inv(output_grid2world) @ mask.grid2world

              mask.imageArray = scipy.ndimage.affine_transform(mask.imageArray.astype(float), np.linalg.inv(T_mov2static), output_shape=output_shape) >= 0.5
              mask.origin = output_grid2world[0:3, 3]
              mask.spacing = (output_grid2world[0, 0], output_grid2world[1, 1], output_grid2world[2, 2])
              mask.grid2world = output_grid2world
  
  def createSphere(self, name, X_world, Y_world, Z_world, COM_world, radius, spacing, roi_type=None):

      Sphere = np.where(((X_world - COM_world[0]) / radius[0]) ** 2 + 
                        ((Y_world - COM_world[1]) / radius[1]) ** 2 + 
                        ((Z_world - COM_world[2]) / radius[2]) ** 2 <= 1, 1, 0).astype(bool)       
                                         
      self.setMask(name, Sphere, spacing=spacing, roi_type=roi_type)
                                               
  def smoothMasks(self, names, size = 2):
      for name in names:
          mask = self.getMaskByName(name)

          mask.closeMask(struct=compute3DStructuralElement([size, size, size], spacing=mask.spacing), tryGPU=True)
          mask.openMask(struct=compute3DStructuralElement([size, size, size], spacing=mask.spacing), tryGPU=True)

  def dilateMasks(self, names, radius):
      for name in names:
          mask = self.getMaskByName(name)
          mask.dilateMask(radius=radius)

  def removeSmallIslets(self, names, volume_threshold):
      for name in names:
          mask = self.getMaskByName(name)
          data = mask.imageArray

          labeled_array, num_features = label(data)
          component_sizes = np.bincount(labeled_array.ravel())

          # Remove background (label 0)
          too_small = component_sizes < volume_threshold
          too_small[0] = False  # Keep background label as-is

          # Mask of voxels to remove
          remove_mask = too_small[labeled_array]
          data[remove_mask] = 0

          # Update the mask
          mask.imageArray = data

  def fillHoles(self, names):
      for name in names:
          mask = self.getMaskByName(name)
          mask.imageArray = binary_fill_holes(mask.imageArray).astype(mask.imageArray.dtype)

  def getLargestCC(self, name, num_components=1):
      """
      Returns a binary mask with the 'num_components_to_keep' largest connected components.

      Parameters:
      - name (str): The name of the mask to process.
      - num_components_to_keep (int): The number of largest connected components to keep.

      Returns:
      - np.ndarray: Binary mask with the specified number of largest connected components.
      """
      mask = self.getMaskByName(name).imageArray

      # Label connected components
      labels, num_labels = label(mask)

      if num_labels == 0:
          return np.zeros_like(mask, dtype=bool)

      # Count the size of each component, ignoring background (label 0)
      counts = np.bincount(labels.ravel())
      counts[0] = 0  # ignore background

      # Get indices of the top 'num_components' largest components
      largest_labels = np.argsort(counts)[-num_components:]

      # Create binary mask with only the selected largest components
      largest_cc = np.isin(labels, largest_labels)

      return largest_cc

  def getCountCC(self, name):
      """
      Counts the number of connected components in a 3D binary mask.

      Returns:
      int: Number of connected components.
      """

      mask = self.getMaskByName(name).imageArray

      # Label connected components
      labeled_mask, num_components = label(mask)
      return num_components

  def getBoundingBox(self, name, margin=5):
      
      mask = self.getMaskByName(name)

      idX, idY, idZ = np.nonzero(mask.imageArray>0)
      
      BB = np.zeros(mask.gridSize, dtype=bool)
      BB[max(0, min(idX)-margin):min(max(idX)+margin, mask.gridSize[0] - 1),
         max(0, min(idY)-margin):min(max(idY)+margin, mask.gridSize[1] - 1),
         max(0, min(idZ)-margin):min(max(idZ)+margin, mask.gridSize[2] - 1)] = True
      
      return BB