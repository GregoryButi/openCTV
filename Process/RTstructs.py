#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:39:36 2023

@author: gregory
"""

import os
import numpy as np
from dipy.io.image import load_nifti
from dipy.align.reslice import reslice
from skimage.measure import label

from opentps.core.data.images._roiMask import ROIMask
from opentps.core.processing.segmentation.segmentationCT import compute3DStructuralElement

class Struct(object):

  def __init__(self):
    self._masks = []    
    
  @property
  def masks(self):
      return self._masks

  @masks.setter
  def masks(self, list):
      self._masks = list  
  
  def loadContours_folder(self, folder_path, contour_names):
      
      file_list = os.listdir(folder_path) 
      
      for file_name in file_list:
          
          file_path = os.path.join(folder_path, file_name)
          
          # search for the string in each element of the list
          for i, element in enumerate(contour_names):
              if element in file_name:
                 
                 imageArray, grid2world, spacing = load_nifti(file_path, return_voxsize=True)
                 
                 if not imageArray.dtype == bool:
                     imageArray = imageArray>=0.5
                 
                 mask = ROIMask(imageArray=imageArray, name=element, spacing=spacing)
                 
                 self.masks.append(mask)            
  
  def reduceGrid_mask(self, cropmask):
      
      for mask in self.masks:
          mask.reduceGrid_mask(cropmask)
          
  def setMask(self, name, imageArray, spacing):
      
      mask_found = False
      for mask in self.masks:
          if mask.name == name:
              mask.imageArray = imageArray
              mask.spacing = spacing
              mask_found = True
      
      if ~mask_found:   
          mask = ROIMask(imageArray=imageArray, name=name, spacing=spacing)
          self.masks.append(mask)
          
  
  def getMaskByName(self, name:str):
      
      for mask in self.masks:
          if mask.name == name:
              return mask
      print(f'No contour with name {name} found in the list of contours')
  
  def transformMasks(self, mapping, direction='forward'):
      
      for mask in self.masks:
          if direction == 'forward': mask.imageArray = mapping.transform(mask.imageArray) >= 0.5
          if direction == 'backward': mask.imageArray = mapping.transform_inverse(mask.imageArray) >= 0.5
  
  def createSphere(self, name, X_world, Y_world, Z_world, COM_world, radius, spacing):

      Sphere = np.where(((X_world - COM_world[0]) / radius[0]) ** 2 + 
                        ((Y_world - COM_world[1]) / radius[1]) ** 2 + 
                        ((Z_world - COM_world[2]) / radius[2]) ** 2 <= 1, 1, 0).astype(bool)       
                                         
      self.setMask(name, Sphere, spacing)
                                               
  def smoothMasks(self, names):
      
      for mask in self.masks:
          for name in names:
              if mask.name == name:
                  mask.closeMask(struct=compute3DStructuralElement([2, 2, 2], spacing=mask.spacing), tryGPU=True)
                  mask.openMask(struct=compute3DStructuralElement([2, 2, 2], spacing=mask.spacing), tryGPU=True)
                  
  def reslice(self, affine, new_spacing):      

      for mask in self.masks:
          image, _ = reslice(mask.imageArray.astype(float), affine, mask.spacing, new_spacing)
          
          # update attributes 
          mask.imageArray = image >= 0.5
          mask.spacing = new_spacing
          # mask.origin = ? 
          
  def getLargestCC(self, name):   
      
      for mask in self.masks:
          if mask.name == name:
              segmentation = mask.imageArray
      
      labels = label(segmentation)
      largest_cc = labels == np.argmax(np.bincount(labels[segmentation]))
      return largest_cc 

  def getBoundingBox(self, name, margin = 5):   
      
      for mask in self.masks:
          if mask.name == name:
              MOI = mask
      
      idX, idY, idZ = np.nonzero(MOI.imageArray>0)
      
      BB = np.zeros(MOI.gridSize, dtype=bool)
      BB[max(0, min(idX)-margin) : min(max(idX)+margin, MOI.gridSize[0] - 1), 
         max(0, min(idY)-margin) : min(max(idY)+margin, MOI.gridSize[1] - 1), 
         max(0, min(idZ)-margin) : min(max(idZ)+margin, MOI.gridSize[2] - 1)] = True    
      
      return BB               