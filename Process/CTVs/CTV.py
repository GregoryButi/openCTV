#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:39:36 2023

@author: gregory
"""

import numpy as np
from opentps.core.data.images._roiMask import ROIMask
from opentps.core.processing.segmentation.segmentationCT import compute3DStructuralElement

class CTV(ROIMask):

  def __init__(self, imageArray=None, origin=(0, 0, 0), spacing=(1, 1, 1)):
    super().__init__(imageArray=imageArray, origin=origin, spacing=spacing)
          
  def getVolume(self):
    return self.imageArray.sum()*self.spacing.prod()    
    
  def smoothMask(self, BS):

      self.closeMask(struct=compute3DStructuralElement([2, 2, 2], spacing=self.spacing), tryGPU=True)
      self.openMask(struct=compute3DStructuralElement([2, 2, 2], spacing=self.spacing), tryGPU=True)
      self.imageArray[BS] = False
  
  def getMeshpoints(self):
    polygonMeshList = self.getROIContour().polygonMesh
      
    polygonMeshArray = np.empty((0,3), int)
    for zSlice in polygonMeshList:
        for point in np.arange(0,len(zSlice),3):
            meshpoint = np.zeros((1,3))
            meshpoint[0,0] = zSlice[point]
            meshpoint[0,1] = zSlice[point+1]
            meshpoint[0,2] = zSlice[point+2]   

            polygonMeshArray = np.append(polygonMeshArray, meshpoint, axis=0)   

    return polygonMeshArray
