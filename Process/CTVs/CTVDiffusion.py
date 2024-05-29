#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:39:36 2023

@author: gregory
"""

from Process.CTVs import CTV

class CTVDiffusion(CTV):

  def __init__(self, imageArray=None, origin=(0, 0, 0), spacing=(1, 1, 1)):
    super().__init__(imageArray=imageArray, origin=origin, spacing=spacing)
    
    self._density3D = None
      
  @property
  def density3D(self):
      return self._distance3D

  @density3D.setter
  def density3D(self, array):
      self._density3D = array  