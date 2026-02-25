#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:50:51 2024

@author: gregory
"""

import numpy as np

class Solver(object):

  def __init__(self, source=None, boundary=None, tensor=None, domain=None):
          
    self.domain = domain
    self.source = source
    self.tensor = tensor
    self.boundary = boundary

  @property
  def source(self):
      return self._source

  @source.setter
  def source(self, object):
      self._source = object

      if self._source is not None and self._domain is not None:
          self._source.reduceGrid_mask(self._domain)
      
  @property
  def boundary(self):
      return self._boundary
  
  @boundary.setter
  def boundary(self, object):
      self._boundary = object

      if self._boundary is not None and self.domain is not None:
          self._boundary.reduceGrid_mask(self.domain)

  @property
  def tensor(self):
      return self._tensor

  @tensor.setter
  def tensor(self, obj):
      self._tensor = obj
      
      if self._tensor is not None and self._domain is not None:
          self._tensor.reduceGrid_mask(self._domain)
      
  @property
  def domain(self):
      return self._domain

  @domain.setter
  def domain(self, object):
      self._domain = object

  def getArray_fullGrid(self, array_reduced):

      array_full = np.full(self.domain.shape, np.inf)
      array_full.ravel()[self.domain.ravel()] = array_reduced.ravel()

      return array_full