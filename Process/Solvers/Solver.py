#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:50:51 2024

@author: gregory
"""

class Solver(object):

  def __init__(self, source=None, barriers=None, tensor=None, domain=None):
          
    self.domain = domain
    self.source = source
    self.tensor = tensor    
    self.barriers = barriers
      
  @property
  def source(self):
      return self._source

  @source.setter
  def source(self, object):
      self._source = object
            
      if self._source is not None and self._domain is not None:
          self._source.reduceGrid_mask(self._domain)
      
  @property
  def barriers(self):
      return self._barriers
  
  @barriers.setter
  def barriers(self, object):
      self._barriers = object 
            
      if self._barriers is not None and self.domain is not None:
          self._barriers.reduceGrid_mask(self.domain)
      
      if self.tensor is not None and self._barriers is not None:
            self._tensor.setBarrier(self._barriers.imageArray)   
      
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
