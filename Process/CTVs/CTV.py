#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:39:36 2023

@author: gregory
"""

from opentps.core.data.images._roiMask import ROIMask
from opentps.core.processing.segmentation.segmentationCT import compute3DStructuralElement

class CTV(ROIMask):

  def __init__(self, rts=None, tensor=None, imageArray=None, origin=(0, 0, 0), spacing=(1, 1, 1)):
    super().__init__(imageArray=imageArray, origin=origin, spacing=spacing)

    self._gtv = None
    self._barriers = None
    self._preferred = {'masks': [], 'resistances': []}
    self._tensor = tensor

    def _ensure_list(x):
      if x is None:
        return []
      return x if isinstance(x, list) else [x]

    if rts is not None:
      self._gtv = rts.getMaskByType('GTV')[0]
      self._barriers = rts.getMaskByType('Barrier')[0]
      self._preferred['masks'] = rts.getMaskByType('Barrier_soft')
      self._preferred['resistances'] = [1. for _ in self._preferred['masks']]

  @property
  def gtv(self):
    return self._gtv

  @gtv.setter
  def gtv(self, object):
    self._gtv = object

  @property
  def barriers(self):
    return self._barriers

  @barriers.setter
  def barriers(self, object):
    self._barriers = object

  @property
  def preferred(self):
    return self._preferred

  @preferred.setter
  def preferred(self, value):
    masks = value['masks']
    resistances = value['resistances']

    self._preferred = {'masks': masks, 'resistances': resistances}

  @property
  def tensor(self):
    return self._tensor

  @tensor.setter
  def tensor(self, object):
    self._tensor = object
    
  def smoothMask(self, exclude_barrier=False, size=2):

      self.closeMask(struct=compute3DStructuralElement([size, size, size], spacing=self.spacing), tryGPU=True)
      self.openMask(struct=compute3DStructuralElement([size, size, size], spacing=self.spacing), tryGPU=True)
      if exclude_barrier and self.barrier is not None:
        self.imageArray[self.barrier.imageArray] = False