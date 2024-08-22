#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 08:54:32 2023

@author: gregory
"""

import numpy as np

class TransformTensor(object):
    
    def __init__(self, mapping=None):
        self._mapping = mapping
        self._jacDomain = None
        self._jacCodomain = None
    
    @property
    def mapping(self):
        return self._mapping
    
    @mapping.setter
    def mapping(self, mapping_obj):
        self._mapping = mapping_obj 
        # Reset cached properties when mapping changes
        self._jacDomain = None
        self._jacCodomain = None
    
    @property
    def jacDomain(self):
        return self._jacDomain
    
    @jacDomain.setter
    def jacDomain(self, array):
        self._jacDomain = array

    @property
    def jacCodomain(self):
        return self._jacCodomain

    @jacCodomain.setter
    def jacCodomain(self, array):
        self._jacCodomain = array

    def _deformTensor(self, tensor):
        xx = self.mapping.transform(tensor[:, :, :, 0, 0])
        xy = self.mapping.transform(tensor[:, :, :, 0, 1])
        xz = self.mapping.transform(tensor[:, :, :, 0, 2])
        yy = self.mapping.transform(tensor[:, :, :, 1, 1])
        yz = self.mapping.transform(tensor[:, :, :, 1, 2])
        zz = self.mapping.transform(tensor[:, :, :, 2, 2])

        tensorDeformed = np.transpose(np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]), [2, 3, 4, 0, 1])

        return tensorDeformed

    def _getIndices_domain(self, mask):
        if mask is None:
            idX, idY, idZ = [idx.ravel() for idx in np.indices(self.mapping.domain_shape)]
            return idX, idY, idZ
        else:
            idX, idY, idZ = np.nonzero(mask)
            return idX, idY, idZ

    def _getIndices_codomain(self, mask):
        if mask is None:
            idX, idY, idZ = [idx.ravel() for idx in np.indices(self.mapping.codomain_shape)]
            return idX, idY, idZ
        else:
            mask_transformed = self.mapping.transform(mask)
            idX, idY, idZ = np.nonzero(mask_transformed)
            return idX, idY, idZ

    @staticmethod
    def transpose(array):
        return np.transpose(array, [0, 1, 2, 4, 3])
    
    @staticmethod
    def invert(array):

        # Reshape the array to (..., 3, 3)
        reshaped_array = array.reshape(-1, 3, 3)

        # Compute the inverse of each 3x3 matrix
        inverted_reshaped_array = np.linalg.inv(reshaped_array)

        # Reshape the result back to (N, M, L, 3, 3)
        inverted_array = inverted_reshaped_array.reshape(array.shape[0], array.shape[1], array.shape[2], 3, 3)

        return inverted_array

    @staticmethod
    def compute_determinant(array):

        # Reshape the array to (N, M, L, 3, 3)
        reshaped_array = array.reshape(-1, 3, 3)

        # Compute the determinant of each 3x3 matrix
        determinants_reshaped_array = np.linalg.det(reshaped_array)

        # Reshape the result to (N, M, L)
        determinants = determinants_reshaped_array.reshape(array.shape[0], array.shape[1], array.shape[2])

        return determinants
    
    @staticmethod
    def rotation_component_affine(Jac, e1, e2):
        
        # follow https://doi.org/10.1109/42.963816 
        
        tmp1 = np.dot(Jac, e1)
        tmp2 = np.dot(Jac, e2)
        
        n1 = tmp1/np.linalg.norm(tmp1)
        n2 = tmp2/np.linalg.norm(tmp2)
        
        R1 = rotation_a_onto_b(e1,n1)
        
        Pn2 = n2 - np.dot(np.dot(n2,n1), n1)
        
        R2 = np.dot(np.dot(R1,e2),Pn2)
                    
        return np.dot(R1,R2)
    
def rotation_a_onto_b(a,b):
    
    # follow https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d to get the rotation matrix between e1 and n1
    v = np.cross(a, b)
    #s = np.linalg.norm(v)
        
    c = np.dot(a, b)
    
    v_cp = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]
                     ])

    return np.identity(3) + v_cp + np.linalg.matrix_power(v_cp,2)/(1+c) #*(1-c)/(s**2)    