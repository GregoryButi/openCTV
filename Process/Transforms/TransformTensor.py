#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 08:54:32 2023

@author: gregory
"""

import numpy as np
#import tensorflow as tf

class TransformTensor(object):
    
    def __init__(self, mapping=None):
        self._mapping = mapping
        self._jac = None
    
    @property
    def mapping(self):
        return self._mapping
    
    @mapping.setter
    def mapping(self, mapping_obj):
        self._mapping = mapping_obj 
        # Reset cached properties when mapping changes
        self._jac = None
    
    @property
    def jac(self):
        return self._jac
    
    @jac.setter
    def jac(self, array):
        self._jac = array 
    
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