#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:36:03 2024

@author: gregory
"""

import numpy as np

from Process.Tensors import Tensor
from Process.Tensors import TensorMetric
from Process.Tensors import TensorDiffusion
from Process.Transforms import TransformTensor

class TransformTensorDeformable(TransformTensor):
    def __init__(self, mapping):
        super().__init__(mapping=mapping)

    @property
    def jacobianDomain(self):
        if self.jacDomain is None:
            self.jacDomain = self._computeJacobian(self.mapping.get_simplified_transform().forward, direction='forward')
        return self.jacDomain

    @property
    def jacobianCodomain(self):
        if self.jacCodomain is None:
            self.jacCodomain = self._computeJacobian(self.mapping.get_simplified_transform().backward, direction='backward')
        return self.jacCodomain

    def getTensorMetricTransformed(self, tensor, mask=None):

        imageTensor = np.zeros(tuple(self.mapping.domain_shape) + (3, 3))  # initialize
        idX, idY, idZ = self._getIndices_domain(mask)

        # transform tensor
        imageTensor[idX, idY, idZ, ...] = np.matmul(np.matmul(self.transpose(self.jacobianDomain)[idX, idY, idZ, ...], tensor.imageArray[idX, idY, idZ, ...]), self.jacobianDomain[idX, idY, idZ, ...])

        # interpolate
        imageTensor = self._deformTensor(imageTensor)

        return TensorMetric(imageArray=imageTensor)

    def getTensorDiffusionTransformed(self, tensor, method='ICT', mask=None):

        if method == 'ICT':  # invariance under coordinate transform

            imageTensor = np.zeros(tuple(self.mapping.domain_shape) + (3, 3))  # initialize
            idX, idY, idZ = self._getIndices_domain(mask)

            # transform tensor
            imageTensor[idX, idY, idZ, ...] = np.matmul(np.matmul(self.invert(self.jacobianDomain)[idX, idY, idZ, ...], tensor.imageArray[idX, idY, idZ, ...]), self.transpose(self.invert(self.jacobianDomain))[idX, idY, idZ, ...])

            # interpolate
            imageTensor = self._deformTensor(imageTensor)

        elif method == 'PPD':

            imageTensor = np.zeros(tuple(self.mapping.codomain_shape) + (3, 3))
            idX, idY, idZ = self._getIndices_codomain(mask)

            tensorCodomain = Tensor(imageArray=self._deformTensor(tensor.imageArray))

            jacInv = self.invert(self.jacobianCodomain)
            for i in range(len(idX)):
                jacInv_voxel = jacInv[idX[i], idY[i], idZ[i], :, :]
                e1 = tensorCodomain.evecs[idX[i], idY[i], idZ[i], :, 0]
                e2 = tensorCodomain.evecs[idX[i], idY[i], idZ[i], :, 1]

                R = self.rotation_component_affine(jacInv_voxel, e1, e2)

                # reorient tensor: (R^-1)^T * tensor * R^-1
                imageTensor[idX[i], idY[i], idZ[i], :, :] = np.dot(np.dot(R, tensorCodomain.imageArray[idX[i], idY[i], idZ[i], :, :]), R.T)

        elif method == 'FS':

            imageTensor = np.zeros(tuple(self.mapping.codomain_shape) + (3, 3))
            idX, idY, idZ = self._getIndices_codomain(mask)

            imageTensorCodomain = self._deformTensor(tensor.imageArray)

            # extract rotation matrix
            R = self._rigid_rotation_component_ndarray(self.invert(self.jacobianCodomain), self.transpose(self.invert(self.jacobianCodomain)))
            R_T = np.transpose(R, [0, 1, 2, 4, 3])

            imageTensor[idX, idY, idZ, ...] = np.matmul(np.matmul(R[idX, idY, idZ, ...], imageTensorCodomain[idX, idY, idZ, ...]), R_T[idX, idY, idZ, ...])

        return TensorDiffusion(imageArray=imageTensor)

    def getJacobianDeterminantCodomain(self):

        # compute Jacobian determinant in the codomain grid
        jacDetCodomain = self.compute_determinant(self.jacobianCodomain)
        print('Statistics Jacobian determinant: ' + str(jacDetCodomain.mean()) + ' +/- ' + str(jacDetCodomain.std()))
        return jacDetCodomain

    def _computeJacobian(self, dvf, direction):

        # compute Jacobian (partial derivative of vector field)

        Dx = dvf[:, :, :, 0]
        Dy = dvf[:, :, :, 1]
        Dz = dvf[:, :, :, 2]

        Dx_dx, Dx_dy, Dx_dz = np.gradient(Dx, edge_order=2)
        Dy_dx, Dy_dy, Dy_dz = np.gradient(Dy, edge_order=2)
        Dz_dx, Dz_dy, Dz_dz = np.gradient(Dz, edge_order=2)

        derivatives = np.array([[1 + Dx_dx, Dx_dy, Dx_dz],
                            [Dy_dx, 1 + Dy_dy, Dy_dz],
                            [Dz_dx, Dz_dy, 1 + Dz_dz]
                            ])
        derivatives = np.transpose(derivatives, [2, 3, 4, 0, 1])

        if direction == 'forward':
            return self.invert(derivatives)
        elif direction == 'backward':
            return derivatives

    @staticmethod
    def _rigid_rotation_component_ndarray(A, A_T):
        return np.matmul(fractional_power(np.matmul(A, A_T), -1 / 2), A)

def fractional_power(A, alpha):
    """
    Calculates the fractional matrix power of a 5D array A,
    where the fractional matrix power is applied to the last two dimensions.

    Args:
    A: array-like, shape (N, M, L, 3, 3)
        Input array.
    alpha: float
        Fractional power to compute.

    Returns:
    B: ndarray, shape (N, M, L, 3, 3)
        The result of raising A to the power of alpha.
    """
    # Reshape A into a 4D array with shape (N*M*L, 3, 3)
    n, m, l, _, _ = A.shape
    A_2d = np.reshape(A, (n * m * l, 3, 3))

    # Calculate eigendecomposition of each 3x3 matrix in A_2d
    eigvals = np.zeros((n * m * l, 3))
    eigvecs = np.zeros((n * m * l, 3, 3))
    for i in range(n * m * l):
        eigvals[i], eigvecs[i] = np.linalg.eig(A_2d[i])

    # Calculate fractional power of each 3x3 matrix in A_2d
    eigvals_power = np.power(eigvals, alpha)
    B_2d = np.zeros((n * m * l, 3, 3))
    for i in range(n * m * l):
        B_2d[i] = eigvecs[i] @ np.diag(eigvals_power[i]) @ np.linalg.inv(eigvecs[i])

    # Reshape B_2d back into a 5D array with the same shape as A
    B = np.reshape(B_2d, (n, m, l, 3, 3))
    return B
