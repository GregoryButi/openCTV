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
    def jacDeformable(self):
        if self.jac is None:
            self.jac = self.invert(self._computeJacobian(self.mapping.get_simplified_transform().forward))
        return self.jac

    def getTensorMetricTransformed(self, tensor, mask=[]):

        if mask == []:
            idX, idY, idZ = np.nonzero(np.ones(self.mapping.codomain_shape))
        else:
            idX, idY, idZ = np.nonzero(mask)

        # initialize
        imageTensor = np.zeros(tensor.gridSize)

        # transform tensor
        imageTensor[idX, idY, idZ, ...] = np.matmul(
            np.matmul(self.transpose(self.jacDeformable)[idX, idY, idZ, ...], tensor.imageArray[idX, idY, idZ, ...]),
            self.jacDeformable[idX, idY, idZ, ...])

        # interpolate
        imageTensor = self.deformTensor(imageTensor, self.mapping, out_shape=self.mapping.domain_shape)

        return TensorMetric(imageArray=imageTensor)

    def getTensorDiffusionTransformed(self, tensor, method='ICT', mask=[]):

        if mask == []:
            idX, idY, idZ = np.nonzero(np.ones(self.mapping.codomain_shape))
        else:
            idX, idY, idZ = np.nonzero(mask)

        imageTensor = np.zeros(tensor.gridSize)  # initialize
        if method == 'ICT':

            # transform tensor
            imageTensor[idX, idY, idZ, ...] = np.matmul(
                np.matmul(self.invert(self.jacDeformable)[idX, idY, idZ, ...], tensor.imageArray[idX, idY, idZ, ...]),
                self.transpose(self.invert(self.jacDeformable))[idX, idY, idZ, ...])

            # interpolate
            imageTensor = self.deformTensor(imageTensor, self.mapping, out_shape=self.mapping.domain_shape)

        elif method == 'PPD':

            tensorDomain = Tensor(
                imageArray=self.deformTensor(tensor.imageArray, self.mapping, out_shape=self.mapping.domain_shape))

            jacInv = self.invert(self.jacDeformable)
            for i in range(len(idX)):
                jac_voxel = jacInv[idX[i], idY[i], idZ[i], :, :]
                e1 = tensorDomain.evecs[idX[i], idY[i], idZ[i], :, 0]
                e2 = tensorDomain.evecs[idX[i], idY[i], idZ[i], :, 1]

                R = self.rotation_component_affine(jac_voxel, e1, e2)

                # reorient tensor: (R^-1)^T * tensor * R^-1
                imageTensor[idX[i], idY[i], idZ[i], :, :] = np.dot(
                    np.dot(R, tensorDomain.imageArray[idX[i], idY[i], idZ[i], :, :]), R.T)

        elif method == 'FS':

            imageTensorDomain = self.deformTensor(tensor.imageArray, self.mapping, out_shape=self.mapping.domain_shape)

            # extract rotation matrix
            R = self.rigid_rotation_component_ndarray(self.invert(self.jacDeformable),
                                                      self.transpose(self.invert(self.jacDeformable)))
            R_T = np.transpose(R, [0, 1, 2, 4, 3])

            imageTensor[idX, idY, idZ, ...] = np.matmul(
                np.matmul(R[idX, idY, idZ, ...], imageTensorDomain[idX, idY, idZ, ...]), R_T[idX, idY, idZ, ...])

        return TensorDiffusion(imageArray=imageTensor)

    def getJacobianDeterminantDomain(self):

        # compute Jacobian determinant in the domain grid
        jacDetDomain = self.compute_determinant(self._computeJacobian(self.mapping.get_simplified_transform().backward))
        print('Statistics Jacobian determinant: ' + str(jacDetDomain.mean()) + ' +/- ' + str(jacDetDomain.std()))
        return jacDetDomain

    @staticmethod
    def _computeJacobian(dvf):

        # compute Jacobian (partial derivative of vector field)

        Dx = dvf[:, :, :, 0]
        Dy = dvf[:, :, :, 1]
        Dz = dvf[:, :, :, 2]

        Dx_dx, Dx_dy, Dx_dz = np.gradient(Dx, edge_order=2)
        Dy_dx, Dy_dy, Dy_dz = np.gradient(Dy, edge_order=2)
        Dz_dx, Dz_dy, Dz_dz = np.gradient(Dz, edge_order=2)

        Jac_inv = np.array([[1 + Dx_dx, Dx_dy, Dx_dz],
                            [Dy_dx, 1 + Dy_dy, Dy_dz],
                            [Dz_dx, Dz_dy, 1 + Dz_dz]
                            ])
        Jac_inv = np.transpose(Jac_inv, [2, 3, 4, 0, 1])

        return Jac_inv

    @staticmethod
    def rigid_rotation_component_ndarray(A, A_T):
        return np.matmul(fractional_power(np.matmul(A, A_T), -1 / 2), A)

    @staticmethod
    def deformTensor(tensor, mapping, out_shape=None):

        xx = mapping.transform(tensor[:, :, :, 0, 0], out_shape=out_shape)
        xy = mapping.transform(tensor[:, :, :, 0, 1], out_shape=out_shape)
        xz = mapping.transform(tensor[:, :, :, 0, 2], out_shape=out_shape)
        yy = mapping.transform(tensor[:, :, :, 1, 1], out_shape=out_shape)
        yz = mapping.transform(tensor[:, :, :, 1, 2], out_shape=out_shape)
        zz = mapping.transform(tensor[:, :, :, 2, 2], out_shape=out_shape)

        tensorDeformed = np.array([[xx, xy, xz],
                                   [xy, yy, yz],
                                   [xz, yz, zz]])
        tensorDeformed = np.transpose(tensorDeformed, [2, 3, 4, 0, 1])

        return tensorDeformed


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
