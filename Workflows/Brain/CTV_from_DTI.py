#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:12:05 2024

@author: gregory
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass as com
from dipy.viz import regtools
from dipy.io.image import load_nifti
import copy

from opentps.core.data.images._image3D import Image3D
from Process.Tensors import TensorDiffusion
from Process.CTVs import CTVDiffusion
from Process import Struct
from Process.ImageRegistrationDIPY import ImageRegistrationRigid
from Process.Transforms import TransformTensorAffine

# input

path_MRI = '../../Input/Kim/Therapy-scan/MRI_CT/T1_norm.nii.gz'
path_RTstructs = '../../Input/Kim/Therapy-scan/Structures'
path_tensor = '../../Input/Kim/Therapy-scan/fMRI/Tensor.nii.gz'

# load data

MRI, grid2world, voxel_size = load_nifti(path_MRI, return_voxsize=True)

tensor = TensorDiffusion()
tensor.loadTensor(path_tensor, format='MRItrix3')

# load structures

RTs = Struct()
RTs.loadContours_folder(path_RTstructs, ['Brain', 'External'], contour_types=['Barrier_soft', None])

# define barrier structures
RTs.setMask('BS', ~RTs.getMaskByName('Brain').imageArray, spacing=voxel_size, roi_type='Barrier')

# create virtual GTV as sphere

Brain = RTs.getMaskByName('Brain')
X_world, Y_world, Z_world = Brain.getMeshGridPositions()
RTs.createSphere('GTV', X_world, Y_world, Z_world, Brain.centerOfMass + np.array([30, 50, 0]), (5, 5, 5), voxel_size, roi_type='GTV')

# Perform registration

fa, _, _, = tensor.get_FA_MD_RGB()
rigid = ImageRegistrationRigid(MRI, grid2world, fa, tensor.affine)
mapping = rigid.get_mapping()

# plot IR results
aligned = mapping.transform(fa)
regtools.overlay_slices(MRI, aligned, None, 2, "Static", "Aligned", None)

transform = TransformTensorAffine(mapping)
tensor_aligned = transform.getTensorDiffusionTransformed(tensor, method='ICT', mask=Brain.imageArray)

# reduce calculation  of images and structures

bb = RTs.getBoundingBox('GTV', margin=50)
External = RTs.getMaskByName('External').imageArray
MRI = Image3D(imageArray=MRI, spacing=voxel_size)

MRI.reduceGrid_mask(bb)
RTs.reduceGrid_mask(bb)
tensor_aligned.reduceGrid_mask(bb)

check = tensor_aligned.isPositiveDefinite(mask=RTs.getMaskByName('External').imageArray)
print(check.sum())
# reload contour masks

GTV = RTs.getMaskByName('GTV').imageArray
External = RTs.getMaskByName('External').imageArray
Brain = RTs.getMaskByName('Brain').imageArray
BS = RTs.getMaskByName('BS').imageArray

# define tumor spread model

model = {
    'obstacle': True,
    'model': 'Anisotropic',
    'cell_capacity': 100,  # [%]
    'proliferation_rate': 0.0001,  # [fraction/day],
    'diffusion_magnitude': 0.00001,  # [mm^2/day],
    'system': 'diffusion',
    'timepoint': [10] # list of [days]
    }

# Solver FKPP equation

CTV = CTVDiffusion(rts=RTs, tensor=copy.deepcopy(tensor_aligned), model=model)
CTV.setCTV_isodensity(0.5)

# Create 2D plots

# voxel display
COM = np.array(com(GTV))
Z_coord = int(COM[2])

# prepare figures
plotMR = MRI.imageArray.copy()
plotMR[~External] = 0
plotGTV = GTV.astype(float).copy()
plotGTV[~GTV] = np.NaN
plotCells = CTV.density3D.copy()
plotCells[~External] = 0
_, _, RGB = tensor_aligned.get_FA_MD_RGB()

plt.figure()
plt.imshow(np.flip(plotMR[:, :, Z_coord].transpose(), axis=0), cmap='gray', vmin=0, vmax=2.5)
plt.imshow(np.flip(np.transpose(RGB[:, :, Z_coord], axes=(1,0,2)), axis=0), alpha=0.5)
plt.contourf(np.flip(plotGTV[:, :, Z_coord].transpose(), axis=0), colors='yellow', alpha=0.5)
plt.contour(np.flip(CTV.imageArray[:, :, Z_coord].transpose(), axis=0), colors='white', linewidths=1.5)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

plt.figure()
plt.imshow(np.flip(plotMR[:, :, Z_coord].transpose(), axis=0), cmap='gray', vmin=0, vmax=2.5)
plt.imshow(np.flip(plotCells[:, :, Z_coord].transpose(), axis=0), alpha=0.75)
plt.contour(np.flip(CTV.imageArray[:, :, Z_coord].transpose(), axis=0), colors='white', linewidths=1.5)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
plt.show()