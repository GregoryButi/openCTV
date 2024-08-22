#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:12:05 2024

@author: gregory
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass as com
from dipy.io.image import load_nifti
import copy

from opentps.core.data.images._image3D import Image3D
from Process.Tensors import TensorMetric, TensorDiffusion
from Process.CTVs import CTVGeometric, CTVDiffusion
from Process import Struct

# input

path_MRI = '../../Input/Kim/Therapy-scan/MRI_CT/T1_norm.nii.gz'
path_RTstructs = '../../Input/Kim/Therapy-scan/Structures'
path_tensor = '../../Input/Kim/Therapy-scan/fMRI/Tensor.nii.gz'

# load data

MRI, grid2world, voxel_size = load_nifti(path_MRI, return_voxsize=True)

tensor = TensorDiffusion()
tensor.loadTensor(path_tensor)

# load structures

RTs = Struct()
RTs.loadContours_folder(path_RTstructs, ['Brain', 'ExternalT1'])
#RTs.loadContours_folder(path_RTstructs, ['Brain', 'WM', 'ExternalT1'])

# define barrier structures
RTs.setMask('BS', ~RTs.getMaskByName('Brain').imageArray, voxel_size)

# define structure of preferred spread
#RTs.setMask('PS', RTs.getMaskByName('Brain').imageArray, voxel_size)

# create virtual GTV as sphere

Brain = RTs.getMaskByName('Brain')
X_world, Y_world, Z_world = Brain.getMeshGridPositions()
RTs.createSphere('GTV', X_world, Y_world, Z_world, Brain.centerOfMass + np.array([30, 50, 0]), (5, 5, 5), voxel_size)

# reduce calculation  of images and structures

External = RTs.getMaskByName('ExternalT1').imageArray
MRI = Image3D(imageArray=MRI, spacing=voxel_size)

MRI.reduceGrid_mask(External)
RTs.reduceGrid_mask(External)
tensor.reduceGrid_mask(External)

# reload contour masks

GTV = RTs.getMaskByName('GTV').imageArray
External = RTs.getMaskByName('ExternalT1').imageArray
Brain = RTs.getMaskByName('Brain').imageArray
BS = RTs.getMaskByName('BS').imageArray
#WM = RTs.getMaskByName('WM').imageArray

# define tumor spread model

model = {
    'obstacle': True,
    'model': 'Anisotropic',
    'cell_capacity': 100,  # [%]
    'proliferation_rate': 0.01,  # [fraction/day],
    'diffusion_magnitude': 0.025,  # [mm^2/day],
    'system': 'reaction_diffusion',
    'timepoint': [200] # list of [days]
    }

margin = 20

# Solver FKPP equation

ctv_dti = CTVDiffusion()
ctv_dti.setCTV_isodensity(1, RTs, tensor=copy.deepcopy(tensor), model=model)
ctv_dti.smoothMask(BS)
    
# Create 2D plots

# voxel display
COM = np.array(com(GTV))
Z_coord = int(COM[2])

# prepare figures
plotMR = MRI.imageArray.copy()
plotMR[~External] = 0
plotGTV = GTV.astype(float).copy()
plotGTV[~GTV] = np.NaN
_, _, RGB = tensor.get_FA_MD_RGB()

plt.figure()
plt.imshow(np.flip(plotMR[:, :, Z_coord].transpose(), axis=0), cmap='gray', vmin=0, vmax=2.5)
plt.imshow(np.flip(np.transpose(RGB[:, :, Z_coord], axes=(1,0,2)), axis=0), alpha=0.5)
plt.contourf(np.flip(plotGTV[:, :, Z_coord].transpose(), axis=0), colors='yellow', alpha=0.5)
plt.contour(np.flip(ctv_dti.imageArray[:, :, Z_coord].transpose(), axis=0), colors='white', linewidths=1.5)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    
#plt.savefig(os.path.join(os.getcwd(),'CTV_'+modelDTI['model-DTI']+'.pdf'), format='pdf',bbox_inches='tight')

plt.show()