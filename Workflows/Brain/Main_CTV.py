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
from Process.Tensors import TensorMetric
from Process.CTVs import CTVGeometric
from Process import Struct

# input

path_MRI = '/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Glioma/GLIS-RT/GLI_003_AAC/MNI152/T1_norm.nii.gz'
path_RTstructs = '/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Glioma/GLIS-RT/GLI_003_AAC/MNI152/'
path_tensor = '/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Glioma/GLIS-RT/GLI_003_AAC/MNI152/MT_deformable_PDR_DIPY.nii.gz'

# load data

MRI, grid2world, voxel_size = load_nifti(path_MRI, return_voxsize=True)

tensor = TensorMetric()
tensor.loadTensor(path_tensor, 'metric')

# load structures

RTs = Struct()
RTs.loadContours_folder(path_RTstructs, ['GTV', 'CTV', 'BS', 'CC', 'Brain_mask', 'WM', 'GM', 'External'])

RTs.smoothMasks(['GTV', 'CTV', 'CC', 'Brain_mask', 'External'])

BS = np.logical_or(~RTs.getMaskByName('Brain_mask').imageArray, RTs.getMaskByName('BS').imageArray)
RTs.setMask('BS', BS, voxel_size)

# define structure of preferred spread
RTs.setMask('PS', np.logical_and(RTs.getMaskByName('WM').imageArray, ~RTs.getMaskByName('BS').imageArray), voxel_size)

MRI = Image3D(imageArray=MRI, spacing=voxel_size)

External = RTs.getMaskByName('External').imageArray

# reduce calculation grid

MRI.reduceGrid_mask(External)
RTs.reduceGrid_mask(External)
tensor.reduceGrid_mask(External)

# reload contours

GTV = RTs.getMaskByName('GTV').imageArray
External = RTs.getMaskByName('External').imageArray
Brain = RTs.getMaskByName('Brain_mask').imageArray
CC = RTs.getMaskByName('CC').imageArray
BS = RTs.getMaskByName('BS').imageArray
WM = RTs.getMaskByName('WM').imageArray
GM = RTs.getMaskByName('GM').imageArray
    
# define model

modelDTI = {
    'obstacle': True,
    'model': 'Anisotropic', # None, 'Nonuniform', 'Anisotropic'
    'model-DTI': 'Clatz', # 'Clatz', 'Rekik'
    'resistance': 0.1,
    'anisotropy': 1.0
    }

margin = 20

# Run fast marching method for classic CTV

ctv_classic = CTVGeometric()
ctv_classic.setCTV_isodistance(margin, RTs, model = {'model': None, 'obstacle': True})
volume = ctv_classic.getVolume()

# smooth masks and remove holes
ctv_classic.smoothMask(BS)

# Run fast marching method

ctv_dti = CTVGeometric()
ctv_dti.setCTV_volume(volume, RTs, tensor=copy.deepcopy(tensor), model = modelDTI, x0 = margin)
ctv_dti.smoothMask(BS)
    
# Create 2D plots

# GTV COM for display
COM = np.array(com(GTV))
X_coord = int(COM[0])
Y_coord = int(COM[1])
Z_coord = int(COM[2])

plotMR = MRI.imageArray.copy()
plotMR[~External] = 0
plotCC = CC.astype(float).copy()
plotCC[~CC] = np.NaN
plotGTV = GTV.astype(float).copy()
plotGTV[~GTV] = np.NaN
ctv_dti.distance3D[BS] = np.NaN

plt.imshow(np.flip(plotMR[:, :, Z_coord].transpose(), axis=0), cmap='gray', vmin=0, vmax=2.5)
plt.contourf(np.flip(plotGTV[:, :, Z_coord].transpose(), axis=0), colors='yellow', alpha=0.4)
plt.contourf(np.flip(plotCC[:, :, Z_coord].transpose(), axis=0), colors='blue', alpha=0.4)
plt.contour(np.flip(ctv_classic.imageArray[:, :, Z_coord].transpose(), axis=0), colors='green', linewidths=1.5)
plt.contour(np.flip(ctv_dti.imageArray[:, :, Z_coord].transpose(), axis=0), colors='white', linewidths=1.5)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    
#plt.savefig(os.path.join(os.getcwd(),'CTV_'+modelDTI['model-DTI']+'.pdf'), format='pdf',bbox_inches='tight')

plt.show()
