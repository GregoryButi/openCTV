#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:12:05 2024

@author: gregory
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass as com
from matplotlib.widgets import Slider
from dipy.viz import regtools
from dipy.io.image import load_nifti

from opentps.core.data.images._image3D import Image3D

from Process.Tensors import TensorDiffusion
from Process.CTVs import CTVGeometric
from Process.Transforms import TransformTensorDeformable
from Process.ImageRegistrationDIPY import ImageRegistrationDeformable
from Process import Struct

# input

path_MRI = '../../Input/Kim/Therapy-scan/MRI_CT/T1_norm.nii.gz'
path_RTstructs = '../../Input/Kim/Therapy-scan/Structures'
path_atlas = '../../Input/Atlas/MNI152_T1_1mm_brain_norm.nii.gz'
path_tensor = '../../Input/Atlas/FSL_HCP1065_tensor_1mm_Ants.nii.gz'

# load data

MRI, static_grid2world, voxel_size = load_nifti(path_MRI, return_voxsize=True)
template, moving_grid2world = load_nifti(path_atlas)

tensor = TensorDiffusion()
tensor.loadTensor(path_tensor, format='ANTs')

# load structures

RTs = Struct()
RTs.loadContours_folder(path_RTstructs, ['Brain', 'WM', 'External'], contour_types=[None, 'Barrier_soft', None])

# define barrier structures
RTs.setMask('BS', ~RTs.getMaskByName('Brain').imageArray, spacing=voxel_size, roi_type='Barrier')

# create virtual GTV as sphere

Brain = RTs.getMaskByName('Brain')
X_world, Y_world, Z_world = Brain.getMeshGridPositions()
RTs.createSphere('GTV', X_world, Y_world, Z_world, Brain.centerOfMass + np.array([30, 50, 0]), (5, 5, 5), voxel_size, roi_type='GTV')

# define target image for registration
target = np.where(Brain.imageArray, MRI, 0.)

# Perform registration

diffeomorphic = ImageRegistrationDeformable(target, static_grid2world, template, moving_grid2world, level_iters=[10, 10, 5], metric='CC')
mapping = diffeomorphic.get_mapping()

# plot DIR results
warped = mapping.transform(template)
regtools.overlay_slices(target, warped, None, 2, "Static", "Warped", None)

# apply tensor transformation

transform = TransformTensorDeformable(mapping)
tensor_transformed = transform.getTensorDiffusionTransformed(tensor, method='ICT')

# reduce calculation grid of images and structures

External = RTs.getMaskByName('External').imageArray
MRI = Image3D(imageArray=MRI, spacing=voxel_size)

MRI.reduceGrid_mask(External)
RTs.reduceGrid_mask(External)
tensor.reduceGrid_mask(External)
tensor_transformed.reduceGrid_mask(External)

# reload contour masks

GTV = RTs.getMaskByName('GTV').imageArray
External = RTs.getMaskByName('External').imageArray
Brain = RTs.getMaskByName('Brain').imageArray
BS = RTs.getMaskByName('BS').imageArray
WM = RTs.getMaskByName('WM').imageArray

# define tumor resistance model

modelDTI = {
    'obstacle': True,
    'model': 'Anisotropic',
    'model-DTI': 'Rekik',
    }

resistance = [1., 0.5, 0.1, 0.05]

# GTV-to-CTV margin
margin = 20

# Run fast marching method for classic CTV

ctv_classic = CTVGeometric(rts=RTs, model={'model': None, 'obstacle': True})
ctv_classic.setCTV_isodistance(margin)
volume = ctv_classic.getVolume()
ctv_classic.smoothMask()

# Run fast marching method for anisotropic CTV

# sweep over resistance values
CTVs_dti = np.zeros(np.append(MRI.gridSize, [len(resistance)]))
for j in range(len(resistance)):

    ctv_dti = CTVGeometric(rts=RTs, model=modelDTI, tensor=tensor_transformed)

    # update model
    ctv_dti.preferred['resistances'] = [resistance[j]]

    ctv_dti.setCTV_metric(ctv_classic, metric='volume', x0=margin)
    ctv_dti.smoothMask()

    # store mask in array
    CTVs_dti[:, :, :, j] = ctv_dti.imageArray
    
# Create 2D plots

# voxel display
COM = np.array(com(GTV))
Z_coord = int(COM[2])

# prepare figures
plotMR = MRI.imageArray.copy()
plotMR[~External] = 0
plotGTV = GTV.astype(float).copy()
plotGTV[~GTV] = np.NaN
_, _, RGB = tensor_transformed.get_FA_MD_RGB()

plt.figure()
plt.imshow(np.flip(plotMR[:, :, Z_coord].transpose(), axis=0), cmap='gray', vmin=0, vmax=2.5)
plt.imshow(np.flip(np.transpose(RGB[:, :, Z_coord], axes=(1,0,2)), axis=0), alpha=0.75)
plt.contourf(np.flip(plotGTV[:, :, Z_coord].transpose(), axis=0), colors='yellow', alpha=0.5)
plt.contour(np.flip(ctv_classic.imageArray[:, :, Z_coord].transpose(), axis=0), colors='green', linewidths=1.5)
plt.contour(np.flip(ctv_dti.imageArray[:, :, Z_coord].transpose(), axis=0), colors='white', linewidths=1.5)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

# Build interactive plot

def plot_isodistance(ax, Z, j):
    fig.add_axes(ax)
    plt.imshow(np.flip(plotMR[:, :, Z].transpose(), axis=0), cmap='gray', vmin=0, vmax=2.5)
    plt.contourf(np.flip(plotGTV[:, :, Z].transpose(), axis=0), colors='yellow', alpha=0.5)
    plt.contour(np.flip(ctv_classic.imageArray[:, :, Z_coord].transpose(), axis=0), colors='green', linewidths=1.5)
    plt.contour(np.flip(CTVs_dti[:, :, Z, j].transpose(), axis=0), colors='white')

    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

# Plot
fig = plt.figure()
plt.axis('off')
# plt.title('Interactive slider')

ax = fig.add_subplot(111)
plot_isodistance(ax, Z_coord, 0)

# Define sliders

# Make a vertically oriented slider to control the slice
# position x, position y, x-length, y-length
alpha_axis_1 = plt.axes([0.2, 0.2, 0.0125, 0.62])
alpha_slider_1 = Slider(
    ax=alpha_axis_1,
    label="Slice",
    valmin=0,
    valmax=MRI.gridSize[2],
    valinit=Z_coord,
    valstep=1,
    orientation="vertical"
)

# Make horizontal oriented slider to control the ''isotropicness''
alpha_axis_2 = plt.axes([0.3, 0.02, 0.4, 0.03])
alpha_slider_2 = Slider(
    ax=alpha_axis_2,
    label='Resistance',
    valmin=0,
    valmax=len(resistance) - 1,
    valinit=0,
    valstep=1
)


def update(val):
    alpha1 = int(alpha_slider_1.val)
    alpha2 = int(alpha_slider_2.val)

    ax.cla()
    plot_isodistance(ax, alpha1, alpha2)

    plt.draw()

alpha_slider_1.on_changed(update)
alpha_slider_2.on_changed(update)

plt.show()
