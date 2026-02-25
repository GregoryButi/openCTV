#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:12:05 2024

@author: gregory
"""

import numpy as np
import os

import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass as com
from matplotlib.widgets import Slider
from dipy.io.image import load_nifti, save_nifti
from totalsegmentator.python_api import totalsegmentator

from opentps.core.data.images._image3D import Image3D

from Process.CTVs import CTVGeometric
from Process import Struct
from Process.ImageSegmentation_nnUNet.imageSegmentation import run_segmentation
from Process.config_runtime import configure_agd
configure_agd()

visualization = True

# patient ID
PID = 'GBM'

barrier_names = ['Brain', 'Brainstem', 'Cerebellum', 'Chiasm', 'OpticNerve_L', 'OpticNerve_R', 'Midline', 'Ventricles_connected']

# nn-UNet parameters
model_segmentation = 'Dataset_CT_Brain_Segmentation_Radiotherapy'
config_segmentation = '3d_fullres'

model = {'model': 'Uniform',
         'obstacle': True
         }

# model parameters
margin = 15  # [mm]

# evaluation metrics
SDS_tolerance = [1, 2, 3]  # [mm]

# input

path_CT = f'../../Input/{PID}/Therapy-scan/MRI_CT/CT.nii.gz'
path_RTstructs = f'../../Input/{PID}/Therapy-scan/Structures'
path_RTstructs_DL = f'../../Input/{PID}/Therapy-scan/Structures_{model_segmentation}_{config_segmentation}'
path_TotalSegmentator = f'../../Input/{PID}/Therapy-scan/Structures_totalsegmentator'

# load data

ct, static_grid2world, voxel_size = load_nifti(path_CT, return_voxsize=True)
CT = Image3D(imageArray=ct, spacing=voxel_size)

# define structure object

RTs_DL = Struct()

# import GTV
RTs_DL.loadContours_folder(path_RTstructs, ['GTV'], contour_types=['GTV'])

# load gtv mask
gtv = RTs_DL.getMaskByName('GTV').imageArray

# run DL model and import structures

structure_dictionary = run_segmentation([path_CT], model_segmentation, config_segmentation)

for name in barrier_names:
    if name in structure_dictionary['labels']:
        index = structure_dictionary['labels'][name]
        RTs_DL.setMask(name, structure_dictionary['array'][..., index], grid2world=structure_dictionary['affine'])

brainDL = RTs_DL.getMaskByName('Brain').imageArray
chiasmDL = RTs_DL.getMaskByName('Chiasm').imageArray
optic_nervesDL = np.logical_or(RTs_DL.getMaskByName('OpticNerve_L').imageArray,
                             RTs_DL.getMaskByName('OpticNerve_R').imageArray)
brainstemDL = RTs_DL.getMaskByName('Brainstem').imageArray
cerebellumDL = RTs_DL.getMaskByName('Cerebellum').imageArray
midlineDL = RTs_DL.getMaskByName('Midline').imageArray
ventriclesDL = RTs_DL.getMaskByName('Ventricles_connected').imageArray

RTs_DL.setMask('OpticNerves', optic_nervesDL)

# load TotalSegmentator structures

# totalsegmentator(path_CT, path_TotalSegmentator, task="total") # brain_structures

RTs_TS = Struct()
RTs_TS.loadContours_folder(path_TotalSegmentator, ['brainstem', 'cerebellum', 'ventricle', 'caudate_nucleus',
                                                   'central_sulcus', 'frontal_lobe', 'insular_cortex', 'internal_capsule',
                                                   'lentiform_nucleus', 'occipital_lobe', 'parietal_lobe', 'septum_pellucidum',
                                                   'subarachnoid_space', 'temporal_lobe', 'thalamus'])

brainTS = ((RTs_TS.getMaskByName('caudate_nucleus').imageArray).astype(int)
           + (RTs_TS.getMaskByName('central_sulcus').imageArray).astype(int)
           + (RTs_TS.getMaskByName('frontal_lobe').imageArray).astype(int)
           + (RTs_TS.getMaskByName('insular_cortex').imageArray).astype(int)
           + (RTs_TS.getMaskByName('internal_capsule').imageArray).astype(int)
           + (RTs_TS.getMaskByName('lentiform_nucleus').imageArray).astype(int)
           + (RTs_TS.getMaskByName('occipital_lobe').imageArray).astype(int)
           + (RTs_TS.getMaskByName('parietal_lobe').imageArray).astype(int)
           + (RTs_TS.getMaskByName('septum_pellucidum').imageArray).astype(int)
           + (RTs_TS.getMaskByName('subarachnoid_space').imageArray).astype(int)
           + (RTs_TS.getMaskByName('temporal_lobe').imageArray).astype(int)
           + (RTs_TS.getMaskByName('thalamus').imageArray).astype(int))
brainstemTS = (RTs_TS.getMaskByName('brainstem').imageArray).astype(int)
cerebellumTS = (RTs_TS.getMaskByName('cerebellum').imageArray).astype(int)
ventriclesTS = (RTs_TS.getMaskByName('ventricle').imageArray).astype(int)

RTs_TS.setMask('Brain', brainTS)

# perform morphological operations (post-processing step)

Brain_dilated_DL = RTs_DL.getMaskByName('Brain').copy()
Brain_dilated_DL.dilateMask(radius=(2, 2, 3))

Brain_dilated_TS = RTs_TS.getMaskByName('Brain').copy()
Brain_dilated_TS.dilateMask(radius=(2, 2, 3))

cerebellumDL = np.logical_and(cerebellumDL, ~Brain_dilated_DL.imageArray)
cerebellumTS = np.logical_and(cerebellumTS, ~Brain_dilated_TS.imageArray)

# define barrier structures

bsDL = np.logical_or((brainDL + cerebellumDL + chiasmDL + optic_nervesDL + brainstemDL) == 0, (midlineDL + ventriclesDL) > 0)
bsTS = np.logical_or((brainTS + cerebellumTS + brainstemTS) == 0, ventriclesTS)

RTs_DL.setMask('BS', bsDL, spacing=voxel_size, roi_type='Barrier')
RTs_TS.setMask('BS', bsTS, spacing=voxel_size, roi_type='Barrier')

RTs_DL.setMask('GTV', gtv, spacing=voxel_size, roi_type='GTV')
RTs_TS.setMask('GTV', gtv, spacing=voxel_size, roi_type='GTV')

# Run distance transform

ctv_DL = CTVGeometric(rts=RTs_DL, model=model, spacing=voxel_size)
ctv_TS = CTVGeometric(rts=RTs_TS, model=model, spacing=voxel_size)

# define calculation domain
GTVBox = RTs_DL.getBoundingBox('GTV', int(margin) + 15)

ctv_DL.setCTV_isodistance(margin, solver='FMM', domain=GTVBox)
ctv_TS.setCTV_isodistance(margin, solver='FMM', domain=GTVBox)

# smoothing to remove holes etc.
ctv_DL.smoothMask(size=2)
ctv_TS.smoothMask(size=2)

# Create 2D plots

if visualization:

    # voxel display
    COM = np.array(com(gtv))
    X_coord = int(COM[0])
    Y_coord = int(COM[1])
    Z_coord = int(COM[2])

    # prepare figures
    plotCT = CT.imageArray

    x, y, z = CT.getMeshGridAxes()

    plotGTV = gtv.astype(float).copy()
    plotGTV[~gtv] = np.nan

    # Build interactive plot

    def plot_isodistance_sagittal(ax, X, plotCTV_DL, plotCTV_TS):

        fig.add_axes(ax)
        plt.imshow(np.flip(plotCT[X, :, :].transpose(), axis=0), extent=[y[0], y[-1], z[0], z[-1]], cmap='gray', vmin=-1000, vmax=600)
        plt.contourf(plotGTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='yellow', alpha=0.25)
        plt.contour(plotCTV_DL[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')
        plt.contour(plotCTV_TS[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='orange', linewidths=1.5, linestyles='dashed')

        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    def plot_isodistance_coronal(ax, Y, plotCTV_DL, plotCTV_TS):

        fig.add_axes(ax)
        plt.imshow(np.flip(plotCT[:, Y, :].transpose(), axis=0), extent=[x[0], x[-1], z[0], z[-1]], cmap='gray', vmin=-1000, vmax=600)
        plt.contourf(plotGTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='yellow', alpha=0.5)
        plt.contour(plotCTV_DL[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='solid')
        plt.contour(plotCTV_TS[:, Y, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='orange', linewidths=1.5, linestyles='dashed')

        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    def plot_isodistance_axial(ax, Z, plotCTV_DL, plotCTV_TS):

        fig.add_axes(ax)
        plt.imshow(np.flip(plotCT[:, :, Z].transpose(), axis=0), extent=[x[0], x[-1], y[0], y[-1]], cmap='gray', vmin=-1000, vmax=600)
        plt.contourf(plotGTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='yellow', alpha=0.25)
        plt.contour(plotCTV_DL[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', linewidths=1.5, linestyles='dashed')
        plt.contour(plotCTV_TS[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='orange', linewidths=1.5, linestyles='dashed')

        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    # Plot
    fig = plt.figure()
    plt.axis('off')

    ax1 = fig.add_subplot(131)
    plot_isodistance_sagittal(ax1, X_coord, ctv_DL.imageArray, ctv_TS.imageArray)
    ax2 = fig.add_subplot(132)
    plot_isodistance_coronal(ax2, Y_coord, ctv_DL.imageArray, ctv_TS.imageArray)
    ax3 = fig.add_subplot(133)
    plot_isodistance_axial(ax3, Z_coord, ctv_DL.imageArray, ctv_TS.imageArray)

    # Define sliders

    # Make a vertically oriented slider to control the slice
    # position x, position y, x-length, y-length
    alpha_axis_1 = plt.axes([0.1, 0.3, 0.0125, 0.42])
    alpha_slider_1 = Slider(
        ax=alpha_axis_1,
        label="Slice",
        valmin=0,
        valmax=CT.gridSize[0],
        valinit=X_coord,
        valstep=1,
        orientation="vertical"
    )

    alpha_axis_2 = plt.axes([0.38, 0.3, 0.0125, 0.42])
    alpha_slider_2 = Slider(
        ax=alpha_axis_2,
        label="Slice",
        valmin=0,
        valmax=CT.gridSize[1],
        valinit=Y_coord,
        valstep=1,
        orientation="vertical"
    )

    alpha_axis_3 = plt.axes([0.65, 0.3, 0.0125, 0.42])
    alpha_slider_3 = Slider(
        ax=alpha_axis_3,
        label="Slice",
        valmin=0,
        valmax=CT.gridSize[2],
        valinit=Z_coord,
        valstep=1,
        orientation="vertical"
    )

    def update1(val):

        alpha1 = int(alpha_slider_1.val)
        alpha2 = int(alpha_slider_2.val)
        alpha3 = int(alpha_slider_3.val)

        ax1.cla()
        plot_isodistance_sagittal(ax1, alpha1, ctv_DL.imageArray, ctv_TS.imageArray)
        ax2.cla()
        plot_isodistance_coronal(ax2, alpha2, ctv_DL.imageArray, ctv_TS.imageArray)
        ax3.cla()
        plot_isodistance_axial(ax3, alpha3, ctv_DL.imageArray, ctv_TS.imageArray)

        plt.draw()

    alpha_slider_1.on_changed(update1)
    alpha_slider_2.on_changed(update1)
    alpha_slider_3.on_changed(update1)

    plt.show()