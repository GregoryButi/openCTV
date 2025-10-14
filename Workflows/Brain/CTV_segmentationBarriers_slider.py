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
from dipy.io.image import load_nifti

from opentps.core.data.images._image3D import Image3D

from Process.CTVs import CTVGeometric
from Process import Struct
from Process.Analysis.contourComparison import dice_score, percentile_hausdorff_distance, surface_dice_score
from Process.ImageSegmentation_nnUNet.imageSegmentation import run_segmentation

visualization = True
postprocessing_barriers = True

# input
patient_dir = '/media/gregory/Elements/Data/MGH_Glioma/GLIS-RT_Processed/GLI_016_GBM'
path_CT = os.path.join(patient_dir, f'Therapy-scan/MRI_CT/CT.nii.gz')
path_RTstructs_manual = os.path.join(patient_dir, f'Therapy-scan/Structures')

barrier_names = ['Brain', 'Brainstem', 'Cerebellum', 'Chiasm', 'OpticNerve_L', 'OpticNerve_R', 'Midline', 'Ventricles_connected']
barrier_names_eval = ['Brain', 'Brainstem', 'Cerebellum', 'Chiasm', 'OpticNerves', 'Midline', 'Ventricles_connected']

# nn-UNet parameters
model_segmentation = 'Dataset_CT_Brain_Segmentation_Radiotherapy'
config_segmentation = '3d_fullres'

model = {'model': 'Uniform',
         'obstacle': True}

# model parameters
margin = 15  # [mm]

# evaluation metric
metric = dice_score

# load data

ct, static_grid2world, voxel_size = load_nifti(path_CT, return_voxsize=True)
CT = Image3D(imageArray=ct, spacing=voxel_size)

# load ground truth structures

RTs = Struct()
RTs.loadContours_folder(path_RTstructs_manual, barrier_names)
RTs.loadContours_folder(path_RTstructs_manual, ['GTV'], contour_types=['GTV'])

gtv = RTs.getMaskByName('GTV').imageArray

# load ground truth masks
brain = RTs.getMaskByName('Brain').imageArray
chiasm = RTs.getMaskByName('Chiasm').imageArray
optic_nerves = np.logical_or(RTs.getMaskByName('OpticNerve_L').imageArray,
                             RTs.getMaskByName('OpticNerve_R').imageArray)
ventricles = RTs.getMaskByName('Ventricles_connected').imageArray
brainstem = RTs.getMaskByName('Brainstem').imageArray
cerebellum = RTs.getMaskByName('Cerebellum').imageArray
midline = RTs.getMaskByName('Midline').imageArray

# remove overlaps for similarity tests
brain = np.logical_and(brain, ~ventricles)
brain = np.logical_and(brain, ~chiasm)
brain = np.logical_and(brain, ~brainstem)
brainstem = np.logical_and(brainstem, ~cerebellum)

RTs.setMask('Brain', brain)
RTs.setMask('Brainstem', brainstem)
RTs.setMask('OpticNerves', optic_nerves)

# run DL model and import structures

structure_dictionary = run_segmentation([path_CT], model_segmentation, config_segmentation)

RTs_DL = Struct()
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

if postprocessing_barriers:

    # perform morphological operations (post-processing step)

    Brain_dilated = RTs.getMaskByName('Brain').copy()
    Brain_dilated.dilateMask(radius=(2, 2, 3)) # 2 * voxel_size[0], 2 * voxel_size[1], 1 * voxel_size[2]

    Brain_dilated_DL = RTs_DL.getMaskByName('Brain').copy()
    Brain_dilated_DL.dilateMask(radius=(2, 2, 3))

    cerebellum = np.logical_and(cerebellum, ~Brain_dilated.imageArray)
    cerebellumDL = np.logical_and(cerebellumDL, ~Brain_dilated_DL.imageArray)

# define baseline barrier structures

bs = np.logical_or((brain + cerebellum + chiasm + optic_nerves + brainstem) == 0, (midline + ventricles) > 0)
bsDL = np.logical_or((brainDL + cerebellumDL + chiasmDL + optic_nervesDL + brainstemDL) == 0, (midlineDL + ventriclesDL) > 0)

RTs.setMask('BS', bs, spacing=voxel_size, roi_type='Barrier')
RTs_DL.setMask('BS', bsDL, spacing=voxel_size, roi_type='Barrier')

RTs_DL.setMask('GTV', gtv, spacing=voxel_size, roi_type='GTV')

# define CTV objects
ctv = CTVGeometric(rts=RTs, model=model, spacing=voxel_size)
ctv_DL = CTVGeometric(rts=RTs_DL, model=model, spacing=voxel_size)

# define calculation domain
GTVBox = RTs.getBoundingBox('GTV', int(margin) + 15)

# Run fast marching method

ctv.setCTV_isodistance(margin, solver='FMM', domain=GTVBox)
ctv_DL.setCTV_isodistance(margin, solver='FMM', domain=GTVBox)

# smoothing to remove holes etc.
ctv.smoothMask(size=2)
ctv_DL.smoothMask(size=2)

# compute similarity scores
print(f"CTV DL dice: {metric(np.logical_and(ctv.imageArray, ~gtv), np.logical_and(ctv_DL.imageArray, ~gtv)) * 100:.2f}%")

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
    plotGTV[~gtv] = np.NaN

    plotComp = (~bs).astype(float).copy()

    plotCompDL = (~bsDL).astype(float).copy()
    plotCompDL[bsDL] = np.NaN

    # Build interactive plot

    def plot_isodistance_sagittal(ax, X, plotCTV, plotCTV_DL):

        fig.add_axes(ax)
        plt.imshow(np.flip(plotCT[X, :, :].transpose(), axis=0), extent=[y[0], y[-1], z[0], z[-1]], cmap='gray', vmin=-1000, vmax=600)
        plt.contour(plotComp[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='blue', alpha=0.5)
        plt.contourf(plotGTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='yellow', alpha=0.25)
        plt.contour(plotCTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='green', linewidths=1.5)
        plt.contour(plotCTV_DL[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')

        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    def plot_isodistance_coronal(ax, Y, plotCTV, plotCTV_DL):

        fig.add_axes(ax)
        plt.imshow(np.flip(plotCT[:, Y, :].transpose(), axis=0), extent=[x[0], x[-1], z[0], z[-1]], cmap='gray', vmin=-1000, vmax=600)
        plt.contour(plotComp[:, Y, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='blue', alpha=0.5)
        plt.contourf(plotGTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='yellow', alpha=0.5)
        plt.contour(plotCTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='green', linewidths=1.5)
        plt.contour(plotCTV_DL[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='solid')

        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    def plot_isodistance_axial(ax, Z, plotCTV, plotCTV_DL):

        fig.add_axes(ax)
        plt.imshow(np.flip(plotCT[:, :, Z].transpose(), axis=0), extent=[x[0], x[-1], y[0], y[-1]], cmap='gray', vmin=-1000, vmax=600)
        plt.contour(plotComp[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='blue') #, alpha=0.5)
        plt.contourf(plotGTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='yellow', alpha=0.25)
        plt.contour(plotCTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='green', linewidths=1.5)
        plt.contour(plotCTV_DL[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', linewidths=1.5, linestyles='dashed')

        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    # Plot
    fig = plt.figure()
    plt.axis('off')
    # plt.title('Interactive slider')

    ax1 = fig.add_subplot(131)
    plot_isodistance_sagittal(ax1, X_coord, ctv.imageArray, ctv_DL.imageArray)
    ax2 = fig.add_subplot(132)
    plot_isodistance_coronal(ax2, Y_coord, ctv.imageArray, ctv_DL.imageArray)
    ax3 = fig.add_subplot(133)
    plot_isodistance_axial(ax3, Z_coord, ctv.imageArray, ctv_DL.imageArray)

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

    def update(val):

        alpha1 = int(alpha_slider_1.val)
        alpha2 = int(alpha_slider_2.val)
        alpha3 = int(alpha_slider_3.val)

        ax1.cla()
        plot_isodistance_sagittal(ax1, alpha1, ctv.imageArray, ctv_DL.imageArray)
        ax2.cla()
        plot_isodistance_coronal(ax2, alpha2, ctv.imageArray, ctv_DL.imageArray)
        ax3.cla()
        plot_isodistance_axial(ax3, alpha3, ctv.imageArray, ctv_DL.imageArray)

        plt.draw()

    alpha_slider_1.on_changed(update)
    alpha_slider_2.on_changed(update)
    alpha_slider_3.on_changed(update)

    plt.show()