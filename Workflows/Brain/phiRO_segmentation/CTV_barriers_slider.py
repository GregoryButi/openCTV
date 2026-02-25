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

from opentps.core.data.images._image3D import Image3D

from Process.CTVs import CTVGeometric
from Process import Struct
from Process.Analysis.contourComparison import dice_score, percentile_hausdorff_distance, surface_dice_score

visualization = True
postprocessing_barriers = True

# patient IDs
patient_dir = '/media/gregory/Elements/Data/MGH_Glioma/GLIS-RT_Processed'
PIDs = ['GLI_016_GBM', 'GLI_019_GBM', 'GLI_022_GBM', 'GLI_023_GBM', 'GLI_028_GBM', 'GLI_033_GBM', 'GLI_037_GBM', 'GLI_044', 'GLI_055_GBM', 'GLI_061_GBM', 'GLI_070_GBM', 'GLI_072_GBM', 'GLI_086_GBM', 'GLI_100_GBM', 'GLI_107_GBM', 'GLI_122_AAC', 'GLI_165_GBM',  'GLI_167_GBM', 'GLI_168_GBM','GLI_184_GBM']

# Set the output directory
output_folder = "/home/gregory/Documents/Projects/CTV_RO1/Results/New"
os.makedirs(output_folder, exist_ok=True)

barrier_names = ['Brain', 'Brainstem', 'Cerebellum', 'Chiasm', 'OpticNerve_L', 'OpticNerve_R', 'Midline', 'Ventricles_connected']
barrier_names_eval = ['Brain', 'Brainstem', 'Cerebellum', 'Chiasm', 'OpticNerves', 'Midline', 'Ventricles_connected']

# nn-UNet parameters
model_segmentation = 'Dataset015_GLIS-RT'
config_segmentation = '3d_fullres'

model = {'model': 'Uniform',
         'obstacle': True
         }

# model parameters
margins = [15]  # [mm]

# evaluation metrics
SDS_tolerance = [1, 2, 3]  # [mm]

dice_ctv_DL = np.zeros((len(PIDs), len(margins)))
HD95_ctv_DL = np.zeros((len(PIDs), len(margins)))
SDS_ctv_DL = np.zeros((len(PIDs), len(margins), len(SDS_tolerance)))
dice_ctv_TS = np.zeros((len(PIDs), len(margins)))
HD95_ctv_TS = np.zeros((len(PIDs), len(margins)))
SDS_ctv_TS = np.zeros((len(PIDs), len(margins), len(SDS_tolerance)))
dice_barriers_DL = np.zeros((len(PIDs), len(barrier_names_eval)))
HD95_barriers_DL = np.zeros((len(PIDs), len(barrier_names_eval)))
SDS_barriers_DL = np.zeros((len(PIDs), len(barrier_names_eval), len(SDS_tolerance)))

for PID, i in zip(PIDs, range(len(PIDs))):

    # input

    path_CT = os.path.join(patient_dir, f'{PID}/Therapy-scan/MRI_CT/CT.nii.gz')
    path_RTstructs = os.path.join(patient_dir, f'{PID}/Therapy-scan/Structures')
    path_RTstructs_DL = os.path.join(patient_dir, f'{PID}/Therapy-scan/Structures_{model_segmentation}_{config_segmentation}')
    path_TotalSegmentator = os.path.join(patient_dir, f'{PID}/Therapy-scan/Structures_totalsegmentator')

    # load data

    ct, static_grid2world, voxel_size = load_nifti(path_CT, return_voxsize=True)
    CT = Image3D(imageArray=ct, spacing=voxel_size)

    # load ground truth structures

    RTs = Struct()
    RTs.loadContours_folder(path_RTstructs, barrier_names)
    RTs.loadContours_folder(path_RTstructs, ['GTV'], contour_types=['GTV'])

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

    # load DL model structures

    RTs_DL = Struct()
    RTs_DL.loadContours_folder(path_RTstructs_DL, barrier_names)

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

    if postprocessing_barriers:

        # perform morphological operations (post-processing step)

        Brain_dilated = RTs.getMaskByName('Brain').copy()
        Brain_dilated.dilateMask(radius=(2, 2, 3)) # 2 * voxel_size[0], 2 * voxel_size[1], 1 * voxel_size[2]

        Brain_dilated_DL = RTs_DL.getMaskByName('Brain').copy()
        Brain_dilated_DL.dilateMask(radius=(2, 2, 3))

        Brain_dilated_TS = RTs_TS.getMaskByName('Brain').copy()
        Brain_dilated_TS.dilateMask(radius=(2, 2, 3))

        cerebellum = np.logical_and(cerebellum, ~Brain_dilated.imageArray)
        cerebellumDL = np.logical_and(cerebellumDL, ~Brain_dilated_DL.imageArray)
        cerebellumTS = np.logical_and(cerebellumTS, ~Brain_dilated_TS.imageArray)

    # define baseline barrier structures

    bs = np.logical_or((brain + cerebellum + chiasm + optic_nerves + brainstem) == 0, (midline + ventricles) > 0)
    bsDL = np.logical_or((brainDL + cerebellumDL + chiasmDL + optic_nervesDL + brainstemDL) == 0, (midlineDL + ventriclesDL) > 0)
    bsTS = np.logical_or((brainTS + cerebellumTS + brainstemTS) == 0, ventriclesTS)

    RTs.setMask('BS', bs, spacing=voxel_size, roi_type='Barrier')
    RTs_DL.setMask('BS', bsDL, spacing=voxel_size, roi_type='Barrier')
    RTs_TS.setMask('BS', bsTS, spacing=voxel_size, roi_type='Barrier')

    RTs_DL.setMask('GTV', gtv, spacing=voxel_size, roi_type='GTV')
    RTs_TS.setMask('GTV', gtv, spacing=voxel_size, roi_type='GTV')

    # Run fast marching method

    ctv = CTVGeometric(rts=RTs, model=model, spacing=voxel_size)
    ctv_DL = CTVGeometric(rts=RTs_DL, model=model, spacing=voxel_size)
    ctv_TS = CTVGeometric(rts=RTs_TS, model=model, spacing=voxel_size)

    for margin, m in zip(margins, range(len(margins))):

        # define calculation domain
        GTVBox = RTs.getBoundingBox('GTV', int(margin) + 15)

        ctv.setCTV_isodistance(margin, solver='FMM', domain=GTVBox)
        ctv_DL.setCTV_isodistance(margin, solver='FMM', domain=GTVBox)
        ctv_TS.setCTV_isodistance(margin, solver='FMM', domain=GTVBox)

        # smoothing to remove holes etc.
        ctv.smoothMask(size=2)
        ctv_DL.smoothMask(size=2)
        ctv_TS.smoothMask(size=2)

        # compute and store metrics for CTVs (remove gtv from volume overlap evaluation)
        dice_ctv_DL[i, m] = dice_score(np.logical_and(ctv.imageArray, ~gtv), np.logical_and(ctv_DL.imageArray, ~gtv))
        dice_ctv_TS[i, m] = dice_score(np.logical_and(ctv.imageArray, ~gtv), np.logical_and(ctv_TS.imageArray, ~gtv))

        for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
            SDS_ctv_DL[i, m, j] = surface_dice_score(ctv.imageArray, ctv_DL.imageArray, tau, voxel_spacing=voxel_size)
            SDS_ctv_TS[i, m, j] = surface_dice_score(ctv.imageArray, ctv_TS.imageArray, tau, voxel_spacing=voxel_size)

        HD95_ctv_DL[i, m] = percentile_hausdorff_distance(ctv.getMeshpoints(), ctv_DL.getMeshpoints(), percentile=95, voxel_spacing=voxel_size)
        HD95_ctv_TS[i, m] = percentile_hausdorff_distance(ctv.getMeshpoints(), ctv_TS.getMeshpoints(), percentile=95, voxel_spacing=voxel_size)

        print(f"patient -- {PID}")

        print(f"{margin} mm CTV DL dice: {dice_ctv_DL[i, m]*100:.2f}%")
        for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
            print(f"{margin} mm CTV DL surface dice (\u03C4 = {tau} mm): {SDS_ctv_DL[i, m, j]*100:.2f}%")
        print(f"{margin} mm CTV DL HD95: {HD95_ctv_DL[i, m]:.2f} mm")

        print(f"     -------     ")

        print(f"{margin} mm CTV TS dice: {dice_ctv_TS[i, m] * 100:.2f}%")
        for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
            print(f"{margin} mm CTV TS surface dice (\u03C4 = {tau} mm): {SDS_ctv_TS[i, m, j] * 100:.2f}%")
        print(f"{margin} mm CTV TS HD95: {HD95_ctv_TS[i, m]:.2f} mm")

        print(f"     -------     ")
        print(f"     -------     ")

        # Construct the filename
        filename = os.path.join(output_folder, f"{PID}_DL_results.txt")

        with open(filename, "w") as f:
            f.write(f"##################################\n")
            f.write(f"Patient -- {PID}\n")
            f.write(f"##################################\n\n")

            f.write(f"CTV dice: {dice_ctv_DL[i, m] * 100:.2f}%\n")
            for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
                f.write(f"CTV surface dice (\u03C4 = {tau} mm): {SDS_ctv_DL[i, m, j] * 100:.2f}%\n")
            f.write(f"CTV HD95: {HD95_ctv_DL[i, m]:.2f} mm\n")

            f.write(f"##################################\n")
            f.write(f"##################################\n")
            f.write(f"##################################\n")

        # Construct the filename
        filename = os.path.join(output_folder, f"{PID}_TS_results.txt")

        with open(filename, "w") as f:
            f.write(f"##################################\n")
            f.write(f"Patient -- {PID}\n")
            f.write(f"##################################\n\n")

            f.write(f"CTV dice: {dice_ctv_TS[i, m] * 100:.2f}%\n")
            for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
                f.write(f"CTV surface dice (\u03C4 = {tau} mm): {SDS_ctv_TS[i, m, j] * 100:.2f}%\n")
            f.write(f"CTV HD95: {HD95_ctv_TS[i, m]:.2f} mm\n")

            f.write(f"##################################\n")
            f.write(f"##################################\n")
            f.write(f"##################################\n")

        # save CTV

        directory = os.path.join(os.path.join(os.path.join(patient_dir, PID), 'Therapy-scan'), 'Targets_'+model_segmentation+'_'+config_segmentation)
        if not os.path.exists(directory):
            os.makedirs(directory)

        name_CTV_ref = 'CTV_ref'
        name_CTV_DL = 'CTV_auto'
        name_CTV_TS = 'CTV_TS'

        save_nifti(os.path.join(directory, name_CTV_DL+'.nii.gz'), ctv_DL.imageArray.astype(float), static_grid2world)
        save_nifti(os.path.join(directory, name_CTV_ref + '.nii.gz'), ctv.imageArray.astype(float), static_grid2world)
        save_nifti(os.path.join(directory, name_CTV_TS + '.nii.gz'), ctv_TS.imageArray.astype(float), static_grid2world)

    # compute and store metrics for barriers

    for name, b in zip(barrier_names_eval, range(len(barrier_names_eval))):

        barrier_DL = RTs_DL.getMaskByName(name)
        barrier = RTs.getMaskByName(name)

        dice_barriers_DL[i, b] = dice_score(barrier.imageArray, barrier_DL.imageArray)

        for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
            SDS_barriers_DL[i, b, j] = surface_dice_score(barrier.imageArray, barrier_DL.imageArray, tau, voxel_spacing=voxel_size)

        HD95_barriers_DL[i, b] = percentile_hausdorff_distance(barrier.getMeshpoints(), barrier_DL.getMeshpoints(), percentile=95, voxel_spacing=voxel_size)

    # Create 2D plots

    if visualization:

        # voxel display
        COM = np.array(com(gtv))
        X_coord = int(COM[0])
        Y_coord = int(COM[1])
        Z_coord = int(COM[2])

        # prepare figures
        #CT.reduceGrid_mask(~bs)
        plotCT = CT.imageArray

        x, y, z = CT.getMeshGridAxes()

        plotGTV = gtv.astype(float).copy()
        plotGTV[~gtv] = np.NaN

        plotComp = (~bs).astype(float).copy()
        #plotComp[bs] = np.NaN

        plotCompDL = (~bsDL).astype(float).copy()
        plotCompDL[bsDL] = np.NaN

        # Build interactive plot

        def plot_isodistance_sagittal(ax, X, plotCTV, plotCTV_DL, plotCTV_TS):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[X, :, :].transpose(), axis=0), extent=[y[0], y[-1], z[0], z[-1]], cmap='gray', vmin=-1000, vmax=600)
            plt.contour(plotComp[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='blue', alpha=0.5)
            plt.contourf(plotGTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='yellow', alpha=0.25)
            plt.contour(plotCTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_DL[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')
            plt.contour(plotCTV_TS[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='orange', linewidths=1.5, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_isodistance_coronal(ax, Y, plotCTV, plotCTV_DL, plotCTV_TS):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[:, Y, :].transpose(), axis=0), extent=[x[0], x[-1], z[0], z[-1]], cmap='gray', vmin=-1000, vmax=600)
            plt.contour(plotComp[:, Y, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='blue', alpha=0.5)
            plt.contourf(plotGTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='yellow', alpha=0.5)
            plt.contour(plotCTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_DL[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='solid')
            plt.contour(plotCTV_TS[:, Y, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='orange', linewidths=1.5, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_isodistance_axial(ax, Z, plotCTV, plotCTV_DL, plotCTV_TS):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[:, :, Z].transpose(), axis=0), extent=[x[0], x[-1], y[0], y[-1]], cmap='gray', vmin=-1000, vmax=600)
            plt.contour(plotComp[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='blue') #, alpha=0.5)
            plt.contourf(plotGTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='yellow', alpha=0.25)
            plt.contour(plotCTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_DL[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', linewidths=1.5, linestyles='dashed')
            plt.contour(plotCTV_TS[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='orange', linewidths=1.5, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        # Plot
        fig = plt.figure()
        plt.axis('off')
        # plt.title('Interactive slider')

        ax1 = fig.add_subplot(131)
        plot_isodistance_sagittal(ax1, X_coord, ctv.imageArray, ctv_DL.imageArray, ctv_TS.imageArray)
        ax2 = fig.add_subplot(132)
        plot_isodistance_coronal(ax2, Y_coord, ctv.imageArray, ctv_DL.imageArray, ctv_TS.imageArray)
        ax3 = fig.add_subplot(133)
        plot_isodistance_axial(ax3, Z_coord, ctv.imageArray, ctv_DL.imageArray, ctv_TS.imageArray)

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

        # Make horizontal oriented slider to control the distance
        alpha_axis_4 = plt.axes([0.3, 0.06, 0.4, 0.03])
        alpha_slider_4 = Slider(
            ax=alpha_axis_4,
            label='Margin (mm)',
            valmin=0,
            valmax=30,
            valinit=margin,
            valstep=1
        )

        def update1(val):

            alpha1 = int(alpha_slider_1.val)
            alpha2 = int(alpha_slider_2.val)
            alpha3 = int(alpha_slider_3.val)
            alpha4 = int(alpha_slider_4.val)

            #ctv.setCTV_isodistance(alpha4)
            #ctv_DL.setCTV_isodistance(alpha4)
            #ctv_TS.setCTV_isodistance(alpha4)

            ax1.cla()
            plot_isodistance_sagittal(ax1, alpha1, ctv.imageArray, ctv_DL.imageArray, ctv_TS.imageArray)
            ax2.cla()
            plot_isodistance_coronal(ax2, alpha2, ctv.imageArray, ctv_DL.imageArray, ctv_TS.imageArray)
            ax3.cla()
            plot_isodistance_axial(ax3, alpha3, ctv.imageArray, ctv_DL.imageArray, ctv_TS.imageArray)

            plt.draw()

        alpha_slider_1.on_changed(update1)
        alpha_slider_2.on_changed(update1)
        alpha_slider_3.on_changed(update1)
        alpha_slider_4.on_changed(update1)

        plt.show()

# calculate statistics
dice_ctv_mean, dice_ctv_std = np.mean(dice_ctv_DL, axis=0), np.std(dice_ctv_DL, axis=0)
SDS_ctv_mean, SDS_ctv_std= np.mean(SDS_ctv_DL, axis=0), np.std(SDS_ctv_DL, axis=0)
HD95_ctv_mean, HD95_ctv_std = np.mean(HD95_ctv_DL, axis=0), np.std(HD95_ctv_DL, axis=0)

for margin, m in zip(margins, range(len(margins))):

    print(f"{margin} mm CTV DL mean dice: {dice_ctv_mean[m]*100:.2f}% +/- {dice_ctv_std[m]*100:.2f}%")

    for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
        print(f"{margin} mm CTV DL mean surface dice (\u03C4 = {tau} mm): {SDS_ctv_mean[m, j]*100:.2f}% +/- {SDS_ctv_std[m, j]*100:.2f}%")

    print(f"{margin} mm CTV DL mean HD95 : {HD95_ctv_mean[m]:.2f} mm +/- {HD95_ctv_std[m]:.2f} mm")

print(f"     -------     ")

# calculate statistics
dice_ctv_mean, dice_ctv_std = np.mean(dice_ctv_TS, axis=0), np.std(dice_ctv_TS, axis=0)
SDS_ctv_mean, SDS_ctv_std= np.mean(SDS_ctv_TS, axis=0), np.std(SDS_ctv_TS, axis=0)
HD95_ctv_mean, HD95_ctv_std = np.mean(HD95_ctv_TS, axis=0), np.std(HD95_ctv_TS, axis=0)

for margin, m in zip(margins, range(len(margins))):

    print(f"{margin} mm CTV TS mean dice: {dice_ctv_mean[m]*100:.2f}% +/- {dice_ctv_std[m]*100:.2f}%")

    for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
        print(f"{margin} mm CTV TS mean surface dice (\u03C4 = {tau} mm): {SDS_ctv_mean[m, j]*100:.2f}% +/- {SDS_ctv_std[m, j]*100:.2f}%")

    print(f"{margin} mm CTV TS mean HD95 : {HD95_ctv_mean[m]:.2f} mm +/- {HD95_ctv_std[m]:.2f} mm")

# calculate statistics
dice_barriers_DL_mean, dice_barriers_DL_std = np.mean(dice_barriers_DL, axis=0), np.std(dice_barriers_DL, axis=0)
SDS_barriers_DL_mean, SDS_barriers_DL_std= np.mean(SDS_barriers_DL, axis=0), np.std(SDS_barriers_DL, axis=0)
HD95_barriers_DL_mean, HD95_barriers_DL_std = np.mean(HD95_barriers_DL, axis=0), np.std(HD95_barriers_DL, axis=0)

for name, b in zip(barrier_names_eval, range(len(barrier_names_eval))):

    print(f"{name} mean dice: {dice_barriers_DL_mean[b]*100:.2f}% +/- {dice_barriers_DL_std[b]*100:.2f}%")

    for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
        print(f"{name} mean surface dice (\u03C4 = {tau} mm): {SDS_barriers_DL_mean[b, j]*100:.2f}% +/- {SDS_barriers_DL_std[b, j]*100:.2f}%")

    print(f"{name} mean HD95 : {HD95_barriers_DL_mean[b]:.2f} mm +/- {HD95_barriers_DL_std[b]:.2f} mm")


# plots

fig, axes = plt.subplots(nrows=len(margins), ncols=len(SDS_tolerance), figsize=(6*len(SDS_tolerance), 5*len(margins)))

# Ensure axes is always treated as a 2D array
if len(margins) == 1 and len(SDS_tolerance) == 1:
    axes = [[axes]]  # Convert single Axes object to a nested list
elif len(margins) == 1:
    axes = [axes]  # Convert 1D array to a 2D list (single row)
elif len(SDS_tolerance) == 1:
    axes = [[ax] for ax in axes]  # Convert 1D array to a 2D list (single column)

for m, margin in enumerate(margins):
    for j, tau in enumerate(SDS_tolerance):
        axes[m][j].hist(SDS_ctv_DL[:, m, j], edgecolor='black', alpha=0.7)
        axes[m][j].set_xlabel("SDS [%]")
        axes[m][j].set_ylabel("Frequency")
        axes[m][j].set_title(f"SDS (\u03C4 = {tau} mm) for {margin} mm CTV")
        axes[m][j].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# calculate statistics
dice_barriers_DL_mean, dice_barriers_DL_std = np.mean(dice_barriers_DL, axis=0), np.std(dice_barriers_DL, axis=0)
SDS_barriers_DL_mean, SDS_barriers_DL_std= np.mean(SDS_barriers_DL, axis=0), np.std(SDS_barriers_DL, axis=0)
HD95_barriers_DL_mean, HD95_barriers_DL_std = np.mean(HD95_barriers_DL, axis=0), np.std(HD95_barriers_DL, axis=0)

# save arrays
np.save(os.path.join(output_folder, 'dice_ctv_TS.npy'), dice_ctv_TS)
np.save(os.path.join(output_folder, 'SDS_ctv_TS.npy'), SDS_ctv_TS)
np.save(os.path.join(output_folder, 'HD95_ctv_TS.npy'), HD95_ctv_TS)
np.save(os.path.join(output_folder, 'dice_ctv_DL.npy'), dice_ctv_DL)
np.save(os.path.join(output_folder, 'SDS_ctv_DL.npy'), SDS_ctv_DL)
np.save(os.path.join(output_folder, 'HD95_ctv_DL.npy'), HD95_ctv_DL)
np.save(os.path.join(output_folder, 'dice_barriers_DL.npy'), dice_barriers_DL)
np.save(os.path.join(output_folder, 'SDS_barriers_DL.npy'), SDS_barriers_DL)
np.save(os.path.join(output_folder, 'HD95_barriers_DL.npy'), HD95_barriers_DL)