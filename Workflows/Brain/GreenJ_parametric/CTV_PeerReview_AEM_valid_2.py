#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:12:05 2024

@author: gregory
"""

import copy
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass as com
from matplotlib.widgets import Slider, Button
from dipy.io.image import load_nifti, save_nifti

from opentps.core.data.images._image3D import Image3D

from Process.CTVs import CTVGeometric
from Process import Struct
from Process.Analysis.contourComparison import dice_score, percentile_hausdorff_distance, hausdorff_distance, surface_dice_score, compute_bidirectional_distance_volume

interactive = False

# input directory
patient_dir = '/media/gregory/Elements/Data/MGH_Glioma/TVD_Processed'
input_dir = "/home/gregory/Downloads/results_folder_edits_AEM_train"

# Set the output directory
output_dir = "/home/gregory/Downloads/results_folder_edits_AEM_valid_2"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(input_dir, "PIDs_valid.txt"), "r") as f:
    PIDs_valid = [line.strip() for line in f]

with open(os.path.join(input_dir, "Crop_GTV_with_ventricles_valid.txt"), "r") as f:
    Crop_GTV_with_ventricles_valid = [line.strip() for line in f]

# nn-UNet parameters
model_segmentation = 'Dataset015_GLIS-RT'
config_segmentation = '3d_fullres'

# similarity metric for optimization
metric = 'dice'

# CTV model
model = {'model': 'Nonuniform',
         'obstacle': True,
         }

rules = {
    'postprocessing_barriers': True,
    'postprocessing_gtv': True,
    'threshold_review': 5  # [mm]
        }

# evaluation
SDS_tolerance = [1, 2, 3]  # [mm]

dice_PIDs = np.zeros((len(PIDs_valid), ))
HD95_PIDs = np.zeros((len(PIDs_valid), ))
HD_PIDs = np.zeros((len(PIDs_valid), ))
SDS_PIDs = np.zeros((len(PIDs_valid), len(SDS_tolerance)))
volume_diff_PIDs = np.zeros((len(PIDs_valid), ))
isodistance_PIDs = np.zeros((len(PIDs_valid), ))
barrier_movement_PIDs = np.zeros((len(PIDs_valid), 2))
resistance_PIDs = np.zeros((len(PIDs_valid), 2))
distance_barrier_movement = np.zeros((len(PIDs_valid), 2))
distance_barrier_soft = np.zeros((len(PIDs_valid), 2))
margin_reduction_soft = np.zeros((len(PIDs_valid), 2))

for PID, i, Crop_GTV in zip(PIDs_valid, range(len(PIDs_valid)), Crop_GTV_with_ventricles_valid):

    # input

    path_CT = os.path.join(patient_dir, f'{PID}/Therapy-scan/MRI_CT/CT.nii.gz')
    path_RTstructs = os.path.join(patient_dir, f'{PID}/Therapy-scan/Structures')
    path_RTstructs_DL = os.path.join(patient_dir, f'{PID}/Therapy-scan/Structures_{model_segmentation}_{config_segmentation}')

    # load data

    ct, static_grid2world, voxel_size = load_nifti(path_CT, return_voxsize=True)
    CT = Image3D(imageArray=ct, spacing=voxel_size)

    # load ground truth structures

    RTs = Struct()
    RTs.loadContours_folder(path_RTstructs, ['GTV', 'CTV'], contour_types=['GTV', None])

    gtv = RTs.getMaskByName('GTV').imageArray
    CTV = RTs.getMaskByName('CTV')

    # load DL model structures

    RTs.loadContours_folder(path_RTstructs_DL, ['Brain', 'Brainstem', 'Cerebellum', 'Chiasm','OpticNerve_L', 'OpticNerve_R', 'Midline', 'Ventricles_connected'], contour_types=[None, 'Barrier_soft', None, None, None, None, 'Barrier_moving', 'Barrier_moving'])

    optic_struct = (RTs.getMaskByName('OpticNerve_L').imageArray + RTs.getMaskByName('OpticNerve_R').imageArray + RTs.getMaskByName('Chiasm').imageArray) > 0
    RTs.setMask('Optic_structure', optic_struct, roi_type='Barrier_soft')

    brainstem = RTs.getMaskByName('Brainstem').imageArray
    cerebellum = RTs.getMaskByName('Cerebellum').imageArray
    midline = RTs.getMaskByName('Midline').imageArray
    brain = RTs.getMaskByName('Brain').imageArray
    ventricles = RTs.getMaskByName('Ventricles_connected').imageArray

    if rules['postprocessing_barriers']:

        # perform morphological operations (post-processing step)
        Brain_dilated = RTs.getMaskByName('Brain').copy()
        Brain_dilated.dilateMask(radius=(3, 3, 4))
        cerebellum = np.logical_and(cerebellum, ~Brain_dilated.imageArray)

    # define barrier structures

    bs = np.logical_or((brain + brainstem + cerebellum + optic_struct) == 0, midline + ventricles)
    RTs.setMask('BS', bs, spacing=voxel_size, roi_type='Barrier')

    #################
    #### GTV QA #####
    #################

    if rules['postprocessing_gtv']:

        # remove barrier voxels from the GTV
        gtv = np.logical_and(gtv, ~midline)

        if Crop_GTV:
            # remove vetricle voxels from the GTV
            gtv = np.logical_and(gtv, ~ventricles)

        # remove cerebellum voxels from the GTV
        gtv = np.logical_and(gtv, ~cerebellum)

        # only keep the same largest connected components as the original GTV
        RTs.setMask('GTV', gtv, spacing=voxel_size)
        count = RTs.getCountCC('GTV')
        gtv = RTs.getLargestCC('GTV', count)
        RTs.setMask('GTV', gtv)

    # define domain
    GTVBox = RTs.getBoundingBox('GTV', int(30))

    # Run fast marching method

    CTV_DL = CTVGeometric(rts=RTs, model=model, spacing=voxel_size)

    # fit isodistance
    CTV_DL.setCTV_metric_isodistance(CTV, metric=metric, x0=15, solver='FMM', method='Nelder-Mead', domain=GTVBox)

    # fit margin reduction (first soft then moving)
    CTV_DL.correctCTV_metric_softBarriers(CTV, metric=metric, method='Grid_search', num_grid_points=20)
    CTV_DL.correctCTV_metric_movingBarriers(CTV, metric=metric, method='Grid_search', num_grid_points=20)

    CTV_DL.smoothMask(size=2)

    BLD_CTV = compute_bidirectional_distance_volume(CTV_DL.getMeshpoints(), CTV.getMeshpoints(), CTV_DL.gridSize, voxel_spacing=voxel_size)

    # relevant surface distance threshold
    ROI_review = BLD_CTV >= rules['threshold_review']
    RTs.setMask('ROI_review', ROI_review, spacing=voxel_size)
    RTs.dilateMasks(['ROI_review'], 2)
    ROI = RTs.getMaskByName('ROI_review')

    isodistance_PIDs[i] = CTV_DL.isodistance
    for j in range(len(CTV_DL.moving['movements'])):
        barrier_movement_PIDs[i, j] = CTV_DL.moving['movements'][j]
    for j in range(len(CTV_DL.preferred['resistances'])):
        resistance_PIDs[i, j] = CTV_DL.preferred['resistances'][j]

    # # get flow vectors and geodesics in ROI
    # flow3D, geodesics = CTV_DL.getGeodesicsROI(ROI)
    #
    # flowX = flow3D[0]
    # flowY = flow3D[1]
    # flowZ = flow3D[2]

    for j, mask in enumerate(CTV_DL.moving['masks']):
        RTs_tmp = copy.deepcopy(RTs)
        bs_tmp = bs.copy()
        bs_tmp = ~np.logical_or(~bs_tmp, mask.imageArray)
        RTs_tmp.setMask('BS', bs_tmp, spacing=voxel_size)
        CTV_tmp = CTVGeometric(rts=RTs_tmp, model=model, spacing=voxel_size)

        distance_barrier_movement[i, j] = CTV_tmp.distance3D[mask.imageArray].min()

    for j, mask in enumerate(CTV_DL.preferred['masks']):
        RTs_tmp = copy.deepcopy(RTs)
        bs_tmp = bs.copy()
        bs_tmp = ~np.logical_or(~bs_tmp, mask.imageArray)
        RTs_tmp.setMask('BS', bs_tmp, spacing=voxel_size)
        CTV_tmp = CTVGeometric(rts=RTs_tmp, model=model, spacing=voxel_size)

        distance_barrier_soft[i, j] = CTV_tmp.distance3D[mask.imageArray].min()

        overlap = np.logical_and(mask.imageArray, CTV_DL.imageArray)
        if np.any(overlap):
            margin_reduction_soft[i, j] = (CTV_DL.isodistance - CTV_tmp.distance3D[overlap].max())
        else:
            margin_reduction_soft[i, j] = np.nan

    volume_diff_PIDs[i] = (CTV.getVolume() - CTV_DL.getVolume()) / CTV.getVolume()

    dice_PIDs[i] = dice_score(CTV.imageArray, CTV_DL.imageArray)

    for j, tau in enumerate(SDS_tolerance):
        SDS_PIDs[i, j] = surface_dice_score(CTV.imageArray, CTV_DL.imageArray, tau, voxel_spacing=voxel_size)

    HD95_PIDs[i] = percentile_hausdorff_distance(CTV.getMeshpoints(), CTV_DL.getMeshpoints(), percentile=95)
    HD_PIDs[i] = hausdorff_distance(CTV.getMeshpoints(), CTV_DL.getMeshpoints())


    print(f"##################################")
    print(f"Patient -- {PID}")
    print(f"##################################")

    print(f"CTV dice: {dice_PIDs[i]*100:.2f}%")
    for j, tau in enumerate(SDS_tolerance):
        print(f"CTV surface dice (\u03C4 = {tau} mm): {SDS_PIDs[i, j]*100:.2f}%")
    print(f"CTV HD95: {HD95_PIDs[i]:.2f} mm")
    print(f"CTV HD: {HD_PIDs[i]:.2f} mm")

    print(f"##################################")
    print(f"##################################")
    print(f"##################################")

    # Construct the filename
    filename = os.path.join(output_dir, f"{PID}_results.txt")

    with open(filename, "w") as f:
        f.write(f"##################################\n")
        f.write(f"Patient -- {PID}\n")
        f.write(f"##################################\n\n")

        f.write(f"CTV dice: {dice_PIDs[i] * 100:.2f}%\n")
        for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
            f.write(f"CTV surface dice (\u03C4 = {tau} mm): {SDS_PIDs[i, j] * 100:.2f}%\n")
        f.write(f"CTV HD95: {HD95_PIDs[i]:.2f} mm\n")
        f.write(f"CTV HD: {HD_PIDs[i]:.2f} mm\n\n")

        f.write(f"Volume difference: {volume_diff_PIDs[i] * 100:.2f}%\n\n")

        f.write(f"##################################\n")
        f.write(f"##################################\n")
        f.write(f"##################################\n")

    # save CTV

    directory = os.path.join(os.path.join(os.path.join(patient_dir, PID), 'Therapy-scan'), 'Targets_' + model_segmentation + '_' + config_segmentation)
    if not os.path.exists(directory):
        os.makedirs(directory)

    name_CTV_DL = 'CTV_auto'
    save_nifti(os.path.join(directory, name_CTV_DL + '.nii.gz'), CTV_DL.imageArray.astype(float), static_grid2world)

    # Create 2D plots

    if interactive:

        GTV = RTs.getMaskByName('GTV')
        BS = RTs.getMaskByName('BS')
        gtv = GTV.imageArray

        # voxel display
        COM = np.array(com(gtv))
        X_coord = int(COM[0])
        Y_coord = int(COM[1])
        Z_coord = int(COM[2])

        x, y, z = CT.getMeshGridAxes()
        X, Y, Z = CT.getMeshGridPositions()

        # prepare figures
        plotCT = ct.copy()

        plotGTV = gtv.astype(float).copy()
        plotGTV[~gtv] = np.NaN

        plotGTV_DL = RTs.getMaskByName('GTV').imageArray.astype(float).copy()

        plotDistance = CTV_DL.distance3D.copy()
        plotDistance[bs] = np.NaN

        plotBS = (~bs).astype(float).copy()

        plotPS = RTs.getMaskByName('Brainstem').imageArray.astype(float).copy()
        plotPS[~RTs.getMaskByName('Brainstem').imageArray] = np.NaN

        # Build interactive plot
        def plot_isodistance_sagittal(ax, X, plotCTV, plotCTV_DL, plotROI=None):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[X, :, :].transpose(), axis=0), extent=[y[0], y[-1], z[0], z[-1]], cmap='gray', vmin=-1000, vmax=600)
            plt.contour(plotBS[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='blue', linewidths=1)
            plt.contourf(plotGTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='yellow', alpha=0.5)
            plt.contourf(plotPS[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='orange', alpha=0.5)

            if plotROI is not None:
                plt.contourf(plotROI[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='red', alpha=0.5)
                #quiver(np.flip(Y[X, :, :].transpose(), axis=0), np.flip(Z[X, :, :].transpose(), axis=0), np.flip(flowY[X, :, :].transpose(), axis=0), np.flip(flowZ[X, :, :].transpose(), axis=0),  subsampling=(2, 2), scale=50, scale_units="width")

            plt.contour(plotCTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_DL[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_isodistance_coronal(ax, Y, plotCTV, plotCTV_DL, plotROI=None):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[:, Y, :].transpose(), axis=0), extent=[x[0], x[-1], z[0], z[-1]], cmap='gray', vmin=-1000, vmax=600)
            plt.contour(plotBS[:, Y, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='blue', linewidths=1)
            plt.contourf(plotGTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='yellow', alpha=0.5)
            plt.contourf(plotPS[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='orange', alpha=0.5)

            if plotROI is not None:
                plt.contourf(plotROI[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='red', alpha=0.5)
                #quiver(np.flip(X[:, Y, :].transpose(), axis=0), np.flip(Z[:, Y, :].transpose(), axis=0), np.flip(flowX[:, Y, :].transpose(), axis=0), np.flip(flowZ[:, Y, :].transpose(), axis=0), subsampling=(2, 2), scale=50, scale_units="width")

            plt.contour(plotCTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_DL[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_isodistance_axial(ax, Z, plotCTV, plotCTV_DL, plotROI=None):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[:, :, Z].transpose(), axis=0), extent=[x[0], x[-1], y[0], y[-1]], cmap='gray', vmin=-1000, vmax=600)

            plt.contour(plotBS[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='blue', linewidths=1)
            plt.contourf(plotGTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='yellow', alpha=0.5)
            plt.contourf(plotPS[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='orange', alpha=0.5)

            if plotROI is not None:
                plt.contourf(plotROI[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', alpha=0.5)
                #quiver(np.flip(X[:, :, Z].transpose(), axis=0), np.flip(Y[:, :, Z].transpose(), axis=0), np.flip(flowX[:, :, Z].transpose(), axis=0), np.flip(flowY[:, :, Z].transpose(), axis=0), subsampling=(4, 4), scale=50, scale_units="width")

            plt.contour(plotCTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_DL[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', linewidths=1.5, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        # Plot
        fig = plt.figure()
        plt.axis('off')
        # plt.title('Interactive slider')

        ax1 = fig.add_subplot(131)
        plot_isodistance_sagittal(ax1, X_coord, CTV.imageArray, CTV_DL.imageArray)
        ax2 = fig.add_subplot(132)
        plot_isodistance_coronal(ax2, Y_coord, CTV.imageArray, CTV_DL.imageArray)
        ax3 = fig.add_subplot(133)
        plot_isodistance_axial(ax3, Z_coord, CTV.imageArray, CTV_DL.imageArray)

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
        alpha_axis_4 = plt.axes([0.3, 0.08, 0.4, 0.03])
        alpha_slider_4 = Slider(
            ax=alpha_axis_4,
            label='Margin (mm)',
            valmin=0.0000,
            valmax=30.0000,
            valinit=CTV_DL.isodistance,
            valstep=1.0000
        )

        # Make horizontal oriented slider to control the distance
        alpha_axis_5 = plt.axes([0.3, 0.02, 0.4, 0.03])
        alpha_slider_5 = Slider(
            ax=alpha_axis_5,
            label='Threshold (mm)',
            valmin=0,
            valmax=10,
            valinit=rules['threshold_review'],
            valstep=1
        )

        alpha4_prev = None # initialize
        alpha5_prev = rules['threshold_review']
        plotROI = None
        def update(val):
            global alpha4_prev
            global alpha5_prev
            global plotROI

            alpha1 = int(alpha_slider_1.val)
            alpha2 = int(alpha_slider_2.val)
            alpha3 = int(alpha_slider_3.val)
            alpha4 = alpha_slider_4.val
            alpha5 = alpha_slider_5.val

            if plotROI is not None and alpha5 != alpha5_prev:

                # relevant surface distance threshold
                ROI_review = BLD_CTV >= alpha5

                RTs.setMask('ROI_review', ROI_review, spacing=voxel_size)
                ROI = RTs.getMaskByName('ROI_review')
                # dilate
                ROI.dilateMask(radius=2)

                plotROI = ROI.imageArray.astype(float).copy()
                plotROI[~ROI.imageArray] = np.NaN

            ax1.cla()
            plot_isodistance_sagittal(ax1, alpha1, CTV.imageArray, CTV_DL.imageArray, plotROI=plotROI)
            ax2.cla()
            plot_isodistance_coronal(ax2, alpha2, CTV.imageArray, CTV_DL.imageArray, plotROI=plotROI)
            ax3.cla()
            plot_isodistance_axial(ax3, alpha3, CTV.imageArray, CTV_DL.imageArray, plotROI=plotROI)

            plt.draw()

        alpha_slider_1.on_changed(update)
        alpha_slider_2.on_changed(update)
        alpha_slider_3.on_changed(update)
        alpha_slider_4.on_changed(update)
        alpha_slider_5.on_changed(update)

        # Create axes for reset button and create button
        resetax1 = plt.axes([0.8, 0.025, 0.1, 0.04])
        button1 = Button(resetax1, 'Reset', color='gold',
                        hovercolor='skyblue')

        resetax2 = plt.axes([0.8, 0.1, 0.1, 0.04])
        button2 = Button(resetax2, 'Review', color='silver',
                        hovercolor='skyblue')


        # Create a function resetSlider to set slider to
        # initial values when Reset button is clicked

        def resetSlider(event):
            alpha_slider_1.reset()
            alpha_slider_2.reset()
            alpha_slider_3.reset()

        # Call resetSlider function when clicked on reset button
        button1.on_clicked(resetSlider)

        def activatePeerReview(event):
            global plotROI
            plotROI = 0 # different from None

        button2.on_clicked(activatePeerReview)

        plt.show()


# Summary statistics

summary_file = os.path.join(output_dir, "summary_statistics.txt")
with open(summary_file, "w") as f:
    f.write("########## Summary Statistics ##########\n\n")

    # Dice
    f.write(f"CTV Dice Mean: {np.mean(dice_PIDs)*100:.2f}%\n")
    f.write(f"CTV Dice Std: {np.std(dice_PIDs)*100:.2f}%\n\n")

    # Surface Dice
    for j, tau in enumerate(SDS_tolerance):
        f.write(f"CTV Surface Dice (\u03C4 = {tau} mm) Mean: {np.mean(SDS_PIDs[:, j])*100:.2f}%\n")
        f.write(f"CTV Surface Dice (\u03C4 = {tau} mm) Std: {np.std(SDS_PIDs[:, j])*100:.2f}%\n\n")

    # HD and HD95
    f.write(f"CTV HD95 Mean: {np.mean(HD95_PIDs):.2f} mm\n")
    f.write(f"CTV HD95 Std: {np.std(HD95_PIDs):.2f} mm\n\n")
    f.write(f"CTV HD Mean: {np.mean(HD_PIDs):.2f} mm\n")
    f.write(f"CTV HD Std: {np.std(HD_PIDs):.2f} mm\n\n")

    # Volume Difference
    f.write(f"Volume Difference Mean: {np.mean(volume_diff_PIDs)*100:.2f}%\n")
    f.write(f"Volume Difference Std: {np.std(volume_diff_PIDs)*100:.2f}%\n\n")

# make plots
num_SDS = len(SDS_tolerance)
cols = max(2, num_SDS)  # Ensure at least 2 columns
rows = 2  # Two rows for better layout

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6 * cols, 5 * rows))

# Flatten axes array for easier iteration
axes = axes.flatten()

i = 0
axes[i].hist(dice_PIDs, edgecolor='black', alpha=0.7)
axes[i].set_xlabel("Dice")
axes[i].set_ylabel("Frequency")
axes[i].set_title("Histogram of Dice Scores")
axes[i].grid(axis='y', linestyle='--', alpha=0.7)

i += 1
axes[i].hist(HD95_PIDs, edgecolor='black', alpha=0.7)
axes[i].set_xlabel("HD95")
axes[i].set_ylabel("Frequency")
axes[i].set_title("Histogram of HD95 Values")
axes[i].grid(axis='y', linestyle='--', alpha=0.7)

i += 1
for j, tau in enumerate(SDS_tolerance):
    axes[i].hist(SDS_PIDs[:, j], edgecolor='black', alpha=0.7)
    axes[i].set_xlabel(f"SDS (τ = {tau} mm)")
    axes[i].set_ylabel("Frequency")
    axes[i].set_title(f"Histogram of SDS (τ = {tau} mm)")
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    i += 1

# Hide any unused subplots
for ax in axes[i:]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()

# calculate statistics

dice_PIDs_mean, dice_PIDs_std = np.mean(dice_PIDs), np.std(dice_PIDs)
SDS_PIDs_mean, SDS_PIDs_std= np.mean(SDS_PIDs, axis=0), np.std(SDS_PIDs, axis=0)
HD95_PIDs_mean, HD95_PIDs_std = np.mean(HD95_PIDs), np.std(HD95_PIDs)
HD_PIDs_mean, HD_PIDs_std = np.mean(HD_PIDs), np.std(HD_PIDs)

print(f"CTV DL mean dice: {dice_PIDs_mean*100:.2f}% +/- {dice_PIDs_std*100:.2f}%")
for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
    print(f"CTV DL mean surface dice (\u03C4 = {tau} mm): {SDS_PIDs_mean[j]*100:.2f}% +/- {SDS_PIDs_std[j]*100:.2f}%")
print(f"CTV DL mean HD95 : {HD95_PIDs_mean:.2f} mm +/- {HD95_PIDs_std:.2f} mm")
print(f"CTV DL mean HD: {HD_PIDs_mean:.2f} mm +/- {HD_PIDs_std:.2f} mm")


# save numpy arrays
np.save(os.path.join(output_dir, "volume_diff_PIDs.npy"), volume_diff_PIDs)
np.save(os.path.join(output_dir, "isodistance_PIDs.npy"), isodistance_PIDs)
np.save(os.path.join(output_dir, "barrier_movement_PIDs.npy"), barrier_movement_PIDs)
np.save(os.path.join(output_dir, "resistance_PIDs.npy"), resistance_PIDs)
np.save(os.path.join(output_dir, "distance_barrier_movement.npy"), distance_barrier_movement)
np.save(os.path.join(output_dir, "distance_barrier_soft.npy"), distance_barrier_soft)
np.save(os.path.join(output_dir, "margin_reduction_soft.npy"), margin_reduction_soft)

np.save(os.path.join(output_dir, "dice_PIDs.npy"), dice_PIDs)
np.save(os.path.join(output_dir, "SDS_PIDs.npy"), SDS_PIDs)
np.save(os.path.join(output_dir, "HD95_PIDs.npy"), HD95_PIDs)

