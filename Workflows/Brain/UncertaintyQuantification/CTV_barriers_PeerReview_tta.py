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
from matplotlib.widgets import Slider, Button
from dipy.io.image import load_nifti
from scipy.stats import pearsonr
import itertools

from opentps.core.data.images._image3D import Image3D

from Process import CTVGeometric
from Process import Struct
from Process import dice_score, percentile_hausdorff_distance, hausdorff_distance, surface_dice_score, compute_bidirectional_distance_volume

visualization = False
interactive = False

postprocessing_barriers = False
preprocessing_gtv = False
postprocessing_gtv = False

# patient IDs
patient_dir = '/media/gregory/Elements/Data/MGH_Glioma/GLIS-RT_Processed'
PIDs = ['GLI_016_GBM', 'GLI_019_GBM', 'GLI_022_GBM', 'GLI_023_GBM', 'GLI_028_GBM', 'GLI_033_GBM', 'GLI_037_GBM', 'GLI_044', 'GLI_055_GBM', 'GLI_061_GBM', 'GLI_072_GBM', 'GLI_167_GBM', 'GLI_184_GBM']

# nn-UNet parameters
model_segmentation = 'Dataset013_GLIS-RT_Regions'
config_segmentation = '3d_fullres'

# GTV-CTV margin according to clinical guidelines
margin_guideline = 15  # [mm]

# infiltration distance according to clinical guidelines
infiltration_guideline = 5  # [mm]

# resistance coefficient that matches the guidelines
resistance_guideline = 1.

# CTV model
model = {
    'obstacle': True,
    'obstacle_soft': True,
    'model': 'Nonuniform',  # None, 'Nonuniform', 'Anisotropic',
    'model-DTI': 'Rekik', # applies only in case of Anisotropic
    'resistance': resistance_guideline
}

augmentations = ['original', 'flip_lr', 'flip_ud', 'flip_ap', 'rotate_90', 'rotate_180', 'rotate_270', 'zoom_in', 'zoom_out', 'gaussian_noise', 'gaussian_blur', 'brightness', 'saturation', 'hue', 'elastic'] # 'contrast'

# evaluation
SDS_tolerance = [2]  # [mm]

dice_PIDs = np.zeros((len(PIDs)))
HD95_PIDs = np.zeros((len(PIDs)))
HD_PIDs = np.zeros((len(PIDs)))
SDS_PIDs = np.zeros((len(PIDs), len(SDS_tolerance)))
pairwise_dice_PIDs = np.zeros((len(PIDs)))
pairwise_SDS_PIDs = np.zeros((len(PIDs), len(SDS_tolerance)))
for PID, i in zip(PIDs, range(len(PIDs))):

    # input

    path_CT = os.path.join(patient_dir, f'{PID}/Therapy-scan/MRI_CT/CT.nii.gz')
    path_RTstructs = os.path.join(patient_dir, f'{PID}/Therapy-scan/Structures')
    path_RTstructs_DL = os.path.join(patient_dir, f'{PID}/Therapy-scan/Structures_{model_segmentation}_{config_segmentation}')
    path_RTstructs_DL_TTA = os.path.join(patient_dir, f'{PID}/Therapy-scan/Structures_{model_segmentation}_{config_segmentation}/tta')

    # load data

    ct, static_grid2world, voxel_size = load_nifti(path_CT, return_voxsize=True)
    CT = Image3D(imageArray=ct, spacing=voxel_size)

    # load DL model structures

    CTVs_tta = []
    for augmentation, a in zip(augmentations, range(len(augmentations))):

        RTs_DL = Struct()
        RTs_DL.loadContours_folder(path_RTstructs, ['GTV'])

        gtv = RTs_DL.getMaskByName('GTV').imageArray

        RTs_DL.loadContours_folder(path_RTstructs_DL_TTA,
                                [f'Brain_{augmentation}', f'Brainstem_{augmentation}', f'Cerebellum_{augmentation}', f'Chiasm_{augmentation}',f'OpticNerve_L_{augmentation}', f'OpticNerve_R_{augmentation}', f'Midline_{augmentation}', f'Ventricles_connected_{augmentation}'],
                                ['Brain', 'Brainstem', 'Cerebellum', 'Chiasm','OpticNerve_L', 'OpticNerve_R', 'Midline', 'Ventricles_connected'])

        chiasm = RTs_DL.getMaskByName('Chiasm').imageArray
        optic_nerves = np.logical_or(RTs_DL.getMaskByName('OpticNerve_L').imageArray,  RTs_DL.getMaskByName('OpticNerve_R').imageArray)
        brainstem = RTs_DL.getMaskByName('Brainstem').imageArray
        cerebellum = RTs_DL.getMaskByName('Cerebellum').imageArray
        midline = RTs_DL.getMaskByName('Midline').imageArray
        brain = RTs_DL.getMaskByName('Brain').imageArray
        ventricles = RTs_DL.getMaskByName('Ventricles_connected').imageArray

        if preprocessing_gtv:
            # remove GTV overlap from ventricles
            overlap = np.logical_and(RTs_DL.getMaskByName('Ventricles_connected').imageArray, gtv)
            ventricles = np.logical_and(RTs_DL.getMaskByName('Ventricles_connected').imageArray, ~overlap)
            RTs_DL.setMask('Ventricles_connected', ventricles)
            # add GTV overlap to brain
            brain = np.logical_or(RTs_DL.getMaskByName('Brain').imageArray, overlap)
            RTs_DL.setMask('Brain', brain)

        if postprocessing_barriers:

            Brain_dilated = RTs_DL.getMaskByName('Brain').copy()
            Brain_dilated.dilateMask(radius=(2*voxel_size[0], 2*voxel_size[1], 2*voxel_size[2]))
            cerebellum = np.logical_and(cerebellum, ~Brain_dilated.imageArray)

        # define barrier structures

        # barrier (including soft barriers)
        bs = np.logical_or((brain + brainstem + cerebellum + chiasm + optic_nerves) == 0, midline + ventricles)
        RTs_DL.setMask('BS', bs, voxel_size)

        # set preferred spread structure
        RTs_DL.setMask('PS', ~bs, voxel_size)

        #################
        #### GTV QA #####
        #################

        if postprocessing_gtv:
            # remove barrier voxels from the GTV
            gtv = np.logical_and(gtv, ~bs)
            RTs_DL.setMask('GTV', gtv, voxel_size)

            # only keep largest connected component
            gtv = RTs_DL.getLargestCC('GTV')
            RTs_DL.setMask('GTV', gtv)

        else:
            RTs_DL.setMask('GTV', gtv, voxel_size)

        # define domain
        GTVBox = RTs_DL.getBoundingBox('GTV', int(margin_guideline + 10))

        # Run fast marching method
        CTV_DL = CTVGeometric(rts=RTs_DL, model=model, spacing=voxel_size)

        # fit isodistance
        CTV_DL.setCTV_isodistance(margin_guideline, solver='FMM', domain=GTVBox)

        # store CTV in list
        CTVs_tta.append(CTV_DL.imageArray)

    # Initialize RT structures
    RTs_DL = Struct()
    RTs_DL.loadContours_folder(path_RTstructs, ['GTV', 'CTV'])
    gtv = RTs_DL.getMaskByName('GTV').imageArray

    # Load additional contours
    RTs_DL.loadContours_folder(
        path_RTstructs_DL,
        ['Brain', 'Brainstem', 'Cerebellum', 'Chiasm', 'OpticNerve_L', 'OpticNerve_R', 'Midline',
         'Ventricles_connected']
    )

    # Extract masks
    chiasm = RTs_DL.getMaskByName('Chiasm').imageArray
    optic_nerves = np.logical_or(
        RTs_DL.getMaskByName('OpticNerve_L').imageArray,
        RTs_DL.getMaskByName('OpticNerve_R').imageArray
    )
    brainstem = RTs_DL.getMaskByName('Brainstem').imageArray
    cerebellum = RTs_DL.getMaskByName('Cerebellum').imageArray
    midline = RTs_DL.getMaskByName('Midline').imageArray
    brain = RTs_DL.getMaskByName('Brain').imageArray
    ventricles = RTs_DL.getMaskByName('Ventricles_connected').imageArray

    # Preprocessing GTV
    if preprocessing_gtv:
        overlap = np.logical_and(ventricles, gtv)
        ventricles = np.logical_and(ventricles, ~overlap)
        RTs_DL.setMask('Ventricles_connected', ventricles)

        brain = np.logical_or(brain, overlap)
        RTs_DL.setMask('Brain', brain)

    # Postprocessing barriers
    if postprocessing_barriers:
        Brain_dilated = RTs_DL.getMaskByName('Brain').copy()
        Brain_dilated.dilateMask(radius=(2 * voxel_size[0], 2 * voxel_size[1], 2 * voxel_size[2]))
        cerebellum = np.logical_and(cerebellum, ~Brain_dilated.imageArray)

    # Define barrier structures
    bs = np.logical_or((brain + brainstem + cerebellum + chiasm + optic_nerves) == 0, midline + ventricles)
    RTs_DL.setMask('BS', bs, voxel_size)
    RTs_DL.setMask('PS', ~bs, voxel_size)  # Preferred spread structure

    # GTV Quality Assurance
    if postprocessing_gtv:
        gtv = np.logical_and(gtv, ~bs)  # Remove barrier voxels from GTV
        RTs_DL.setMask('GTV', gtv, voxel_size)

        gtv = RTs_DL.getLargestCC('GTV')  # Keep largest connected component
        RTs_DL.setMask('GTV', gtv)
    else:
        RTs_DL.setMask('GTV', gtv, voxel_size)

    # Define domain and run fast marching
    GTVBox = RTs_DL.getBoundingBox('GTV', int(margin_guideline + 10))
    CTV_DL = CTVGeometric(rts=RTs_DL, model=model, spacing=voxel_size)
    CTV_DL.setCTV_isodistance(margin_guideline, solver='FMM', domain=GTVBox)

    # Repeat for RTs (separate instance)
    RTs = Struct()
    RTs.loadContours_folder(path_RTstructs, ['GTV', 'CTV'])
    gtv = RTs.getMaskByName('GTV').imageArray

    RTs.loadContours_folder(
        path_RTstructs,
        ['Brain', 'Brainstem', 'Cerebellum', 'Chiasm', 'OpticNerve_L', 'OpticNerve_R', 'Midline',
         'Ventricles_connected']
    )

    # Extract masks again
    chiasm = RTs.getMaskByName('Chiasm').imageArray
    optic_nerves = np.logical_or(
        RTs.getMaskByName('OpticNerve_L').imageArray,
        RTs.getMaskByName('OpticNerve_R').imageArray
    )
    brainstem = RTs.getMaskByName('Brainstem').imageArray
    cerebellum = RTs.getMaskByName('Cerebellum').imageArray
    midline = RTs.getMaskByName('Midline').imageArray
    brain = RTs.getMaskByName('Brain').imageArray
    ventricles = RTs.getMaskByName('Ventricles_connected').imageArray

    # Preprocessing GTV (RTs)
    if preprocessing_gtv:
        overlap = np.logical_and(ventricles, gtv)
        ventricles = np.logical_and(ventricles, ~overlap)
        RTs.setMask('Ventricles_connected', ventricles)

        brain = np.logical_or(brain, overlap)
        RTs.setMask('Brain', brain)

    # Postprocessing barriers (RTs)
    if postprocessing_barriers:
        Brain_dilated = RTs.getMaskByName('Brain').copy()
        Brain_dilated.dilateMask(radius=(2 * voxel_size[0], 2 * voxel_size[1], 2 * voxel_size[2]))
        cerebellum = np.logical_and(cerebellum, ~Brain_dilated.imageArray)

    # Define barrier structures (RTs)
    bs = np.logical_or((brain + brainstem + cerebellum + chiasm + optic_nerves) == 0, midline + ventricles)
    RTs.setMask('BS', bs, voxel_size)
    RTs.setMask('PS', ~bs, voxel_size)

    # GTV Quality Assurance (RTs)
    if postprocessing_gtv:
        gtv = np.logical_and(gtv, ~bs)
        RTs.setMask('GTV', gtv, voxel_size)

        gtv = RTs.getLargestCC('GTV')
        RTs.setMask('GTV', gtv)
    else:
        RTs.setMask('GTV', gtv, voxel_size)

    # Define domain and run fast marching (RTs)
    GTVBox = RTs.getBoundingBox('GTV', int(margin_guideline + 10))
    CTV = CTVGeometric(rts=RTs, model=model, spacing=voxel_size)
    CTV.setCTV_isodistance(margin_guideline, solver='FMM', domain=GTVBox)

    # compute and store parameters and evaluation metrics

    dice_PIDs[i] = dice_score(CTV.imageArray, CTV_DL.imageArray)

    for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
        SDS_PIDs[i, j] = surface_dice_score(CTV.imageArray, CTV_DL.imageArray, tau, voxel_spacing=voxel_size)

    HD95_PIDs[i] = percentile_hausdorff_distance(CTV.getMeshpoints(), CTV_DL.getMeshpoints(), percentile=95)
    HD_PIDs[i] = hausdorff_distance(CTV.getMeshpoints(), CTV_DL.getMeshpoints())

    for pair in itertools.combinations(CTVs_tta, 2):
        pairwise_dice_PIDs[i] += dice_score(pair[0], pair[1])

        for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
            pairwise_SDS_PIDs[i, j] += surface_dice_score(pair[0], pair[1], tau, voxel_spacing=voxel_size)

    pairwise_dice_PIDs[i] /= len(CTVs_tta)*(len(CTVs_tta) - 1)/2

    for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
        pairwise_SDS_PIDs[i, j] /= len(CTVs_tta)*(len(CTVs_tta) - 1)/2

    print(f"#####################")
    print(f"Patient -- {PID} ####")
    print(f"#####################")

    print(f"CTV dice: {dice_PIDs[i]*100:.2f}%")
    for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
        print(f"CTV surface dice (\u03C4 = {tau} mm): {SDS_PIDs[i, j]*100:.2f}%")
    print(f"CTV HD95: {HD95_PIDs[i]:.2f} mm")
    print(f"CTV HD: {HD_PIDs[i]:.2f} mm")

    print(f"#####################")
    print(f"#####################")
    print(f"#####################")

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

        #plotROI = ROI.imageArray.astype(float).copy()
        #plotROI[~ROI.imageArray] = np.NaN

        # Build interactive plot

        def plot_isodistance_sagittal(ax, X, plotCTV, plotCTV_DL, plotROI=None):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[X, :, :].transpose(), axis=0), extent=[y[0], y[-1], z[0], z[-1]], cmap='gray', vmin=-1000, vmax=600)
            plt.contour(plotBS[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='blue', linewidths=1)
            plt.contourf(plotGTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='yellow', alpha=0.5)

            plt.contour(plotCTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_DL[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_isodistance_coronal(ax, Y, plotCTV, plotCTV_DL, plotROI=None):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[:, Y, :].transpose(), axis=0), extent=[x[0], x[-1], z[0], z[-1]], cmap='gray', vmin=-1000, vmax=600)
            plt.contour(plotBS[:, Y, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='blue', linewidths=1)
            plt.contourf(plotGTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='yellow', alpha=0.5)

            plt.contour(plotCTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_DL[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_isodistance_axial(ax, Z, plotCTV, plotCTV_DL, plotROI=None):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[:, :, Z].transpose(), axis=0), extent=[x[0], x[-1], y[0], y[-1]], cmap='gray', vmin=-1000, vmax=600)

            plt.contour(plotBS[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='blue', linewidths=1)
            plt.contourf(plotGTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='yellow', alpha=0.5)

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
            valinit=CTV_DL.distance_reduction,
            valstep=1
        )

        alpha5_prev = CTV_DL.distance_reduction
        plotROI = None
        def update(val):
            global alpha5_prev
            global plotROI

            alpha1 = int(alpha_slider_1.val)
            alpha2 = int(alpha_slider_2.val)
            alpha3 = int(alpha_slider_3.val)

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

        # Create axes for reset button and create button
        resetax1 = plt.axes([0.8, 0.025, 0.1, 0.04])
        button1 = Button(resetax1, 'Reset', color='gold',
                        hovercolor='skyblue')


        # Create a function resetSlider to set slider to
        # initial values when Reset button is clicked

        def resetSlider(event):
            alpha_slider_1.reset()
            alpha_slider_2.reset()
            alpha_slider_3.reset()

        # Call resetSlider function when clicked on reset button
        button1.on_clicked(resetSlider)

        plt.show()

# Compute Pearson correlation coefficient
#corr_coef_dice, _ = pearsonr(pairwise_dice_PIDs, dice_PIDs)
#risk = 1 - dice_PIDs
#confidence = pairwise_dice_PIDs

corr_coef_dice, _ = pearsonr(pairwise_SDS_PIDs[:, 0], SDS_PIDs[:, 0])
risk = 1 - SDS_PIDs[:, 0]
confidence = pairwise_SDS_PIDs[:, 0]

# Create the scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(confidence, risk, color='blue', alpha=0.7)
plt.xlabel("Confidence")
plt.ylabel("Risk")
plt.title(f"Failure detection plot with Pearson Correlation: {corr_coef_dice:.2f}")

# Display the plot
plt.grid(True)
plt.show()

# Create 2D plots

num_SDS = len(SDS_tolerance)
cols = max(2, num_SDS)  # Ensure at least 2 columns
rows = 2  # Two rows for better layout

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6 * cols, 5 * rows))

# Flatten axes array for easier iteration
axes = axes.flatten()

k = 0
axes[k].hist(dice_PIDs, edgecolor='black', alpha=0.7)
axes[k].set_xlabel("Dice")
axes[k].set_ylabel("Frequency")
axes[k].set_title("Histogram of Dice Scores")
axes[k].grid(axis='y', linestyle='--', alpha=0.7)

k += 1
axes[k].hist(HD95_PIDs, edgecolor='black', alpha=0.7)
axes[k].set_xlabel("HD95")
axes[k].set_ylabel("Frequency")
axes[k].set_title("Histogram of HD95 Values")
axes[k].grid(axis='y', linestyle='--', alpha=0.7)

for j, tau in enumerate(SDS_tolerance):
    k += 1
    axes[k].hist(SDS_PIDs[:, j], edgecolor='black', alpha=0.7)
    axes[k].set_xlabel(f"SDS (τ = {tau} mm)")
    axes[k].set_ylabel("Frequency")
    axes[k].set_title(f"Histogram of SDS (τ = {tau} mm)")
    axes[k].grid(axis='y', linestyle='--', alpha=0.7)

k += 1
axes[k].hist(pairwise_dice_PIDs, edgecolor='black', alpha=0.7)
axes[k].set_xlabel("HD95")
axes[k].set_ylabel("Frequency")
axes[k].set_title("Histogram of Pairwise DSC")
axes[k].grid(axis='y', linestyle='--', alpha=0.7)

# Hide any unused subplots
for ax in axes[k:]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()