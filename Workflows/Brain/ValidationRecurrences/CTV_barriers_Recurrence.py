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

from Process import CTVGeometric
from Process import Struct
from Process import dice_score, percentile_hausdorff_distance, hausdorff_distance, surface_dice_score

visualization = True
postprocessing_barriers = False
postprocessing_gtv = False

# patient IDs
patient_dir = '/media/gregory/Elements/Data/MGH_Glioma/TVD_Processed'
PIDs = ['TVD_006_GBM', 'TVD_007_AC', 'TVD_010_AAC', 'TVD_014_GBM']

barrier_names = ['Brain', 'Brainstem', 'Cerebellum', 'Chiasm', 'OpticNerve_L', 'OpticNerve_R', 'Midline', 'Ventricles_connected']

# nn-UNet parameters
model_segmentation = 'Dataset013_GLIS-RT_Regions'
config_segmentation = '3d_fullres'

# evaluation metrics
SDS_tolerance = [1, 2, 3]  # [mm]

dice_ctv_DL = np.zeros((len(PIDs), ))
HD95_ctv_DL = np.zeros((len(PIDs), ))
HD_ctv_DL = np.zeros((len(PIDs), ))
SDS_ctv_DL = np.zeros((len(PIDs), len(SDS_tolerance)))

for PID, i in zip(PIDs, range(len(PIDs))):

    # input

    path_CT = os.path.join(patient_dir, f'{PID}/Therapy-scan/MRI_CT/CT.nii.gz')
    path_RTstructs = os.path.join(patient_dir, f'{PID}/Therapy-scan/Structures')
    path_RTstructs_DL = os.path.join(patient_dir, f'{PID}/Therapy-scan/Structures_{model_segmentation}_{config_segmentation}')
    path_RTstructs_retreat = os.path.join(patient_dir, f'{PID}/Therapy-scan-retreatment/Structures')

    # load data

    ct, static_grid2world, voxel_size = load_nifti(path_CT, return_voxsize=True)
    CT = Image3D(imageArray=ct, spacing=voxel_size)

    # load ground truth structures

    RTs = Struct()
    RTs.loadContours_folder(path_RTstructs, ['GTV', 'CTV', 'PTV'])

    CTV = RTs.getMaskByName('CTV')
    gtv = RTs.getMaskByName('GTV').imageArray

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

    RTs_retreat = Struct()
    RTs_retreat.loadContours_folder(path_RTstructs_retreat, ['GTV', 'PTV'])

    gtv_retreat = RTs_retreat.getMaskByName('GTV').imageArray

    #################
    #################
    #################

    if postprocessing_barriers:
        # perform morphological operations (post-processing step)

        # Brainstem_dilated = RTs.getMaskByName('Brainstem').copy()
        # Brainstem_dilated.dilateMask(radius=(2.0, 2.0, 0.0))

        Brain_dilated = RTs.getMaskByName('Brain').copy()
        Brain_dilated.dilateMask(radius=(2 * voxel_size[0], 2 * voxel_size[1], 2 * voxel_size[2]))

        # Cerebellum_dilated = RTs.getMaskByName('Cerebellum').copy()
        # Cerebellum_dilated.dilateMask(radius=(2.0, 2.0, 2.0))

        # Cerebellum_eroded = RTs.getMaskByName('Cerebellum').copy()
        # Cerebellum_eroded.erodeMask(radius=(2.0, 2.0, 2.0))

        # Brain / Brainstem connection
        # brain = np.logical_and(brain, ~Brainstem_dilated.imageArray)

        # Brain / Cerebellum connection
        # brain = np.logical_and(brain, ~Cerebellum_dilated.imageArray)

        cerebellum = np.logical_and(cerebellum, ~Brain_dilated.imageArray)

        # Brainstem / Cerebellum connection
        # V1 = np.logical_and(cerebellum, ~Brainstem_dilated.imageArray)
        # V2 = np.logical_or(Cerebellum_eroded.imageArray, )

    #################
    #################
    #################

    # define barrier structures

    bsDL = np.logical_or((brainDL + brainstemDL + cerebellumDL + chiasmDL + optic_nervesDL) == 0, (midlineDL + ventriclesDL) > 0)
    RTs_DL.setMask('BS', bsDL, voxel_size)

    # set preferred spread structure
    RTs_DL.setMask('PS', ~bsDL, voxel_size)

    #################
    #### GTV QA #####
    #################

    if postprocessing_gtv:
        # remove barrier voxels from the GTV
        gtvDL = np.logical_and(gtv, ~bsDL)
        RTs_DL.setMask('GTV', gtvDL, voxel_size)

        # only keep largest connected component
        gtvDL = RTs_DL.getLargestCC('GTV')
        RTs_DL.setMask('GTV', gtvDL)

    else:
        RTs_DL.setMask('GTV', gtv, voxel_size)

    # remove predicted barriers from GTV
    gtv_check = np.logical_and(gtv, ~bsDL)
    RTs_DL.setMask('GTV_check', gtv_check, voxel_size)

    num_components = RTs_DL.getCountCC('GTV_check')
    if num_components > 1:
        print("Warning: More than one connected component detected for the GTV.")

    # Run fast marching method

    ctv_DL = CTVGeometric(rts=RTs_DL, model={'model': 'Nonuniform', 'obstacle': True, 'resistance': 1}, spacing=voxel_size)

    # define calculation domain
    GTVBox = RTs.getBoundingBox('GTV', 30)

    metric = 'dice'

    # fit isodistance
    ctv_DL.setCTV_metric(CTV, metric=metric, x0=10, solver='FMM', method='Nelder-Mead', domain=GTVBox)

    margin = ctv_DL.isodistance

    # smoothing to remove holes etc.
    #ctv_DL.smoothMask(size=2)

    # compute and store metrics for CTVs (remove gtv from volume overlap evaluation)
    dice_ctv_DL[i] = dice_score(np.logical_and(CTV.imageArray, ~gtv), np.logical_and(ctv_DL.imageArray, ~gtv))

    for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
        SDS_ctv_DL[i, j] = surface_dice_score(CTV.imageArray, ctv_DL.imageArray, tau, voxel_spacing=voxel_size)

    HD95_ctv_DL[i] = percentile_hausdorff_distance(CTV.getMeshpoints(), ctv_DL.getMeshpoints(), percentile=95)
    HD_ctv_DL[i] = hausdorff_distance(CTV.getMeshpoints(), ctv_DL.getMeshpoints())

    print(f"patient -- {PID}")

    print(f"{margin} mm CTV DL dice: {dice_ctv_DL[i]*100:.2f}%")
    for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
        print(f"{margin} mm CTV DL surface dice (\u03C4 = {tau} mm): {SDS_ctv_DL[i, j]*100:.2f}%")
    print(f"{margin} mm CTV DL HD95: {HD95_ctv_DL[i]:.2f} mm")
    print(f"{margin} mm CTV DL HD: {HD_ctv_DL[i]:.2f} mm")

    print(f"     -------     ")
    print(f"     -------     ")

    # Create 2D plots

    if visualization:

        gtv = RTs.getMaskByName('GTV').imageArray

        # voxel display
        COM = np.array(com(gtv))
        X_coord = int(COM[0])
        Y_coord = int(COM[1])
        Z_coord = int(COM[2])

        # prepare figures
        #CT.reduceGrid_mask(~bs)
        plotCT = CT.imageArray

        x, y, z = CT.getMeshGridAxes()

        plotCTV = CTV.imageArray.astype(float).copy()

        plotCTV_DL = ctv_DL.imageArray.astype(float).copy()

        plotGTV = gtv.astype(float).copy()
        plotGTV[~gtv] = np.NaN

        plotGTV_retreat = gtv_retreat.astype(float).copy()
        plotGTV_retreat[~gtv_retreat] = np.NaN

        plotCompDL = (~bsDL).astype(float).copy()
        plotCompDL[bsDL] = np.NaN
        plotCompDL[gtv] = np.NaN


        # Build interactive plot

        def plot_isodistance_sagittal(ax, X):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[X, :, :].transpose(), axis=0), extent=[y[0], y[-1], z[0], z[-1]], cmap='gray', vmin=-1000, vmax=600)
            plt.contourf(plotCompDL[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='blue', alpha=0.25)
            plt.contourf(plotGTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='yellow', alpha=0.5)
            plt.contourf(plotGTV_retreat[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='red', alpha=0.5)
            plt.contour(plotCTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_DL[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_isodistance_coronal(ax, Y):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[:, Y, :].transpose(), axis=0), extent=[x[0], x[-1], z[0], z[-1]], cmap='gray', vmin=-1000, vmax=600)
            plt.contourf(plotCompDL[:, Y, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='blue', alpha=0.25)
            plt.contourf(plotGTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='yellow', alpha=0.5)
            plt.contourf(plotGTV_retreat[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='red', alpha=0.5)
            plt.contour(plotCTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_DL[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_isodistance_axial(ax, Z):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[:, :, Z].transpose(), axis=0), extent=[x[0], x[-1], y[0], y[-1]], cmap='gray', vmin=-1000, vmax=600)
            plt.contourf(plotCompDL[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='blue', alpha=0.25)
            plt.contourf(plotGTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='yellow', alpha=0.5)
            plt.contourf(plotGTV_retreat[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', alpha=0.5)
            plt.contour(plotCTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_DL[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', linewidths=1.5, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        # Plot
        fig = plt.figure()
        plt.axis('off')
        # plt.title('Interactive slider')

        ax1 = fig.add_subplot(131)
        plot_isodistance_sagittal(ax1, X_coord)
        ax2 = fig.add_subplot(132)
        plot_isodistance_coronal(ax2, Y_coord)
        ax3 = fig.add_subplot(133)
        plot_isodistance_axial(ax3, Z_coord)

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
            plot_isodistance_sagittal(ax1, alpha1)
            ax2.cla()
            plot_isodistance_coronal(ax2, alpha2)
            ax3.cla()
            plot_isodistance_axial(ax3, alpha3)

            plt.draw()

        alpha_slider_1.on_changed(update1)
        alpha_slider_2.on_changed(update1)
        alpha_slider_3.on_changed(update1)

        plt.show()
