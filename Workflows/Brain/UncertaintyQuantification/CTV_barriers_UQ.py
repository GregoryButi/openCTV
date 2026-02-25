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
from dipy.io.image import load_nifti

from opentps.core.data.images._image3D import Image3D
from Process import CTVGeometric
from Process import Struct
from Process import dice_score, percentile_hausdorff_distance, hausdorff_distance, surface_dice_score

visualization = True
postprocessing_barriers = True
postprocessing_gtv = False

PIDs = ['GLI_016_GBM', 'GLI_019_GBM', 'GLI_022_GBM', 'GLI_023_GBM', 'GLI_028_GBM', 'GLI_044']
#PIDs = ['GLI_044']

models = ['Dataset004_GLIS-RT', 'Dataset005_GLIS-RT_Regions']
configs = ['3d_fullres'] # '3d_fullres', '3d_cascade_fullres', '2d', '3d_lowres']
barrier_names = ['Background', 'Brain', 'Brainstem', 'Cerebellum', 'Chiasm', 'OpticNerve_L', 'OpticNerve_R', 'Ventricles_connected']

margins = [15]  # [mm]
SDS_tolerance = [1, 2, 3]  # [mm]

def to_one_hot(labelmap, num_classes):
    """
    Convert a label map to one-hot encoding.

    Args:
        labelmap (numpy.ndarray): Label map of shape (X, Y, Z, M)
        num_classes (int): Number of classes

    Returns:
        numpy.ndarray: One-hot encoded label map of shape (X, Y, Z, M, C)
    """
    shape = (num_classes,) + labelmap.shape
    one_hot = np.zeros(shape, dtype=bool)
    for c in range(num_classes):
        one_hot[c, ...] = (labelmap == c)
    return one_hot

# start workflow
model = models[1]

dice_ctv = np.zeros((len(PIDs), len(margins)))
HD95_ctv = np.zeros((len(PIDs), len(margins)))
HD_ctv = np.zeros((len(PIDs), len(margins)))
SDS_ctv = np.zeros((len(PIDs), len(margins), len(SDS_tolerance)))
dice_bs = np.zeros((len(PIDs), len(margins)))
HD95_bs = np.zeros((len(PIDs), len(margins)))
HD_bs = np.zeros((len(PIDs), len(margins)))
SDS_bs = np.zeros((len(PIDs), len(margins), len(SDS_tolerance)))
for PID, i in zip(PIDs, range(len(PIDs))):

    # input

    path_CT = f'/media/gregory/Elements/Data/MGH_Glioma/Processed/{PID}/Therapy-scan/MRI_CT/CT.nii.gz'
    path_RTstructs = f'/media/gregory/Elements/Data/MGH_Glioma/Processed/{PID}/Therapy-scan/Structures'

    # load data

    ct, static_grid2world, voxel_size = load_nifti(path_CT, return_voxsize=True)
    CT = Image3D(imageArray=ct, spacing=voxel_size)

    barrier_pred = np.zeros(tuple(CT.gridSize) + (len(configs), ))
    barrier_prob = np.zeros((len(barrier_names), ) + tuple(CT.gridSize) + (len(configs), ))
    for config, j in zip(configs, range(len(configs))):

        # input
        path_Barrier_BinaryPred = f'/home/gregory/Documents/Projects/CTV_RO1/Models/nnUNet/nnUNet_results/{model}/nnUNetTrainer__nnUNetPlans__{config}/outputs/{PID}.nii.gz'
        #path_Barrier_SoftPred = f'/home/gregory/Documents/Projects/CTV_RO1/Models/nnUNet/nnUNet_results/{model}/nnUNetTrainer__nnUNetPlans__{config}/outputs/{PID}.npz'

        data, _, _ = load_nifti(path_Barrier_BinaryPred, return_voxsize=True)
        barrier_pred[..., j] = data
        #data = np.load(path_Barrier_SoftPred)
        #barrier_prob[..., j] = (data.f.probabilities).transpose((0, 3, 2, 1))

    # convert to one hot format
    one_hot_barrier_pred = to_one_hot(barrier_pred, len(barrier_names))

    brainPred = (barrier_pred[..., 0] == 1).astype(int)
    brainstemPred = (barrier_pred[..., 0] == 2).astype(int)
    cerebellumPred = (barrier_pred[..., 0] == 3).astype(int)
    ventriclesPred = (barrier_pred[..., 0] == 7).astype(int)

    # define prediction metrics across the configs
    #entr = entropy(barrier_prob)
    #var = variance(barrier_prob)
    ##mi = mutual_information(one_hot_barrier_pred)  # Measures epistemic (model) uncertainty by capturing disagreement

    #cerebellumEntropy = entr[3, ...]
    #cerebellumVar = var[3, ...]
    #cerebellumDisagreement = disagreement_score(one_hot_barrier_pred[3, ...], cerebellumPred)

    # load structures into rt struct object

    RTs = Struct()
    # load ground truth contours
    RTs.loadContours_folder(path_RTstructs, ['Brain', 'Brainstem', 'Cerebellum', 'Midline', 'Ventricles', 'GTV'])

    RTs_Pred = Struct()
    RTs_Pred.setMask('Brainstem', brainstemPred, voxel_size)
    RTs_Pred.setMask('Cerebellum', cerebellumPred, voxel_size)

    # load ground truth masks
    brain = RTs.getMaskByName('Brain').imageArray
    brainstem = RTs.getMaskByName('Brainstem').imageArray
    cerebellum = RTs.getMaskByName('Cerebellum').imageArray
    midline = RTs.getMaskByName('Midline').imageArray
    ventricles = RTs.getMaskByName('Ventricles').imageArray

    #################
    #################
    #################

    if postprocessing_barriers:

        # perform morphological operations (post-processing step)

        Brainstem_dilated = RTs.getMaskByName('Brainstem').copy()
        Brainstem_dilated.dilateMask(radius=(2.0, 2.0, 0.0))

        BrainstemPred_dilated = RTs_Pred.getMaskByName('Brainstem').copy()
        BrainstemPred_dilated.dilateMask(radius=(2.0, 2.0, 0.0))

        Cerebellum_dilated = RTs.getMaskByName('Cerebellum').copy()
        Cerebellum_dilated.dilateMask(radius=(2.0, 2.0, 2.0))

        CerebellumPred_dilated = RTs_Pred.getMaskByName('Cerebellum').copy()
        CerebellumPred_dilated.dilateMask(radius=(2.0, 2.0, 2.0))

        #Brain_eroded = RTs.getMaskByName('Brain').copy()
        #Brain_eroded.erodeMask(radius=(2.0, 2.0, 2.0))

        #BrainPred_eroded = RTs_Pred.getMaskByName('Brain').copy()
        #BrainPred_eroded.erodeMask(radius=(2.0, 2.0, 2.0))

        CerebellumPred_eroded = RTs_Pred.getMaskByName('Cerebellum').copy()
        CerebellumPred_eroded.erodeMask(radius=(2.0, 2.0, 2.0))

        # Brain / Brainstem connection
        brain = np.logical_and(brain, ~Brainstem_dilated.imageArray)
        brainPred = np.logical_and(brainPred, ~BrainstemPred_dilated.imageArray)

        # Brain / Cerebellum connection
        brain = np.logical_and(brain, ~Cerebellum_dilated.imageArray)
        brainPred = np.logical_and(brainPred, ~CerebellumPred_dilated.imageArray)

        # Brainstem / Cerebellum connection
        #V1 = np.logical_and(cerebellumPred, ~BrainstemPred_dilated.imageArray)
        #V2 = np.logical_or(CerebellumPred_eroded.imageArray, )

    #################
    #################
    #################

    # define barrier structures

    bs = np.logical_or((brain + brainstem + cerebellum) == 0, (midline + ventricles) > 0)
    RTs.setMask('BS', bs, voxel_size)

    bsPred = np.logical_or((brainPred + brainstemPred + cerebellumPred) == 0, ventriclesPred == 7)
    RTs_Pred.setMask('BS', bsPred, voxel_size)

    #################
    #### GTV QA #####
    #################

    # remove barrier voxels from the GTV
    gtv = np.logical_and(RTs.getMaskByName('GTV').imageArray, ~bs)
    RTs.setMask('GTV', gtv)

    # only keep largest connected component
    gtv = RTs.getLargestCC('GTV')
    RTs.setMask('GTV', gtv)

    if postprocessing_gtv:
        # remove barrier voxels from the GTV
        gtvPred = np.logical_and(gtv, ~bsPred)
        RTs_Pred.setMask('GTV', gtvPred, voxel_size)

        # only keep largest connected component
        gtvPred = RTs_Pred.getLargestCC('GTV')
        RTs_Pred.setMask('GTV', gtvPred)

    else:
        RTs_Pred.setMask('GTV', gtv, voxel_size)

    # remove predicted barriers from GTV
    gtv_check = np.logical_and(gtv, ~bsPred)
    RTs_Pred.setMask('GTV_check', gtv_check, voxel_size)

    num_components = RTs_Pred.getCountCC('GTV_check')
    if num_components > 1:
        print("Warning: More than one connected component detected for the GTV.")

    # Run fast marching method

    ctv = CTVGeometric()
    ctv_Pred = CTVGeometric()

    for margin, m in zip(margins, range(len(margins))):

        # define calculation domain
        GTVBox = RTs.getBoundingBox('GTV', margin + 5)

        ctv.setCTV_isodistance(margin, RTs, model={'model': None, 'obstacle': True}, domain=GTVBox)
        ctv_Pred.setCTV_isodistance(margin, RTs_Pred, model={'model': None, 'obstacle': True}, domain=GTVBox)

        # smoothing to remove holes etc.
        ctv.smoothMask()
        ctv_Pred.smoothMask()

        # compute and store metrics
        dice_ctv[i, m] = dice_score(ctv.imageArray, ctv_Pred.imageArray)

        for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
            SDS_ctv[i, m, j] = surface_dice_score(ctv.imageArray, ctv_Pred.imageArray, tau, voxel_spacing=voxel_size)

        HD95_ctv[i, m] = percentile_hausdorff_distance(ctv.getMeshpoints(), ctv_Pred.getMeshpoints(), percentile=95)
        HD_ctv[i, m] = hausdorff_distance(ctv.getMeshpoints(), ctv_Pred.getMeshpoints())

        dice_bs[i, m] = dice_score(~bs, ~bsPred)

        for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
            SDS_bs[i, m, j] = surface_dice_score(bs, bsPred, tau, voxel_spacing=voxel_size)
        HD95_bs[i, m] = percentile_hausdorff_distance(RTs.getMaskByName('BS').getMeshpoints(), RTs_Pred.getMaskByName('BS').getMeshpoints(), percentile=95)

        print(f"patient -- {PID}")

        print(f"{margin} mm CTV dice: {dice_ctv[i, m]*100:.2f}%")

        for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
            print(f"{margin} mm CTV surface dice (\u03C4 = {tau} mm): {SDS_ctv[i, m, j]*100:.2f}%")

        print(f"{margin} mm CTV HD95: {HD95_ctv[i, m]:.2f} mm")
        print(f"{margin} mm CTV HD: {HD_ctv[i, m]:.2f} mm")
        print(f"     -------     ")

    # Create 2D plots

    if visualization:

        gtv = RTs.getMaskByName('GTV').imageArray

        # voxel display
        COM = np.array(com(gtv))
        X_coord = int(COM[0])
        Y_coord = int(COM[1])
        Z_coord = int(COM[2])

        x, y, z = CT.getMeshGridAxes()

        # prepare figures
        plotCT = ct.copy()

        plotGTV = gtv.astype(float).copy()
        plotGTV[~gtv] = np.NaN

        plotGTV_Pred = RTs_Pred.getMaskByName('GTV').imageArray.astype(float).copy()

        plotComp = (~bs).astype(float).copy()
        plotComp[bs] = np.NaN
        plotComp[gtv] = np.NaN

        plotPred = (~bsPred).astype(float).copy()

        plotBrain = brain.astype(float).copy()
        plotBrainstem = brainstem.astype(float).copy()
        plotCerebellum = cerebellum.astype(float).copy()

        #plotCerebellumEntropy = cerebellumEntropy.copy()
        #plotCerebellumVar = cerebellumVar.copy()
        #plotCerebellumDisagreement = cerebellumDisagreement.copy()

        # Build interactive plot

        def plot_isodistance_sagittal(ax, X, plotCTV, plotCTV_Pred):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[X, :, :].transpose(), axis=0), extent=[y[0], y[-1], z[0], z[-1]], cmap='gray') #, vmin=0, vmax=2.5)
            plt.contourf(plotComp[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='blue', alpha=0.2)
            plt.contour(plotPred[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='orange', linewidths=1, linestyles='dashed')
            plt.contourf(plotGTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='yellow', alpha=0.2)
            plt.contour(plotGTV_Pred[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='yellow', linewidths=1.5, linestyles='dashed')
            plt.contour(plotCTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_Pred[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_isodistance_coronal(ax, Y, plotCTV, plotCTV_Pred):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[:, Y, :].transpose(), axis=0), extent=[x[0], x[-1], z[0], z[-1]], cmap='gray') #, vmin=0, vmax=2.5)
            plt.contourf(plotComp[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='blue', alpha=0.2)
            plt.contour(plotPred[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='orange', linewidths=1, linestyles='dashed')
            plt.contourf(plotGTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='yellow', alpha=0.2)
            plt.contour(plotGTV_Pred[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='yellow', linewidths=1.5, linestyles='dashed')
            plt.contour(plotCTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_Pred[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_isodistance_axial(ax, Z, plotCTV, plotCTV_Pred):

            fig.add_axes(ax)
            plt.imshow(np.flip(plotCT[:, :, Z].transpose(), axis=0), extent=[x[0], x[-1], y[0], y[-1]], cmap='gray') #, vmin=0, vmax=2.5)
            plt.contourf(plotComp[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='blue', alpha=0.2)
            plt.contour(plotPred[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='orange', linewidths=1, linestyles='dashed')
            plt.contourf(plotGTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='yellow', alpha=0.2)
            plt.contour(plotGTV_Pred[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='yellow', linewidths=1.5, linestyles='dashed')
            plt.contour(plotCTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_Pred[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', linewidths=1.5, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_uncertainties_axial(ax1, ax2, ax3, Z, plotCTV, plotCTV_Pred):

            fig.add_axes(ax1)
            #plt.imshow(np.flip(plotCerebellumEntropy[:, :, Z].transpose(), axis=0), extent=[x[0], x[-1], y[0], y[-1]], cmap='gray')
            plt.contour(plotCTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_Pred[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red',
                        linewidths=1.5, linestyles='dashed')
            plt.title('Softmax entropy')

            fig.add_axes(ax2)
            #plt.imshow(np.flip(plotCerebellumVar[:, :, Z].transpose(), axis=0), extent=[x[0], x[-1], y[0], y[-1]], cmap='gray')
            plt.contour(plotCTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_Pred[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red',
                        linewidths=1.5, linestyles='dashed')
            plt.title('Softmax variance')

            fig.add_axes(ax3)
            #plt.imshow(np.flip(plotCerebellumDisagreement[:, :, Z].transpose(), axis=0), extent=[x[0], x[-1], y[0], y[-1]], cmap='gray')
            plt.contour(plotCTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='green', linewidths=1.5)
            plt.contour(plotCTV_Pred[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red',
                        linewidths=1.5, linestyles='dashed')
            plt.title('Prediction disagreement')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        # Plot
        fig = plt.figure()
        plt.axis('off')
        # plt.title('Interactive slider')

        ax1 = fig.add_subplot(131)
        plot_isodistance_sagittal(ax1, X_coord, ctv.imageArray, ctv_Pred.imageArray)
        ax2 = fig.add_subplot(132)
        plot_isodistance_coronal(ax2, Y_coord, ctv.imageArray, ctv_Pred.imageArray)
        ax3 = fig.add_subplot(133)
        plot_isodistance_axial(ax3, Z_coord, ctv.imageArray, ctv_Pred.imageArray)

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

            ctv.setCTV_isodistance(alpha4)
            ctv_Pred.setCTV_isodistance(alpha4)

            ax1.cla()
            plot_isodistance_sagittal(ax1, alpha1, ctv.imageArray, ctv_Pred.imageArray)
            ax2.cla()
            plot_isodistance_coronal(ax2, alpha2, ctv.imageArray, ctv_Pred.imageArray)
            ax3.cla()
            plot_isodistance_axial(ax3, alpha3, ctv.imageArray, ctv_Pred.imageArray)

            plt.draw()

        alpha_slider_1.on_changed(update1)
        alpha_slider_2.on_changed(update1)
        alpha_slider_3.on_changed(update1)
        alpha_slider_4.on_changed(update1)

        plt.show()

        # plot uncertainties

        # Plot
        fig = plt.figure()
        plt.axis('off')
        # plt.title('Interactive slider')

        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        plot_uncertainties_axial(ax1, ax2, ax3, Z_coord, ctv.imageArray, ctv_Pred.imageArray)

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

        def update2(val):

            alpha = int(alpha_slider_3.val)

            ax1.cla()
            ax2.cla()
            ax3.cla()
            plot_uncertainties_axial(ax1, ax2, ax3, alpha, ctv.imageArray, ctv_Pred.imageArray)

            plt.draw()


        alpha_slider_3.on_changed(update2)

        plt.show()

# calculate statistics
dice_ctv_mean, dice_ctv_std = np.mean(dice_ctv, axis=0), np.std(dice_ctv, axis=0)
SDS_ctv_mean, SDS_ctv_std= np.mean(SDS_ctv, axis=0), np.std(SDS_ctv, axis=0)
HD95_ctv_mean, HD95_ctv_std = np.mean(HD95_ctv, axis=0), np.std(HD95_ctv, axis=0)
HD_ctv_mean, HD_ctv_std = np.mean(HD_ctv, axis=0), np.std(HD_ctv, axis=0)

dice_bs_mean, dice_bs_std = np.mean(dice_bs, axis=0), np.std(dice_bs, axis=0)
SDS_bs_mean, SDS_bs_std = np.mean(SDS_bs, axis=0), np.std(SDS_bs, axis=0)
HD95_bs_mean, HD95_bs_std = np.mean(HD95_bs, axis=0), np.std(HD95_bs, axis=0)

for margin, m in zip(margins, range(len(margins))):

    print(f"{margin} mm CTV mean dice: {dice_ctv_mean[m]*100:.2f}% +/- {dice_ctv_std[m]*100:.2f}%")

    for tau, j in zip(SDS_tolerance, range(len(SDS_tolerance))):
        print(f"{margin} mm CTV mean surface dice (\u03C4 = {tau} mm): {SDS_ctv_mean[m, j]*100:.2f}% +/- {SDS_ctv_std[m, j]*100:.2f}%")

    print(f"{margin} mm CTV mean HD95: {HD95_ctv_mean[m]:.2f} mm +/- {HD95_ctv_std[m]:.2f} mm")
    print(f"{margin} mm CTV mean HD: {HD_ctv_mean[m]:.2f} mm +/- {HD_ctv_std[m]:.2f} mm")

# create plots

fig, axs = plt.subplots(nrows=1, ncols=2+len(SDS_tolerance), figsize=(7 * (2+len(SDS_tolerance)), 6))

i = 0
for m in range(len(margins)):
    axs[i].scatter(dice_bs[:, m], dice_ctv[:, m])
axs[i].set_title('DSC')
axs[i].set_xlabel('DSC Barriers')
axs[i].set_ylabel('DSC CTVs')

for j in range(len(SDS_tolerance)):
    i += 1
    for m in range(len(margins)):
        axs[i].scatter(SDS_ctv[:, m, j], SDS_bs[:, m, j])
    axs[i].set_title(f'SDS (\u03C4 = {SDS_tolerance[j]} mm)')
    axs[i].set_xlabel('SDS Barriers')
    axs[i].set_ylabel('SDS CTVs')

i += 1
for m in range(len(margins)):
    axs[i].scatter(HD95_ctv[:, m], HD95_bs[:, m])
axs[i].set_title('HD95')
axs[i].set_xlabel('HD95 Barriers')
axs[i].set_ylabel('HD95 CTVs')

plt.show()