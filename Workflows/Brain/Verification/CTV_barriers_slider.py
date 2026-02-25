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
from dipy.viz import regtools
import itertools

from opentps.core.data.images._image3D import Image3D

from Process.Tensors import TensorDiffusion
from Process.CTVs import CTVGeometric
from Process.Transforms import TransformTensorAffine, TransformTensorDeformable
from Process.ImageRegistrationDIPY import ImageRegistrationRigid, ImageRegistrationDeformable
from Process import Struct

visualization = True
includeDTI = False
register_post2pre = False
postprocessing_barriers = True
save_results = False
calculate_CTV_rejection = True

# patient IDs
patient_dir = '/media/gregory/Elements/Data/MGH_Glioma/TVD_Processed'
PIDs = ['TVD_077_GBM', 'TVD_080_GBM'] # 'TVD_003_GBM', 'TVD_077_GBM', 'TVD_080_GBM'

# atlas
path_atlas = '../../../Input/Atlas/MNI152_T1_1mm_brain_norm.nii.gz'
path_tensor = '../../../Input/Atlas/FSL_HCP1065_tensor_1mm_Ants.nii.gz'
template, template_grid2world = load_nifti(path_atlas)
tensor = TensorDiffusion()
tensor.loadTensor(path_tensor, format='ANTs')

# Set the output directory
output_dir = "/home/gregory/Downloads/Test"
os.makedirs(output_dir, exist_ok=True)

barrier_names = ['Brain', 'Brainstem', 'Cerebellum', 'Chiasm', 'OpticNerve_L', 'OpticNerve_R', 'Midline', 'Ventricles_connected']
OAR_names = ['Hippocampus_L', 'Hippocampus_R', 'Pituitary']

# nn-UNet parameters
model_segmentation = 'Dataset015_GLIS-RT'
config_segmentation = '3d_fullres'

model = {'model': 'Uniform',
         'obstacle': True
         }

# without DTI
#biopsies_voxel_PIDs = [[[231,301, 76], [191,216, 86], [261,271, 86], [236,321, 86], [261,256, 91]]] # TVD_003_GBM
#biopsies_world_PIDs = [[[157.23, 205.08, 187.50], [129.88, 146.97, 212.50], [177.73, 184.57, 212.50], [160.64, 218.75, 212.50], [177.73, 174.32, 225.00]]] # TVD_003_GBM

#biopsies_voxel_PIDs = [[[231,301, 76], [181,221, 86], [261,271, 86], [236,321, 86], [261,256, 91]]] # TVD_003_GBM
#biopsies_world_PIDs = [[[157.23, 205.08, 187.50], [123.05, 150.39, 212.50], [177.73, 184.57, 212.50], [160.64, 218.75, 212.50], [177.73, 174.32, 225.00]]] # TVD_003_GBM

#biopsies_voxel_PIDs = [[[221,286, 56], [196,301, 56], [256,261, 66], [181,181, 71], [181,196, 76]], # TVD_077_GBM
#                       [[271,206, 71], [306,186, 76], [251,186, 86], [256,201, 86], [241,201, 91]]] # TVD_080_GBM
#biopsies_world_PIDs = [[[141.80, 183.69, 137.50], [125.68, 193.36, 137.50], [164.36, 167.58, 162.50], [116.02, 116.02, 175.00], [116.02, 125.68, 187.50]], # TVD_077_GBM
#                       [[263.67, 200.20, 175.00], [297.85, 180.66, 187.50], [244.14, 180.66, 212.50], [249.02, 195.31, 212.50], [234.37, 195.31, 225.00]]] # TVD_080_GBM

# with DTI
biopsies_voxel_PIDs = [[[221,326, 81], [261,271, 86], [236,321, 86], [261,256, 91], [236,216, 96]]] # TVD_003_GBM
biopsies_world_PIDs = [[[150.39, 222.17, 200.00], [177.73, 184.57, 212.50], [160.64, 218.75, 212.50], [177.73, 174.32, 225.00], [160.64, 146.97, 237.50]]] # TVD_003_GBM

biopsies_voxel_PIDs = [[[221,296, 56], [196,301, 56], [226,316, 61], [261,256, 66], [221,191, 76]], # TVD_077_GBM
                       [[271,211, 71], [276,226, 71], [306,186, 76], [251,186, 86], [256,201, 86]]] # TVD_080_GBM
biopsies_world_PIDs = [[[141.80, 190.14, 137.50], [125.68, 193.36, 137.50], [145.02, 203.03, 150.00], [167.58, 164.36, 162.50], [141.80, 122.46, 187.50]], # TVD_077_GBM
                       [[263.67, 205.08, 175.00], [268.55, 219.73, 175.00], [297.85, 180.66, 187.50], [244.14, 180.66, 212.50], [249.02, 195.31, 212.50]]] # TVD_080_GBM

# model parameters
margin = 15  # [mm]

for PID, i in zip(PIDs, range(len(PIDs))):

    # input

    path_CT = os.path.join(patient_dir, f'{PID}/Therapy-scan/MRI_CT/CT.nii.gz')
    path_T1c_post = os.path.join(patient_dir, f'{PID}/Therapy-scan/MRI_CT/T1c_aligned_Post-Op.nii.gz')
    path_T1c_pre = os.path.join(patient_dir, f'{PID}/Therapy-scan/MRI_CT/T1c_aligned_Pre-Op.nii.gz')
    path_RTstructs_manual = os.path.join(patient_dir, f'{PID}/Therapy-scan/Structures')
    path_RTstructs_DL = os.path.join(patient_dir, f'{PID}/Therapy-scan/Structures_{model_segmentation}_{config_segmentation}')

    biopsies_voxel = np.array(biopsies_voxel_PIDs[i])
    biopsies_world = np.array(biopsies_world_PIDs[i])

    # output
    output_folder = os.path.join(output_dir, PID)
    os.makedirs(output_folder, exist_ok=True)

    # load data

    ct, ct_grid2world, voxel_size = load_nifti(path_CT, return_voxsize=True)
    CT = Image3D(imageArray=ct, spacing=voxel_size)

    t1c_post, post_grid2world, _ = load_nifti(path_T1c_post, return_voxsize=True)
    t1c_pre, pre_grid2world, _ = load_nifti(path_T1c_pre, return_voxsize=True)

    # load ground truth structures

    RTs = Struct()
    RTs.loadContours_folder(path_RTstructs_DL, barrier_names)
    RTs.loadContours_folder(path_RTstructs_manual, ['GTV_T1c', 'GTV_FLAIR', 'CTV'], contour_names=['GTV', 'CTV_HighRisk', 'CTV_LowRisk'], contour_types=['GTV', None, None]) # ['GTV_Gad', 'GTV', 'CTV'], ['GTV_T1c', 'GTV_FLAIR', 'CTV']
    RTs.loadContours_folder(path_RTstructs_manual, OAR_names)

    # load ground truth masks

    gtv = RTs.getMaskByName('GTV').imageArray

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

    if postprocessing_barriers:

        # perform morphological operations (post-processing step)

        Brain_dilated = RTs.getMaskByName('Brain').copy()
        Brain_dilated.dilateMask(radius=(2, 2, 3)) # 2 * voxel_size[0], 2 * voxel_size[1], 1 * voxel_size[2]

        cerebellum = np.logical_and(cerebellum, ~Brain_dilated.imageArray)

    # define baseline barrier structures

    bs = np.logical_or((brain + cerebellum + chiasm + optic_nerves + brainstem) == 0, (midline + ventricles) > 0)
    RTs.setMask('BS', bs, spacing=voxel_size, roi_type='Barrier')

    if includeDTI:

        # include WM as soft barrier
        RTs.loadContours_folder(os.path.join(patient_dir, f'{PID}/Therapy-scan/Structures_samseg'), ['WM'], contour_types=['Barrier_soft'])

    if register_post2pre:

        static = t1c_pre
        moving = t1c_post
        static_grid2world = pre_grid2world
        moving_grid2world = post_grid2world

        diffeomorphic = ImageRegistrationDeformable(static, static_grid2world, moving, moving_grid2world, static_mask=None, level_iters=[10, 10, 5], metric='CC')
        mapping = diffeomorphic.get_mapping()

        # plot DIR results
        warped = mapping.transform(moving)
        regtools.overlay_slices(static, warped, None, 2, "Pre-op T1c", "Warped Post-op T1c", None)

        # apply transformations
        RTs.transformMasksDIPY(mapping, static_grid2world)


    if includeDTI:

        union = np.logical_or(RTs.getMaskByName('Brain').imageArray, RTs.getMaskByName('Cerebellum').imageArray)
        union = np.logical_or(union, RTs.getMaskByName('Brainstem').imageArray)

        # define target image for registration
        target = np.where(union, t1c_pre, 0.)
        target = np.where(~ventricles, target, 0.)

        target_mask = np.where(~gtv, union.astype(int), 0)

        static = target
        moving = template
        static_grid2world = pre_grid2world
        moving_grid2world = template_grid2world

        # Perform registration

        diffeomorphic = ImageRegistrationDeformable(static, static_grid2world, moving, moving_grid2world, static_mask=None, level_iters=[10, 10, 5], metric='CC')
        mapping = diffeomorphic.get_mapping()

        # plot DIR results
        warped = mapping.transform(moving)
        regtools.overlay_slices(static, warped, None, 2, "Static", "Warped", None)

        # apply tensor transformation

        transform = TransformTensorDeformable(mapping)
        tensor_transformed = transform.getTensorDiffusionTransformed(tensor, method='FS')

        # define model
        model['model'] = 'Anisotropic'
        model['model-DTI'] ='Rekik'

    # Run fast marching method

    CTV = CTVGeometric(rts=RTs, model=model, spacing=voxel_size)

    if includeDTI:
        CTV.tensor = tensor_transformed
        CTV.preferred['resistances'] = [0.1]

    # define calculation domain
    GTVBox = RTs.getBoundingBox('GTV', 150)

    CTV.setCTV_isodistance(15, solver='FMM', domain=GTVBox)

    # convert distance map into probability map

    dist = CTV.distance3D.copy()
    # Replace inf and -inf with 0
    dist[np.isinf(dist)] = 1E14

    ctv = RTs.getMaskByName('CTV_LowRisk').imageArray

    # Compute min and max ignoring NaNs
    min_val = 0
    finite_elements = CTV.distance3D[ctv][np.isfinite(CTV.distance3D[ctv])]
    max_val = np.nanmax(finite_elements)

    # Normalize to [0, 1]
    if max_val > min_val:
        dist_norm = (dist - min_val) / (max_val - min_val)
    else:
        dist_norm = np.zeros_like(dist)

    # If smaller distances = higher probability
    prob_map = 1 - dist_norm

    # Optionally ensure that previously infinite values remain 0
    prob_map[np.isnan(prob_map)] = 0.
    prob_map[prob_map<0] = 0.

    if save_results:
        if includeDTI:
            name_prob = 'probability3D_DTI'
        else:
            name_prob = 'probability3D'
        save_nifti(os.path.join(output_folder, name_prob + '.nii.gz'), prob_map.astype(float), pre_grid2world)

    # set GTV type as None
    RTs.setTypeByName('GTV', None)

    # define calculation domain
    reward_map_all = np.zeros(CT.gridSize)
    for name in OAR_names:

        RTs.setTypeByName(name, 'GTV')

        #OAR = CTVGeometric(rts=RTs, model={'model': 'Uniform', 'obstacle': True}, spacing=voxel_size)
        OAR = CTVGeometric(rts=RTs, model=None, spacing=voxel_size)
        OARBox = RTs.getBoundingBox(name, 10)

        OAR.setCTV_isodistance(5, solver='FMM', domain=OARBox)

        # convert distance map into reward map

        dist = OAR.distance3D.copy()
        # Replace inf and -inf with 0
        dist[np.isinf(dist)] = 1E14

        # min and max distance
        min_val = 0.0
        max_val = 10.0

        # Normalize to [0, 1]
        if max_val > min_val:
            dist_norm = (dist - min_val) / (max_val - min_val)
        else:
            dist_norm = np.zeros_like(dist)

        # If smaller distances = higher probability
        reward_map = 1 - dist_norm

        # Optionally ensure that previously infinite values remain 0
        reward_map[np.isnan(reward_map)] = 0.
        reward_map[reward_map <= 0] = 0.  # zero
        reward_map[dist == 0] = 0. # zero reward within structure

        name_reward = 'reward3D_' + name

        if save_results:
            save_nifti(os.path.join(output_folder, name_reward + '.nii.gz'), reward_map.astype(float), pre_grid2world)

        # add to total reward
        reward_map_all += reward_map

        RTs.setTypeByName(name, None)

    # save masks

    name_save = ['GTV', 'CTV_HighRisk', 'CTV_LowRisk'] + ['Brainstem', 'Chiasm', 'OpticNerve_L', 'OpticNerve_R'] + OAR_names
    for name in name_save:
        save_nifti(os.path.join(output_folder, name + '.nii.gz'), RTs.getMaskByName(name).imageArray.astype(float), pre_grid2world)

    if calculate_CTV_rejection:

        p = np.array([prob_map[voxel[0], voxel[1], voxel[2]] for voxel in list(biopsies_voxel)])
        inside = np.array([ctv[voxel[0], voxel[1], voxel[2]] for voxel in list(biopsies_voxel)])

        p = p[inside]

        Ks = [1, 2, 3] # rejection threshold, i.e. number of biopsies necessary to reject the CTV
        n = len(p)
        n_samples = 1000

        for K in Ks:
            if K < n:

                prob = 0.0  # initialize
                for _ in range(n_samples):
                    sample = (np.random.rand(len(p)) < p).astype(int)

                    if sample.sum() >= K:
                        prob += 1

                prob /= n_samples
                print(f"Rejection rate (≥{K} zeros): {prob * 100:.2f}%")
            else: print(f"Rejection rate 0%: not enough samples inside CTV for decision")

    # Create 2D plots

    if visualization:

        # voxel display
        COM = np.array(com(gtv))
        X_coord = int(COM[0])
        Y_coord = int(COM[1])
        Z_coord = int(COM[2])

        gtv = RTs.getMaskByName('GTV').imageArray
        gtvFlair = RTs.getMaskByName('CTV_HighRisk').imageArray
        ctv = RTs.getMaskByName('CTV_LowRisk').imageArray

        plotProbability = prob_map
        plotProbability[plotProbability<=0] = np.NaN

        plotReward = reward_map_all
        plotReward[plotReward <= 0] = np.NaN

        x, y, z = CT.getMeshGridAxes()
        x_grid, y_grid, z_grid = CT.getVoxelGridAxes()

        plotComp = (~bs).astype(float).copy()

        if includeDTI:

            plt.figure()
            _, _, RGB = tensor_transformed.get_FA_MD_RGB()
            plt.imshow(t1c_pre[:, :, Z_coord].transpose(), cmap='gray', vmin=-500, vmax=200)
            plt.imshow(np.transpose(RGB[:, :, Z_coord], axes=(1, 0, 2)), alpha=0.9)
            plt.show()

        # Build interactive plot

        def plot_isodistance_sagittal(ax, X):

            fig.add_axes(ax)
            ax.imshow(np.flip(t1c_pre[X, :, :].T, axis=0), extent=[y[0], y[-1], z[0], z[-1]], cmap='gray')

            ax.contour(gtv[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='yellow', linewidths=1.5)
            ax.contour(gtvFlair[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='red', linewidths=1.5)
            ax.contour(ctv[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='green', linewidths=1.5)

            ax.imshow(np.flip(plotProbability[X, :, :].T, axis=0), extent=[y[0], y[-1], z[0], z[-1]], cmap=plt.cm.coolwarm, alpha=0.75, origin='upper')
            #ax.imshow(np.flip(plotReward[X, :, :].T, axis=0), extent=[y[0], y[-1], z[0], z[-1]], cmap=plt.cm.summer, alpha=0.95, origin='upper')

            #slice_points = biopsies_voxel[biopsies_voxel[:, 0] == X]
            #if len(slice_points) > 0:
                #ax.scatter(slice_points[:, 1], slice_points[:, 2], s=40, c='cyan', edgecolors='black', marker='o', label='Biopsy')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_isodistance_coronal(ax, Y):

            fig.add_axes(ax)
            ax.imshow(np.flip(t1c_pre[:, Y, :].T, axis=0), extent=[x[0], x[-1], z[0], z[-1]], cmap='gray')

            ax.contour(gtv[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='yellow', linewidths=1.5)
            ax.contour(gtvFlair[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='red', linewidths=1.5)
            ax.contour(ctv[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='green', linewidths=1.5)

            ax.imshow(np.flip(plotProbability[:, Y, :].T, axis=0), extent=[x[0], x[-1], z[0], z[-1]], cmap=plt.cm.coolwarm, alpha=0.75, origin='upper')
            #ax.imshow(np.flip(plotReward[:, Y, :].T, axis=0), extent=[x[0], x[-1], z[0], z[-1]], cmap=plt.cm.summer, alpha=0.95, origin='upper')

            #slice_points = biopsies_voxel[biopsies_voxel[:, 1] == Y]
            #if len(slice_points) > 0:
                #ax.scatter(slice_points[:, 0],  slice_points[:, 2],  s=50, c='red', edgecolors='black', marker='x', linewidths=1.5, label='Biopsy')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_isodistance_axial(ax, Z):

            fig.add_axes(ax)

            ax.imshow(t1c_pre[:, :, Z].T, extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], cmap='gray')

            ax.contour(np.flip(gtv[:, :, Z].T, axis=0), extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], colors='yellow', linewidths=1.5)
            ax.contour(np.flip(gtvFlair[:, :, Z].T, axis=0), extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], colors='red', linewidths=1.5)
            ax.contour(np.flip(ctv[:, :, Z].T, axis=0), extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], colors='green', linewidths=1.5)

            ax.imshow(plotProbability[:, :, Z].T, extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], cmap=plt.cm.coolwarm, alpha=0.75, origin='upper')
            #ax.imshow(plotReward[:, :, Z].T, extent=[x[0], x[-1], y[0], y[-1]], cmap=plt.cm.summer, alpha=0.95, origin='upper')

            slice_points = biopsies_voxel[biopsies_voxel[:, 2] == Z]
            if len(slice_points) > 0:
                ax.scatter(slice_points[:, 0], slice_points[:, 1], s=50, c='red', edgecolors='black', marker='x', linewidths=1.5, label='Biopsy')

            ax.set_xlim(x_grid[0], x_grid[-1])
            ax.set_ylim(y_grid[0], y_grid[-1])

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