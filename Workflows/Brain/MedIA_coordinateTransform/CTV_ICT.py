#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:33:18 2022

@author: gregory
"""

import os
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import center_of_mass as com
import copy

from dipy.viz import regtools
from dipy.io.image import load_nifti

from Process import TensorDiffusion
from Process import Struct
from Process import TransformTensorAffine, TransformTensorDeformable
from Process import ImageRegistrationRigid
from Process import SolverPDE

from opentps.core.data.images._image3D import Image3D

# input

path_moving = '../../../Input/Atlas/MNI152_T1_1mm_brain_norm.nii.gz'
path_tensor = '../../../Input/Atlas/FSL_HCP1065_tensor_1mm_Ants.nii.gz'
path_segmentation = '../../../Input/Atlas/seg.mgz'
path_patients_dir = '/media/gregory/Elements/Data/MGH_Glioma/Processed_MNI152'

PIDs = ['GLI_003_AAC'] # ['GLI_001_GBM', 'GLI_003_AAC', 'GLI_004_GBM', 'GLI_005_GBM', 'GLI_006_ODG', 'GLI_008_GBM', 'GLI_009_GBM', 'GLI_017_AAC', 'GLI_044_AC', 'GLI_046_AC']

# load data 

moving, moving_grid2world, voxel_size = load_nifti(path_moving, return_voxsize=True)
brain_mask = moving > 0

tensor = TensorDiffusion()
tensor.loadTensor(path_tensor, format='ANTs')

segment, _, _ = load_nifti(path_segmentation, return_voxsize=True)

# define structures
WM = np.logical_or(segment == 41, segment == 2)

# Simulation parameters 

model = {
    'obstacle': True,
    'cell_capacity': 100.,  # [%]
    'proliferation_rate': 0.01,  # [1/day],
    'diffusion_magnitude': 0.025,  # [mm^2/day]
    #'system': 'diffusion' # diffusion, reaction_diffusion
    'timepoint': [0, 25, 50]  # [days]
}

# scale tensor values in mask
DT = tensor.getDiffusionTensorFKPP(model)
# override values in white matter mask
DT[WM] = 100 * DT[WM]
tensor.imageArray = DT
tensor.setBarrier(~brain_mask)

methods = [] # ['ICT', 'FS', 'PPD']
systems = ['reaction_diffusion', 'diffusion']
colors = ['red', 'green', 'blue']

# iso-cell density front
threshold = 1  # [%]

for system in systems:

    model['system'] = system

    idp = 0
    for PID in PIDs:

        path_static = os.path.join(path_patients_dir, f'{PID}/Therapy-scan/MRI_CT/T1c_brain.nii.gz')
        path_RTstructs = os.path.join(path_patients_dir, f'{PID}/Therapy-scan/Structures')

        # load data

        static, static_grid2world, voxel_size = load_nifti(path_static, return_voxsize=True)

        # load structures

        RTs = Struct()
        RTs.loadContours_folder(path_RTstructs, ['GTV_T1c', 'BS_T1c'], contour_names=['GTV', 'BS'])
        RTs.setMask('Brain_mask', static > 0, voxel_size)

        GTV = RTs.getMaskByName('GTV').imageArray

        BS = np.logical_or(~RTs.getMaskByName('Brain_mask').imageArray, RTs.getMaskByName('BS').imageArray)
        RTs.setMask('BS', BS, voxel_size)

        IR = ImageRegistrationRigid(static, static_grid2world, moving, moving_grid2world, static_mask=(~GTV).astype(int))
        #IR = ImageRegistrationDeformable(static, static_grid2world, moving, moving_grid2world, static_mask=(~GTV).astype(int), metric='SSD')

        mapping = IR.get_mapping()

        # plot DIR results
        warped = mapping.transform(moving)
        regtools.overlay_slices(static, warped, None, 2, "Fixed", "Deformed", None)
        plt.rcParams.update({'font.size': 8})
        #plt.savefig(os.path.join(os.getcwd(),f'DIR_{PID}.pdf'), format='pdf',bbox_inches='tight')

        # Simulate diffusion model

        Source = Image3D(imageArray=gaussian_filter(GTV.astype(float), sigma=1))  # blur distribution to avoid sharp gradients
        Barriers = Image3D(imageArray=RTs.getMaskByName('BS').imageArray)

        # transform source and barriers to global space
        coSource = Image3D(imageArray=mapping.transform_inverse(Source.imageArray) >= 0.5)
        coBarriers = Image3D(imageArray=mapping.transform_inverse(Barriers.imageArray) >= 0.5)

        # define calculation domain  

        domain = RTs.getBoundingBox('Brain_mask', 5)

        # transform domain to global space
        coDomain = mapping.transform_inverse(domain) >= 0.5

        # Solve Fisher–Kolmogorov (FKPP) equation

        # define timestep
        deltat = 1 / (6 * tensor.imageArray.max() * (1 / tensor.spacing ** 2).sum())
        #deltat = 0.2

        if isinstance(IR, ImageRegistrationRigid):
            transform = TransformTensorAffine(mapping)
        elif isinstance(IR, TransformTensorDeformable):
            transform = TransformTensorDeformable(mapping)

        solver = SolverPDE(copy.deepcopy(coSource), copy.deepcopy(coBarriers), copy.deepcopy(tensor), coDomain)
        cells = solver.getDensity_xyz(model, transform, domain, deltat=deltat)

        cells_transformed = []
        for method in methods:

            # apply transformations
            tensor_transformed = transform.getTensorDiffusionTransformed(tensor, method=method)

            # solve FK equation in different coordinate spaces
            solver = SolverPDE(copy.deepcopy(Source), copy.deepcopy(Barriers), copy.deepcopy(tensor_transformed), domain)
            cells_transformed.append(solver.getDensity_uvw(model, transform, deltat=deltat))

        # Plot 2D

        x, y, z = map(int, np.round(com(cells[0])))

        # crop images for visualization

        static_crop = Image3D(imageArray=static.copy())
        static_crop.reduceGrid_mask(domain)

        RTs_crop = copy.deepcopy(RTs)
        RTs_crop.reduceGrid_mask(domain)

        GTV_crop = RTs_crop.getMaskByName('GTV').imageArray

        static_crop = Image3D(imageArray=static.copy())
        static_crop.reduceGrid_mask(domain)

        tensor_crop = TensorDiffusion(imageArray=tensor.imageArray.copy())
        tensor_crop.reduceGrid_mask(domain)

        norm = LogNorm(vmin=0.1, vmax=100)
        #colors = ['blue', 'orange', 'green', 'red', 'black']

        plt.figure()

        _, _, RGB = tensor_crop.get_FA_MD_RGB()
        plt.imshow(np.flip(static_crop.imageArray[:, :, z].transpose(), axis=0), cmap='gray')
        plt.imshow(np.flip(np.transpose(RGB[:, :, z], axes=(1, 0, 2)), axis=0), alpha=0.75)

        for c in cells:
            cells_show = c.copy()
            cells_show[cells_show < norm.vmin] = np.nan

            plt.contour(np.flip((cells_show >= threshold)[:, :, z].transpose(), axis=0), colors='white')

        plt.show()

        # setting font size
        plt.rcParams.update({'font.size': 12})
        cmap = plt.cm.get_cmap('inferno', 15)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

        ax = axs[0]
        ax.imshow(np.flip(static_crop.imageArray[:, :, z].transpose(), axis=0), cmap='gray')
        ax.contour(np.flip(GTV_crop[:, :, z].transpose(), axis=0), colors='magenta', label='GTV')

        im = ax.imshow(np.flip(cells_show[:, :, z].transpose(), axis=0), cmap=cmap, norm=norm, alpha=0.6)
        # Add a colorbar for the imshow plot
        cbar = plt.colorbar(im, ax=ax)
        #cbar.set_label("Tumor cell density")
        ax.contour(np.flip((cells_show >= threshold)[:, :, z].transpose(), axis=0), colors='white', label='CTV Ref.')

        ax.legend()

        ax.set_xlabel(r"$x_1$ (mm)")
        ax.set_ylabel(r"$x_2$ (mm)")

        ax = axs[1]

        ax.imshow(np.flip(static_crop.imageArray[:, :, z].transpose(), axis=0), cmap='gray') # , vmin=0, vmax=2.5)
        ax.contour(np.flip(GTV_crop[:, :, z].transpose(), axis=0), colors='magenta', label='GTV')
        ax.contour(np.flip((cells_show >= threshold)[:, :, z].transpose(), axis=0), colors='white', label='CTV Ref.')

        for c, color, method in zip(cells_transformed, colors, methods):
            cells_show = c[-1].copy()
            cells_show[cells_show < norm.vmin] = np.nan

            ax.contour(np.flip((cells_show >= threshold)[:, :, z].transpose(), axis=0), colors=color, label=methods)

        ax.legend()

        ax.set_xlabel(r"$x_1$ (mm)")
        ax.set_ylabel(r"$x_2$ (mm)")

        X, Y, _ = static_crop.getMeshGridPositions()

        X_plot = X[:, y, z]
        Y_plot = Y[x, :, z]
        idx = int(len(X_plot) / 2 - 1)

        ax = axs[2]

        ax.plot(X_plot[::3], cells[-1][::3, y, z], "o", color='black', markerfacecolor='none', label='Ref.')

        for c, color, method in zip(cells_transformed, colors, methods):

            ax.plot(X_plot, cells_transformed[-1][:, y, z], color=color, label=methods)

        ax.set_ylim(1, 105)

        ax.legend()
        ax.set_xlabel(r"$x_1$ (mm)")
        ax.set_xlim(X_plot[cells[-1][:, y, z] >= 0.1].min() - 5, X_plot[cells[-1][:, y, z] >= 0.1].max() + 5)
        ax.set_ylabel("Tumor cell density")

        #plt.savefig(os.path.join(os.getcwd(), f'CTV_comp_{PID}_'+model['system']+'.pdf'), format='pdf',bbox_inches='tight')
        plt.show()

        # evaluate CTV against reference

        # for i in range(len(model['timepoint'])):
        #     ctv_ref = CTV(imageArray=cells[i] >= threshold, spacing=voxel_size)
        #
        #     ctv_ICT = CTV(imageArray=cells_ICT[i] >= threshold, spacing=voxel_size)
        #     ctv_FS = CTV(imageArray=cells_FS[i] >= threshold, spacing=voxel_size)
        #     ctv_PPD = CTV(imageArray=cells_PPD[i] >= threshold, spacing=voxel_size)
        #
        #     # exclude GTV from volume overlap analysis
        #
        #     dice_ICT[i, idp] = dice_score(np.logical_and(ctv_ref.imageArray, ~GTV_crop),
        #                                   np.logical_and(ctv_ICT.imageArray, ~GTV_crop))
        #     dice_FS[i, idp] = dice_score(np.logical_and(ctv_ref.imageArray, ~GTV_crop),
        #                                  np.logical_and(ctv_FS.imageArray, ~GTV_crop))
        #     dice_PPD[i, idp] = dice_score(np.logical_and(ctv_ref.imageArray, ~GTV_crop),
        #                                   np.logical_and(ctv_PPD.imageArray, ~GTV_crop))
        #
        #     MaxAE_ICT[i, idp] = np.max(abs(cells[i][ctv_ref.imageArray] - cells_ICT[i][ctv_ref.imageArray]))
        #     MaxAE_FS[i, idp] = np.max(abs(cells[i][ctv_ref.imageArray] - cells_FS[i][ctv_ref.imageArray]))
        #     MaxAE_PPD[i, idp] = np.max(abs(cells[i][ctv_ref.imageArray] - cells_PPD[i][ctv_ref.imageArray]))
        #
        #     MAE_ICT[i, idp] = (np.sum(
        #         abs(cells[i][ctv_ref.imageArray] - cells_ICT[i][ctv_ref.imageArray]))) / ctv_ref.imageArray.sum()
        #     MAE_FS[i, idp] = (np.sum(
        #         abs(cells[i][ctv_ref.imageArray] - cells_FS[i][ctv_ref.imageArray]))) / ctv_ref.imageArray.sum()
        #     MAE_PPD[i, idp] = (np.sum(
        #         abs(cells[i][ctv_ref.imageArray] - cells_PPD[i][ctv_ref.imageArray]))) / ctv_ref.imageArray.sum()
        #
        #     HD95_ICT[i, idp] = percentile_hausdorff_distance(ctv_ref.getMeshpoints(), ctv_ICT.getMeshpoints(),
        #                                                      percentile=95)
        #     HD95_FS[i, idp] = percentile_hausdorff_distance(ctv_ref.getMeshpoints(), ctv_FS.getMeshpoints(),
        #                                                     percentile=95)
        #     HD95_PPD[i, idp] = percentile_hausdorff_distance(ctv_ref.getMeshpoints(), ctv_PPD.getMeshpoints(),
        #                                                      percentile=95)

        # increment patient index    
        idp += 1