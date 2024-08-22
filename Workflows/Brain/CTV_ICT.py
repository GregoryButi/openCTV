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
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric, SSDMetric
from dipy.io.image import load_nifti
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

from Process.Tensors import TensorDiffusion
from Process import Struct
from Process.CTVs import CTV
from Process.Transforms import TransformTensorAffine, TransformTensorDeformable
from Process.Solvers import SolverPDE
from Analysis.contourComparison import dice_score, percentile_hausdorff_distance

from opentps.core.data.images._image3D import Image3D

# input 

path_moving = '/media/gregory/Elements/Data/Atlas_Brain/DTI_FSL/MNI152_T1_1mm_brain_norm.nii.gz'
path_tensor = '/media/gregory/Elements/Data/Atlas_Brain/DTI_FSL/FSL_HCP1065_tensor_1mm_Ants.nii.gz'
path_segmentation = '/media/gregory/Elements/Data/Atlas_Brain/DTI_FSL/samseg_output/seg.mgz'
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
    'timepoint': [0, 200]  # [days]
}

# scale tensor values in mask
DT = tensor.getDiffusionTensorFKPP(model)
# override values in white matter mask
DT[WM] = 10 * DT[WM]
tensor.imageArray = DT
tensor.setBarrier(~brain_mask)

for system in ['reaction_diffusion', 'diffusion']:

    model['system'] = system

    # initialize lists
    dice_ICT = np.zeros((len(model['timepoint']), len(PIDs)))
    dice_FS = np.zeros((len(model['timepoint']), len(PIDs)))
    dice_PPD = np.zeros((len(model['timepoint']), len(PIDs)))
    MaxAE_ICT = np.zeros((len(model['timepoint']), len(PIDs)))
    MaxAE_FS = np.zeros((len(model['timepoint']), len(PIDs)))
    MaxAE_PPD = np.zeros((len(model['timepoint']), len(PIDs)))
    MAE_ICT = np.zeros((len(model['timepoint']), len(PIDs)))
    MAE_FS = np.zeros((len(model['timepoint']), len(PIDs)))
    MAE_PPD = np.zeros((len(model['timepoint']), len(PIDs)))
    HD95_ICT = np.zeros((len(model['timepoint']), len(PIDs)))
    HD95_FS = np.zeros((len(model['timepoint']), len(PIDs)))
    HD95_PPD = np.zeros((len(model['timepoint']), len(PIDs)))
    idp = 0
    for PID in PIDs:

        path_static = os.path.join(path_patients_dir, f'{PID}/Therapy-scan/MRI_CT/T1c_brain_norm.nii.gz')
        path_RTstructs = os.path.join(path_patients_dir, f'{PID}/Therapy-scan/Structures')

        # load data

        static, static_grid2world, voxel_size = load_nifti(path_static, return_voxsize=True)

        # load structures

        RTs = Struct()
        RTs.loadContours_folder(path_RTstructs, ['GTV_T1c', 'BS_T1c', 'Brain'], contour_names=['GTV', 'BS', 'Brain_mask'])
        RTs.setMask('External', static > 0, voxel_size)

        GTV = RTs.getMaskByName('GTV').imageArray

        BS = np.logical_or(~RTs.getMaskByName('Brain_mask').imageArray, RTs.getMaskByName('BS').imageArray)
        RTs.setMask('BS', BS, voxel_size)

        # Perform registration

        level_iters = [10000, 1000, 100]
        sigmas = [3.0, 1.0, 0.0]
        factors = [4, 2, 1]

        nbins = 32
        sampling_prop = None
        metric = MutualInformationMetric(nbins, sampling_prop)
        c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                              moving, moving_grid2world)

        affreg = AffineRegistration(metric=metric,
                                    level_iters=level_iters,
                                    sigmas=sigmas,
                                    factors=factors)
        # [STAGE 1]

        transform = TranslationTransform3D()
        params0 = None
        starting_affine = c_of_mass.affine
        translation = affreg.optimize(static, moving, transform, params0,
                                      static_grid2world, moving_grid2world,
                                      starting_affine=starting_affine, static_mask=(~GTV).astype(int))

        # [STAGE 2]

        transform = RigidTransform3D()
        params0 = None
        starting_affine = translation.affine
        rigid = affreg.optimize(static, moving, transform, params0,
                                static_grid2world, moving_grid2world,
                                starting_affine=starting_affine, static_mask=(~GTV).astype(int))

        # [STAGE 3]

        transform = AffineTransform3D()
        params0 = None
        starting_affine = rigid.affine
        affine = affreg.optimize(static, moving, transform, params0,
                                 static_grid2world, moving_grid2world,
                                 starting_affine=starting_affine, static_mask=(~GTV).astype(int))

        # [STAGE 4]

        metric = SSDMetric(3, smooth=8)  # Sum of Squared Differences
        level_iters = [10, 10, 5]  # 100, 100, 25

        sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)  # inv_iter=100, inv_tol=1e-7
        prealign = affine.affine
        mapping = sdr.optimize(static, moving, static_grid2world, moving_grid2world, prealign=prealign)

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

        domain = RTs.getBoundingBox('External', 5)

        # transform domain to global space
        coDomain = mapping.transform_inverse(domain) >= 0.5

        # Solve Fisher–Kolmogorov (FKPP) equation

        # define timestep
        deltat = 1 / (6 * tensor.imageArray.max() * (1 / tensor.spacing ** 2).sum())
        #deltat = 0.2

        #transform = TransformTensorAffine(affine)
        transform = TransformTensorDeformable(mapping)

        # apply transformations
        tensor_ICT = transform.getTensorDiffusionTransformed(tensor, method='ICT')
        tensor_FS = transform.getTensorDiffusionTransformed(tensor, method='FS')
        tensor_PPD = transform.getTensorDiffusionTransformed(tensor, method='PPD')

        # solve FK equation in different coordinate spaces

        solver = SolverPDE(copy.deepcopy(coSource), copy.deepcopy(coBarriers), copy.deepcopy(tensor), coDomain)
        cells = solver.getDensity_xyz(model, transform, domain, deltat=deltat)

        solver_ICT = SolverPDE(copy.deepcopy(Source), copy.deepcopy(Barriers), copy.deepcopy(tensor_ICT), domain)
        cells_ICT = solver_ICT.getDensity_uvw(model, transform, deltat=deltat)

        solver_FS = SolverPDE(copy.deepcopy(Source), copy.deepcopy(Barriers), copy.deepcopy(tensor_FS), domain)
        cells_FS = solver_FS.getDensity_uvw(model, transform, deltat=deltat)

        solver_PPD = SolverPDE(copy.deepcopy(Source), copy.deepcopy(Barriers), copy.deepcopy(tensor_PPD), domain)
        cells_PPD = solver_PPD.getDensity_uvw(model, transform, deltat=deltat)

        # Plot 2D

        x, y, z = map(int, np.round(com(cells[0])))

        static_crop = Image3D(imageArray=static.copy())
        static_crop.reduceGrid_mask(domain)

        # transform RTs
        RTs_crop = copy.deepcopy(RTs)
        RTs_crop.reduceGrid_mask(domain)

        GTV_crop = RTs_crop.getMaskByName('GTV').imageArray

        norm = LogNorm(vmin=0.1, vmax=100)
        #colors = ['blue', 'orange', 'green', 'red', 'black']

        threshold = 1

        cells_show = cells[-1].copy()
        cells_show[cells_show < norm.vmin] = np.nan

        cells_ICT_show = cells_ICT[-1].copy()
        cells_ICT_show[cells_ICT_show < norm.vmin] = np.nan

        cells_FS_show = cells_FS[-1].copy()
        cells_FS_show[cells_FS_show < norm.vmin] = np.nan

        cells_PPD_show = cells_PPD[-1].copy()
        cells_PPD_show[cells_PPD_show < norm.vmin] = np.nan

        # setting font size
        plt.rcParams.update({'font.size': 12})
        cmap = plt.cm.get_cmap('inferno', 15)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

        ax = axs[0]
        ax.imshow(np.flip(static_crop.imageArray[:, :, z].transpose(), axis=0), cmap='gray', vmin=0, vmax=2.5)
        ax.contour(np.flip(GTV_crop[:, :, z].transpose(), axis=0), colors='magenta')
        im = ax.imshow(np.flip(cells_show[:, :, z].transpose(), axis=0), cmap=cmap, norm=norm, alpha=0.6)
        # Add a colorbar for the imshow plot
        cbar = plt.colorbar(im, ax=ax)
        #cbar.set_label("Tumor cell density")
        ax.contour(np.flip((cells_show >= threshold)[:, :, z].transpose(), axis=0), colors='white')

        # Create the labels and corresponding colors
        labels = ['GTV', 'CTV Ref.']
        colors = ['magenta', 'white']

        # Add the labels as text boxes
        for i, label in enumerate(labels):
            ax.text(0.05, 0.925 - i * 0.075, label, color=colors[i], transform=ax.transAxes,
                    bbox=dict(facecolor='none', alpha=0.5, edgecolor='none'))

        ax.set_xlabel(r"$x_1$ (mm)")
        ax.set_ylabel(r"$x_2$ (mm)")

        ax = axs[1]

        ax.imshow(np.flip(static_crop.imageArray[:, :, z].transpose(), axis=0), cmap='gray', vmin=0, vmax=2.5)
        ax.contour(np.flip(GTV_crop[:, :, z].transpose(), axis=0), colors='magenta')
        ax.contour(np.flip((cells_show >= threshold)[:, :, z].transpose(), axis=0), colors='white')
        ax.contour(np.flip((cells_ICT_show >= threshold)[:, :, z].transpose(), axis=0), colors='red')
        ax.contour(np.flip((cells_FS_show >= threshold)[:, :, z].transpose(), axis=0), colors='green')
        ax.contour(np.flip((cells_PPD_show >= threshold)[:, :, z].transpose(), axis=0), colors='blue')

        # Create the labels and corresponding colors
        labels = ['GTV', 'CTV Ref.', 'CTV ICT', 'CTV FS', 'CTV PPD']
        colors = ['magenta', 'white', 'red', 'green', 'blue']

        # Add the labels as text boxes
        for i, label in enumerate(labels):
            ax.text(0.05, 0.925 - i * 0.075, label, color=colors[i], transform=ax.transAxes,
                    bbox=dict(facecolor='none', alpha=0.5, edgecolor='none'))

        ax.set_xlabel(r"$x_1$ (mm)")
        ax.set_ylabel(r"$x_2$ (mm)")

        X, Y, _ = static_crop.getMeshGridPositions()

        X_plot = X[:, y, z]
        Y_plot = Y[x, :, z]
        idx = int(len(X_plot) / 2 - 1)

        ax = axs[2]

        #for i in range(1,len(model['timepoint'])):
        ax.plot(X_plot[::3], cells[-1][::3, y, z], "o", color='black', markerfacecolor='none')
        ax.plot(X_plot, cells_ICT[-1][:, y, z], color='red')
        ax.plot(X_plot, cells_FS[-1][:, y, z], color='green')
        ax.plot(X_plot[::3], cells_PPD[-1][::3, y, z], "v", color='blue', markerfacecolor='none')

        #ax.set_yscale('log')
        ax.set_ylim(1, 105)

        ax.legend(['Ref.', 'ICT', 'FS', 'PPD'])
        ax.set_xlabel(r"$x_1$ (mm)")
        ax.set_xlim(X_plot[cells[-1][:, y, z] >= 0.1].min() - 5, X_plot[cells[-1][:, y, z] >= 0.1].max() + 5)
        ax.set_ylabel("Tumor cell density")

        #plt.savefig(os.path.join(os.getcwd(), f'CTV_comp_{PID}_'+model['system']+'.pdf'), format='pdf',bbox_inches='tight')
        plt.show()

        # compute FA, and colored FA maps
        _, _, RGB = tensor.get_FA_MD_RGB()
        _, _, RGB_ICT = tensor_ICT.get_FA_MD_RGB()
        _, _, RGB_FS = tensor_FS.get_FA_MD_RGB()
        _, _, RGB_PPD = tensor_PPD.get_FA_MD_RGB()

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(np.transpose(RGB[:, :, z, :], (1, 0, 2)))
        plt.title('Original')
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        plt.subplot(1, 3, 2)
        plt.imshow(np.transpose(RGB_ICT[:, :, z, :], (1, 0, 2)))
        plt.title('ICT')
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        plt.subplot(1, 3, 3)
        plt.imshow(np.transpose(RGB_PPD[:, :, z, :], (1, 0, 2)))
        plt.title('PPD')
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        plt.show()

        # evaluate CTV against reference

        for i in range(len(model['timepoint'])):
            ctv_ref = CTV(imageArray=cells[i] >= threshold, spacing=voxel_size)

            ctv_ICT = CTV(imageArray=cells_ICT[i] >= threshold, spacing=voxel_size)
            ctv_FS = CTV(imageArray=cells_FS[i] >= threshold, spacing=voxel_size)
            ctv_PPD = CTV(imageArray=cells_PPD[i] >= threshold, spacing=voxel_size)

            # exclude GTV from volume overlap analysis

            dice_ICT[i, idp] = dice_score(np.logical_and(ctv_ref.imageArray, ~GTV_crop),
                                          np.logical_and(ctv_ICT.imageArray, ~GTV_crop))
            dice_FS[i, idp] = dice_score(np.logical_and(ctv_ref.imageArray, ~GTV_crop),
                                         np.logical_and(ctv_FS.imageArray, ~GTV_crop))
            dice_PPD[i, idp] = dice_score(np.logical_and(ctv_ref.imageArray, ~GTV_crop),
                                          np.logical_and(ctv_PPD.imageArray, ~GTV_crop))

            MaxAE_ICT[i, idp] = np.max(abs(cells[i][ctv_ref.imageArray] - cells_ICT[i][ctv_ref.imageArray]))
            MaxAE_FS[i, idp] = np.max(abs(cells[i][ctv_ref.imageArray] - cells_FS[i][ctv_ref.imageArray]))
            MaxAE_PPD[i, idp] = np.max(abs(cells[i][ctv_ref.imageArray] - cells_PPD[i][ctv_ref.imageArray]))

            MAE_ICT[i, idp] = (np.sum(
                abs(cells[i][ctv_ref.imageArray] - cells_ICT[i][ctv_ref.imageArray]))) / ctv_ref.imageArray.sum()
            MAE_FS[i, idp] = (np.sum(
                abs(cells[i][ctv_ref.imageArray] - cells_FS[i][ctv_ref.imageArray]))) / ctv_ref.imageArray.sum()
            MAE_PPD[i, idp] = (np.sum(
                abs(cells[i][ctv_ref.imageArray] - cells_PPD[i][ctv_ref.imageArray]))) / ctv_ref.imageArray.sum()

            HD95_ICT[i, idp] = percentile_hausdorff_distance(ctv_ref.getMeshpoints(), ctv_ICT.getMeshpoints(),
                                                             percentile=95)
            HD95_FS[i, idp] = percentile_hausdorff_distance(ctv_ref.getMeshpoints(), ctv_FS.getMeshpoints(),
                                                            percentile=95)
            HD95_PPD[i, idp] = percentile_hausdorff_distance(ctv_ref.getMeshpoints(), ctv_PPD.getMeshpoints(),
                                                             percentile=95)

        # increment patient index    
        idp += 1

        # save results
    #np.save(os.path.join(os.getcwd(), 'timepoint_'+model['system']), model['timepoint'])
    #np.save(os.path.join(os.getcwd(), 'dice_ICT_'+model['system']), dice_ICT)
    #np.save(os.path.join(os.getcwd(), 'dice_FS_'+model['system']), dice_FS)
    #np.save(os.path.join(os.getcwd(), 'dice_PPD_'+model['system']), dice_PPD)
    #np.save(os.path.join(os.getcwd(), 'MAE_ICT_'+model['system']), MAE_ICT)
    #np.save(os.path.join(os.getcwd(), 'MAE_FS_'+model['system']), MAE_FS)
    #np.save(os.path.join(os.getcwd(), 'MAE_PPD_'+model['system']), MAE_PPD)
    #np.save(os.path.join(os.getcwd(), 'HD95_ICT_'+model['system']), HD95_ICT)
    #np.save(os.path.join(os.getcwd(), 'HD95_FS_'+model['system']), HD95_FS)
    #np.save(os.path.join(os.getcwd(), 'HD95_PPD_'+model['system']), HD95_PPD)
