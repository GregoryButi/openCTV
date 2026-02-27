#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:12:05 2024

@author: gregory
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass as com
from dipy.io.image import load_nifti, save_nifti
from totalsegmentator.python_api import totalsegmentator

from opentps.core.data.images._image3D import Image3D

from Process.CTVs import CTVGeometric
from Process import Struct
from Process.ImageSegmentation_nnUNet.imageSegmentation import run_segmentation
from Process.config_runtime import configure_agd
configure_agd()

# patient ID
PID = 'GBM'

# DL model structure names
barrier_names = ['Brain', 'Brainstem', 'Cerebellum', 'Chiasm', 'OpticNerve_L', 'OpticNerve_R', 'Midline', 'Ventricles_connected']

# nn-UNet parameters
model_segmentation = 'Dataset_CT_Brain_Segmentation_Radiotherapy'
config_segmentation = '3d_fullres'

# model parameters
model = {'model': 'Uniform',
         'obstacle': True
         }

# model parameters
margin = 15  # [mm]

# input

path_CT = f'../../Input/{PID}/Therapy-scan/MRI_CT/CT.nii.gz'
path_RTstructs = f'../../Input/{PID}/Therapy-scan/Structures'
path_TotalSegmentator = f'../../Input/{PID}/Therapy-scan/Structures_totalsegmentator_new'

def main():

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

    totalsegmentator(path_CT, path_TotalSegmentator, task='brain_structures')

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

    # save CTV

    name_CTV_DL = 'CTV_auto'
    name_CTV_TS = 'CTV_TS'

    save_nifti(name_CTV_DL + '.nii.gz', ctv_DL.imageArray.astype(float), static_grid2world)
    save_nifti(name_CTV_TS + '.nii.gz', ctv_TS.imageArray.astype(float), static_grid2world)

    # voxel display
    COM = np.array(com(gtv))
    Z_coord = int(COM[2])

    # prepare figures
    plotCT = CT.imageArray
    x, y, z = CT.getMeshGridAxes()

    plotGTV = gtv.astype(float).copy()
    plotGTV[~gtv] = np.nan

    # Plot
    fig = plt.figure()
    plt.axis('off')

    plt.imshow(np.flip(plotCT[:, :, Z_coord].transpose(), axis=0), extent=[x[0], x[-1], y[0], y[-1]], cmap='gray', vmin=-1000, vmax=600)
    plt.contourf(plotGTV[:, :, Z_coord].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='yellow', alpha=0.25)
    plt.contour(ctv_DL.imageArray[:, :, Z_coord].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', linewidths=1.5, linestyles='dashed')
    plt.contour(ctv_TS.imageArray[:, :, Z_coord].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='orange', linewidths=1.5, linestyles='dashed')
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    plt.savefig(f"plot_{PID}.pdf", bbox_inches='tight')

if __name__ == "__main__":
    main()