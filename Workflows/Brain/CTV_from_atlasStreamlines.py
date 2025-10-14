#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:12:05 2024

@author: gregory
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass as com
import nibabel as nib
import copy
from read_mrtrix_tracks import read_mrtrix_tracks

from Process.Tensors import TensorDiffusion
from Process.CTVs import CTVGeometric
from Process import Struct

from dipy.io.image import load_nifti
from dipy.viz import colormap, has_fury, actor, window
from dipy.tracking import utils

interactive = False

# input

path_tractography = "/media/gregory/Elements/Data/Atlas_Brain/DTI_IIT/IIT_HARDI_tractogram_256.tck"
path_tensor = '/media/gregory/Elements/Data/Atlas_Brain/DTI_IIT/IITmean_tensor_256.nii.gz'
path_atlas = '/media/gregory/Elements/Data/Atlas_Brain/DTI_IIT/IITmean_t1_256.nii.gz'
path_brain_mask = '/media/gregory/Elements/Data/Atlas_Brain/DTI_IIT/IITmean_tensor_mask_256.nii.gz'
path_wm = '/media/gregory/Elements/Data/Atlas_Brain/DTI_IIT/IIT_WM_tissue_prob_256.nii.gz'
path_csf = '/media/gregory/Elements/Data/Atlas_Brain/DTI_IIT/IIT_CSF_tissue_prob_256.nii.gz'

# load data

t1, grid2world, voxel_size = load_nifti(path_atlas, return_voxsize=True)
brain = nib.load(path_brain_mask).get_fdata().astype(bool)
white_matter = nib.load(path_wm).get_fdata() >= 0.5
csf = nib.load(path_csf).get_fdata() >= 0.5

tensor = TensorDiffusion()
tensor.loadTensor(path_tensor, format='ANTs')

# load the streamlines from the trk file
header, vertices, line_starts, line_ends = read_mrtrix_tracks(path_tractography)
streamlines = [vertices[start:stop] for (start, stop) in zip(line_starts[::100], line_ends[::100])]
#streamlines = [vertices[start:stop] for (start, stop) in zip(line_starts, line_ends)]

if interactive and has_fury:
    # Prepare the display objects.
    color = colormap.line_colors(streamlines)

    streamlines_actor = actor.line(streamlines, colors=colormap.line_colors(streamlines))

    # Create the 3D display.
    scene = window.Scene()
    scene.add(streamlines_actor)

    # Save still images for this static example. Or for interactivity use
    #window.record(scene, out_path='tractogram_EuDX.png', size=(800, 800))
    window.show(scene)

# load structures

RTs = Struct()

# define barrier structures
RTs.setMask('WM', white_matter, spacing=voxel_size, roi_type='Barrier_soft')
RTs.setMask('Brain', brain, spacing=voxel_size)
RTs.setMask('BS', csf, spacing=voxel_size, roi_type='Barrier')

# create virtual GTV as sphere

Brain = RTs.getMaskByName('Brain')
X_world, Y_world, Z_world = Brain.getMeshGridPositions()
RTs.createSphere('GTV', X_world, Y_world, Z_world, Brain.centerOfMass + [10,5,-15], (5, 5, 5), spacing=voxel_size, roi_type='GTV')
GTV = RTs.getMaskByName('GTV').imageArray

# reduce calculation  of images and structures

#External = RTs.getMaskByName('External').imageArray
#MRI = Image3D(imageArray=MRI, spacing=voxel_size)

#MRI.reduceGrid_mask(External)
#RTs.reduceGrid_mask(External)
#tensor.reduceGrid_mask(External)

FA, _, RGB = tensor.get_FA_MD_RGB()

# calculate the WMPL
wmpl = utils.path_length(streamlines, grid2world, GTV, fill_value=np.inf)

margin = 20

ctv_stl = CTVGeometric(rts=RTs)
ctv_stl.distance3D = wmpl
ctv_stl.setCTV_isodistance(margin)
volume = ctv_stl.getVolume()

# define tumor spread model

modelDTI = {
    'obstacle': True,
    'model': 'Anisotropic',
    'model-DTI': 'Rekik',
    'resistance': 0.1,
    }


# Run fast marching method for classic CTV

ctv_classic = CTVGeometric(rts=RTs, model={'model': None, 'obstacle': True})
ctv_classic.setCTV_metric(ctv_stl, metric='volume')

# Run fast marching method for anisotropic CTV

ctv_dti = CTVGeometric(rts=RTs, tensor=copy.deepcopy(tensor), model=modelDTI)
ctv_dti.setCTV_metric(ctv_stl, metric='volume')
    
# Create 2D plots

GTV_show = np.where(~GTV, np.nan, GTV)

# voxel display
COM = np.array(com(GTV))
X_coord, Y_coord, Z_coord = int(COM[0]), int(COM[1]), int(COM[2])

# prepare figures
plotMR = t1.copy()
plotGTV = np.where(GTV, GTV, np.nan)
plotCSF = np.where(csf, csf, np.nan)
plotWM = np.where(white_matter, white_matter, np.nan)

vmax = 50
pltWMPL = np.where(wmpl > vmax, np.nan, wmpl)
pltRiemann = np.where(ctv_dti.distance3D / np.sqrt(modelDTI['resistance']) > vmax, np.nan, ctv_dti.distance3D / np.sqrt(modelDTI['resistance']))

plt.figure()

plt.subplot(2,3,1)
plt.imshow(np.flip(plotMR[X_coord, :, :].transpose(), axis=0), cmap='gray')
plt.imshow(np.flip(np.transpose(RGB[X_coord, :, :], axes=(1,0,2)), axis=0), alpha=0.8)
plt.title('Color-FA')
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

plt.subplot(2,3,2)
plt.imshow(np.flip(plotMR[X_coord, :, :].transpose(), axis=0), cmap='gray')
plt.contourf(np.flip(plotCSF[X_coord, :, :].transpose(), axis=0), colors='blue', alpha=0.5)
plt.contourf(np.flip(plotWM[X_coord, :, :].transpose(), axis=0), colors='green', alpha=0.5)
plt.contourf(np.flip(plotGTV[X_coord, :, :].transpose(), axis=0), colors='red', alpha=0.5)
plt.title('Segmentations')
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

plt.subplot(2,3,4)
plt.imshow(np.flip(plotMR[X_coord, :, :].transpose(), axis=0), cmap='gray')
plt.imshow(np.flip(pltWMPL[X_coord, :, :].transpose(), axis=0), alpha=0.8, vmax=vmax)
plt.colorbar()
plt.title('White matter path length')
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

plt.subplot(2,3,5)
plt.imshow(np.flip(plotMR[X_coord, :, :].transpose(), axis=0), cmap='gray')
plt.imshow(np.flip(pltRiemann[X_coord, :, :].transpose(), axis=0), alpha=0.8, vmax=vmax)
plt.colorbar()
plt.title('Riemann distance')
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

plt.subplot(2,3,6)
plt.imshow(np.flip(plotMR[X_coord, :, :].transpose(), axis=0), cmap='gray')
plt.contourf(np.flip(plotGTV[X_coord, :, :].transpose(), axis=0), colors='red', alpha=0.5)
plt.contour(np.flip(ctv_classic.imageArray[X_coord, :, :].transpose(), axis=0), colors='red', linewidths=1.5)
plt.contour(np.flip(ctv_dti.imageArray[X_coord, :, :].transpose(), axis=0), colors='green', linewidths=1.5)
plt.contour(np.flip(ctv_stl.imageArray[X_coord, :, :].transpose(), axis=0), colors='blue', linestyles='dashed', linewidths=1.5)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

plt.show()
