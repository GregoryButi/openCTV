#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:12:05 2024

@author: gregory
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass as com

import copy

from Process.Tensors import TensorDiffusion
from Process.CTVs import CTVGeometric
from Process import Struct

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.viz import *
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.viz import colormap
import dipy.reconst.dti as dti

interactive = False

# load data

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
label_fname = get_fnames('stanford_labels')

# get the T1 to show anatomical context of the WMPL
t1_fname = get_fnames('stanford_t1')
t1_data, affine, voxel_size = load_nifti(t1_fname, return_voxsize=True)

data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

white_matter = (labels == 1) | (labels == 2)
brain = labels > 0

# load structures

RTs = Struct()

# define barrier structures
RTs.setMask('WM', white_matter, spacing=voxel_size, roi_type='Barrier_soft')
RTs.setMask('Brain', brain, spacing=voxel_size)
RTs.setMask('BS', ~RTs.getMaskByName('Brain').imageArray, spacing=voxel_size, roi_type='Barrier')

# create virtual GTV as sphere

Brain = RTs.getMaskByName('Brain')
X_world, Y_world, Z_world = Brain.getMeshGridPositions()
RTs.createSphere('GTV', X_world, Y_world, Z_world, Brain.centerOfMass + np.array([30, 50, 0]), (5, 5, 5), spacing=voxel_size, roi_type='GTV')
GTV = RTs.getMaskByName('GTV').imageArray

# fit DTI signal to the data
dti_wls = dti.TensorModel(gtab)

fit_wls = dti_wls.fit(data)

fa = fit_wls.fa
evals = fit_wls.evals
evecs = fit_wls.evecs

# define tensors
Lambda = np.zeros(evecs.shape)
Lambda[..., 0:3, 0:3] = np.eye(3)

Lambda[..., 0, 0] = evals[..., 0]
Lambda[..., 1, 1] = evals[..., 1]
Lambda[..., 2, 2] = evals[..., 2]

tensorfit = np.matmul(evecs, np.matmul(Lambda, np.linalg.inv(evecs)))

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csa_model = CsaOdfModel(gtab, sh_order_max=6)
csa_peaks = peaks_from_model(csa_model, data, default_sphere, relative_peak_threshold=.8, min_separation_angle=45, mask=white_matter)

if interactive and has_fury:
    scene = window.Scene()
    scene.add(actor.peak_slicer(csa_peaks.peak_dirs, peaks_values= csa_peaks.peak_values, colors=None))

    # window.record(scene, out_path='csa_direction_field.png', size=(900, 900))
    window.show(scene, size=(800, 800))

stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, .25)

seed_mask = (labels == 2) # corpus callosum

seeds = utils.seeds_from_mask(white_matter, affine, density=[2, 2, 2])

# Initialization of LocalTracking. The computation happens in the next step.
streamlines_generator = LocalTracking(csa_peaks, stopping_criterion, seeds, affine=affine, step_size=0.5)
# Generate streamlines object
streamlines = Streamlines(streamlines_generator)

if interactive and has_fury:
    # Prepare the display objects.
    color = colormap.line_colors(streamlines)

    streamlines_actor = actor.line(streamlines, colors=colormap.line_colors(streamlines))

    # Create the 3D display.
    scene = window.Scene()
    scene.add(streamlines_actor)

    # Save still images for this static example. Or for interactivity use
    # window.record(scene, out_path='tractogram_EuDX.png', size=(800, 800))
    window.show(scene)

# calculate the WMPL
wmpl = utils.path_length(streamlines, affine, GTV, fill_value=np.inf)

# create tensor object
tensor = TensorDiffusion(imageArray=tensorfit)

FA, _, RGB = tensor.get_FA_MD_RGB()

# define tumor spread model

modelDTI = {
    'obstacle': False,
    'model': 'Anisotropic',
    'model-DTI': 'Rekik',
    'resistance': 0.02,
    }

margin = 15

# Run fast marching methods

ctv_stl = CTVGeometric()
ctv_stl.compute_distance_done = True
ctv_stl.distance3D = wmpl
ctv_stl.setCTV_isodistance(margin)

ctv_classic = CTVGeometric(rts=RTs)
ctv_classic.setCTV_metric(ctv_stl, metric='volume')

ctv_dti = CTVGeometric(rts=RTs, tensor=copy.deepcopy(tensor), model=modelDTI)
ctv_dti.setCTV_metric(ctv_stl, metric='volume')
    
# Create 2D plots

wmpl_show = np.ma.masked_where(wmpl == np.inf, wmpl)
GTV_show = np.where(~GTV, np.nan, GTV)

# voxel display
COM = np.array(com(GTV))
Y_coord = int(COM[1])

# prepare figures
plotMR = t1_data.copy()
plotGTV = GTV.astype(float).copy()
plotGTV[~GTV] = np.NaN

plt.figure()

plt.subplot(1,3,1)
plt.imshow(np.flip(plotMR[:, Y_coord, :].transpose(), axis=0), cmap='gray')
plt.contourf(np.flip(plotGTV[:, Y_coord, :].transpose(), axis=0), colors='red', alpha=0.5)
plt.imshow(np.flip(csa_peaks.gfa[:, Y_coord, :].transpose(), axis=0), alpha=0.8)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

plt.subplot(1,3,2)
plt.imshow(np.flip(plotMR[:, Y_coord, :].transpose(), axis=0), cmap='gray')
plt.contourf(np.flip(plotGTV[:, Y_coord, :].transpose(), axis=0), colors='yellow', alpha=0.5)
plt.imshow(np.flip(wmpl_show[:, Y_coord, :].transpose(), axis=0), alpha=0.8)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

plt.subplot(1,3,3)
plt.imshow(np.flip(plotMR[:, Y_coord, :].transpose(), axis=0), cmap='gray')
plt.contourf(np.flip(plotGTV[:, Y_coord, :].transpose(), axis=0), colors='yellow', alpha=0.5)
plt.imshow(np.flip(np.transpose(RGB[:, Y_coord, :], axes=(1,0,2)), axis=0), alpha=0.5)
plt.contour(np.flip(ctv_classic.imageArray[:, Y_coord, :].transpose(), axis=0), colors='red', linewidths=1.5)
plt.contour(np.flip(ctv_dti.imageArray[:, Y_coord, :].transpose(), axis=0), colors='white', linewidths=1.5)
plt.contour(np.flip(ctv_stl.imageArray[:, Y_coord, :].transpose(), axis=0), colors='white', linestyles='dashed', linewidths=1.5)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

plt.show()
