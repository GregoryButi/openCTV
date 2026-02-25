#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:12:05 2024

@author: gregory
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import center_of_mass, affine_transform
from dipy.io.image import load_nifti
import copy

from opentps.core.data.images._image3D import Image3D
from Process import TensorMetric
from Process import CTVGeometric
from Process import Struct

# input

path_patients_dir = '/media/gregory/Elements/Data/MGH_Glioma/Processed_MNI152'

path_MRI = os.path.join(path_patients_dir, 'GLI_003_AAC/Therapy-scan/MRI_CT/T1c.nii.gz')
path_RTstructs = os.path.join(path_patients_dir, 'GLI_003_AAC/Therapy-scan/Structures')
path_tensor = os.path.join(path_patients_dir, 'GLI_003_AAC/MT_warped.nii.gz')

# load data

MRI, grid2world, voxel_size = load_nifti(path_MRI, return_voxsize=True)

tensor = TensorMetric()
tensor.loadTensor(path_tensor, 'metric')

# load structures

RTs = Struct()
RTs.loadContours_folder(path_RTstructs, ['GTV_T1c', 'CTV_T1c', 'BS_T1c', 'CC_T1c', 'Brain', 'WM', 'GM', 'External_T1c'], contour_names=['GTV', 'CTV', 'BS', 'CC', 'Brain', 'WM', 'GM', 'External'])
RTs.smoothMasks(['GTV', 'CTV', 'CC', 'Brain', 'External'])

#BS = np.logical_or(~RTs.getMaskByName('Brain').imageArray, RTs.getMaskByName('BS').imageArray)
BS = np.logical_or(~RTs.getMaskByName('External').imageArray, RTs.getMaskByName('BS').imageArray)
RTs.setMask('BS', BS, voxel_size)

# define structure of preferred spread
RTs.setMask('PS', np.logical_and(RTs.getMaskByName('WM').imageArray, ~RTs.getMaskByName('BS').imageArray), voxel_size)

MRI = Image3D(imageArray=MRI, spacing=voxel_size)

External = RTs.getMaskByName('External').imageArray

# reduce calculation grid

MRI.reduceGrid_mask(External)
RTs.reduceGrid_mask(External)
tensor.reduceGrid_mask(External)

# reload contours

GTV = RTs.getMaskByName('GTV').imageArray
External = RTs.getMaskByName('External').imageArray
Brain = RTs.getMaskByName('Brain').imageArray
CC = RTs.getMaskByName('CC').imageArray
BS = RTs.getMaskByName('BS').imageArray
WM = RTs.getMaskByName('WM').imageArray
GM = RTs.getMaskByName('GM').imageArray

COM = np.array(center_of_mass(GTV))
    
# define model

model = {
    'obstacle': True,
    'model': 'Nonuniform', # None, 'Nonuniform', 'Anisotropic'
    'model-DTI': 'Rekik', # 'Clatz', 'Rekik'
    'resistance': 0.1,
    'anisotropy': 1.0
    }

margin = 20

# Run fast marching method for classic CTV

ctv_classic = CTVGeometric()
ctv_classic.setCTV_isodistance(margin, RTs, model={'model': None, 'obstacle': True})
volume = ctv_classic.getVolume()

# Run fast marching method

ctv_nominal = CTVGeometric()
#ctv_nominal.setCTV_volume(volume, RTs, tensor=copy.deepcopy(tensor), model=model, x0=margin)
ctv_nominal.setCTV_isodistance(margin, RTs, tensor=copy.deepcopy(tensor), model=model)

# smooth masks and remove holes
#ctv_nominal.smoothMask(BS)

# Set the parameters for the normal distribution

std_dev = 0.5  # Small standard deviation to ensure numbers are close to 0
num_samples = 100

# Generate the samples

#samples_theta_x = np.random.normal(loc=0, scale=std_dev, size=num_samples)
#samples_theta_y = np.random.normal(loc=0, scale=std_dev, size=num_samples)
#samples_theta_z = np.random.normal(loc=0, scale=std_dev, size=num_samples)
#samples_scaling_x = np.random.normal(loc=1, scale=std_dev, size=num_samples)
#samples_scaling_y = np.random.normal(loc=1, scale=std_dev, size=num_samples)
#samples_scaling_z = np.random.normal(loc=1, scale=std_dev, size=num_samples)
#samples_shearing_xy = np.random.normal(loc=0, scale=std_dev, size=num_samples)
#samples_shearing_xz = np.random.normal(loc=0, scale=std_dev, size=num_samples)
#samples_shearing_yz = np.random.normal(loc=0, scale=std_dev, size=num_samples)
samples_translation_x = np.random.normal(loc=0, scale=std_dev, size=num_samples)
samples_translation_y = np.random.normal(loc=0, scale=std_dev, size=num_samples)
samples_translation_z = np.random.normal(loc=0, scale=std_dev, size=num_samples)

x_max = np.max(np.abs(samples_translation_x))
y_max = np.max(np.abs(samples_translation_y))
z_max = np.max(np.abs(samples_translation_z))

u_max = x_max/np.sqrt(model['resistance'])
v_max = y_max/np.sqrt(model['resistance'])
w_max = z_max/np.sqrt(model['resistance'])

CTV_ubound = ctv_nominal.copy()
CTV_lbound = ctv_nominal.copy()
CTV_ubound.dilateMask(radius=[u_max, v_max, w_max])
CTV_lbound.erodeMask(radius=[u_max, v_max, w_max])
CTV_ubound.imageArray[BS] = False
CTV_lbound.imageArray[BS] = False

GTVs = np.zeros(GTV.shape + (num_samples,))
CTVs = np.zeros(GTV.shape + (num_samples,))
Distances = np.zeros(GTV.shape + (num_samples,))
for i in range(num_samples):

    print(f"Sample {i+1}")

    # create affine matrix

    # theta_x = samples_theta_x[i] * np.pi / 180
    # theta_y = samples_theta_y[i] * np.pi / 180
    # theta_z = samples_theta_z[i] * np.pi / 180
    # rotation_x = np.array([[1, 0, 0],
    #                        [0, np.cos(theta_x), -np.sin(theta_x)],
    #                        [0, np.sin(theta_x), np.cos(theta_x)]])
    # rotation_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
    #                        [0, 1, 0],
    #                        [-np.sin(theta_y), 0, np.cos(theta_y)]])
    # rotation_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
    #                        [np.sin(theta_z), np.cos(theta_z), 0],
    #                        [0, 0, 1]])
    # scaling_x = samples_scaling_x[i]
    # scaling_y = samples_scaling_y[i]
    # scaling_z = samples_scaling_z[i]
    # scale = np.array([[scaling_x, 0, 0],
    #                   [0, scaling_y, 0],
    #                   [0, 0, scaling_z]])
    #
    # shearing_factor_xy = samples_shearing_xy[i]
    # shearing_factor_xz = samples_shearing_xz[i]
    # shearing_factor_yz = samples_shearing_yz[i]
    # shear = np.array([[1, shearing_factor_xy, shearing_factor_xz],
    #                   [shearing_factor_xy, 1, shearing_factor_yz],
    #                 [shearing_factor_xz, shearing_factor_yz, 1]])
    #
    # Deformation = np.linalg.multi_dot([rotation_x, rotation_y, rotation_z, scale, shear])

    Deformation = np.eye(3)
    Translation = np.array([samples_translation_x[i], samples_translation_y[i], samples_translation_z[i]])

    delta_com = COM - np.dot(Deformation, COM) + Translation

    affine = np.identity(4)
    affine[0:3, 0:3] = Deformation
    affine[0:3, 3] = delta_com

    # replace existing GTV
    RTs.setMask('GTV', affine_transform(GTV.copy(), np.linalg.inv(affine)), spacing=voxel_size)
    #RTs.smoothMasks(['GTV'])

    # Run fast marching method

    ctv = CTVGeometric()
    #ctv.setCTV_volume(volume, RTs, tensor=copy.deepcopy(tensor), model=model, x0=margin)
    ctv.setCTV_isodistance(margin, RTs, tensor=copy.deepcopy(tensor), model=model)
    #ctv.smoothMask(BS)

    # store masks
    GTVs[..., i] = RTs.getMaskByName('GTV').imageArray
    CTVs[..., i] = ctv.imageArray
    Distances[..., i] = ctv.distance3D

# compute mean and variance
GTV_mean = np.mean(GTVs, axis=-1)
GTV_variance = np.var(GTVs, axis=-1)
CTV_mean = np.mean(CTVs, axis=-1)
CTV_variance = np.var(CTVs, axis=-1)
Distance_variance = np.var(Distances, axis=-1)

# Create 2D plots

X_coord = int(COM[0])
Y_coord = int(COM[1])
Z_coord = int(COM[2])

plotMR = MRI.imageArray.copy()
plotMR[~External] = 0
plotDistance_variance = Distance_variance.copy()
plotDistance_variance[~External] = np.nan

plotGTV_mean_mask = GTV_mean > 0
plotGTV_mean_norm = (GTV_mean - np.min(GTV_mean)) / (np.max(GTV_mean) - np.min(GTV_mean))
plotGTV_variance_mask = GTV_variance > 0
plotGTV_variance_norm = (GTV_variance - np.min(GTV_variance)) / (np.max(GTV_variance) - np.min(GTV_variance))

plotCTV_mean_mask = CTV_mean > 0
plotCTV_mean_norm = (CTV_mean - np.min(CTV_mean)) / (np.max(CTV_mean) - np.min(CTV_mean))
plotCTV_variance_mask = CTV_variance > 0
plotCTV_variance_norm = (CTV_variance - np.min(CTV_variance)) / (np.max(CTV_variance) - np.min(CTV_variance))

red_colormap = LinearSegmentedColormap.from_list('red_colormap', [(1, 1, 1), (1, 0, 0)])
yellow_colormap = LinearSegmentedColormap.from_list('yellow_colormap', [(1, 1, 1), (1, 1, 0)])

plt.figure()
plt.imshow(np.flip(plotMR[:, :, Z_coord].transpose(), axis=0), cmap='gray')
plt.contour(np.flip(GTV[:, :, Z_coord].transpose(), axis=0), colors='red', linewidths=1)
plt.contour(np.flip(CC[:, :, Z_coord].transpose(), axis=0), colors='blue', linewidths=1)
# plt.contour(np.flip(ctv_nominal.imageArray[:, :, Z_coord].transpose(), axis=0), colors='yellow', linewidths=1)
plt.contour(np.flip(CTV_ubound.imageArray[:, :, Z_coord].transpose(), axis=0), colors='yellow', linestyles='dashed', linewidths=0.5)
plt.contour(np.flip(CTV_lbound.imageArray[:, :, Z_coord].transpose(), axis=0), colors='yellow', linestyles='dashed', linewidths=0.5)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

plt.imshow(
    np.flip(plotGTV_variance_mask[:, :, Z_coord].transpose(), axis=0),
    cmap=red_colormap,
    alpha=np.flip(plotGTV_variance_norm[:, :, Z_coord].transpose(), axis=0)
)

plt.imshow(
    np.flip(plotCTV_variance_mask[:, :, Z_coord].transpose(), axis=0),
    cmap=yellow_colormap,
    alpha=np.flip(plotCTV_variance_norm[:, :, Z_coord].transpose(), axis=0)
)
plt.show()

plt.figure()
plt.imshow(np.flip(plotMR[:, :, Z_coord].transpose(), axis=0), cmap='gray')
plt.contour(np.flip(GTV[:, :, Z_coord].transpose(), axis=0), colors='red', linewidths=1)
plt.contour(np.flip(CC[:, :, Z_coord].transpose(), axis=0), colors='blue', linewidths=1)
plt.contour(np.flip(ctv_nominal.imageArray[:, :, Z_coord].transpose(), axis=0), colors='yellow', linewidths=1)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

plt.imshow(
    np.flip(plotCTV_mean_mask[:, :, Z_coord].transpose(), axis=0),
    cmap=yellow_colormap,
    alpha=np.flip(plotCTV_mean_norm[:, :, Z_coord].transpose(), axis=0)
)
plt.show()

plt.figure()
plt.imshow(np.flip(plotMR[:, :, Z_coord].transpose(), axis=0), cmap='gray')
plt.contour(np.flip(GTV[:, :, Z_coord].transpose(), axis=0), colors='red', linewidths=1)
plt.contour(np.flip(CC[:, :, Z_coord].transpose(), axis=0), colors='blue', linewidths=1)
plt.contour(np.flip(ctv_nominal.imageArray[:, :, Z_coord].transpose(), axis=0), colors='yellow', linewidths=1)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

plt.imshow(
    np.flip(plotGTV_mean_mask[:, :, Z_coord].transpose(), axis=0),
    cmap=red_colormap,
    alpha=np.flip(plotGTV_mean_norm[:, :, Z_coord].transpose(), axis=0)
)

# plt.savefig(os.path.join(os.getcwd(),'CTV_'+modelDTI['model-DTI']+'.pdf'), format='pdf',bbox_inches='tight')
plt.show()

# plt.figure()
# plt.imshow(np.flip(plotMR[:, :, Z_coord].transpose(), axis=0), cmap='gray')
# maxplotvalue = 3
# plt.contourf(np.flip(np.minimum(plotDistance_variance, maxplotvalue)[:, :, Z_coord].transpose(), axis=0), 20, cmap='inferno')
# plt.colorbar()
#
# # plt.savefig(os.path.join(os.getcwd(),'CTV_'+modelDTI['model-DTI']+'.pdf'), format='pdf',bbox_inches='tight')
# plt.show()
