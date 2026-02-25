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
from dipy.viz import regtools
from dipy.io.image import load_nifti
import scipy
import copy

from opentps.core.data.images._image3D import Image3D

from Process.CTVs import CTVGeometric
from Process.Transforms import TransformTensorDeformable
from Process.ImageRegistrationDIPY import ImageRegistrationDeformable
from Process import Struct

# input

path_CT = '/media/gregory/Elements/Data/MGH_SpineMets/Spine-RT_Processed/Spine_008_T12/Therapy-scan/MRI_CT/CT.nii.gz'
path_RTstructs_manual = '/media/gregory/Elements/Data/MGH_SpineMets/Spine-RT_Processed/Spine_008_T12/Therapy-scan/Structures_manual'
path_RTstructs_DL = '/media/gregory/Elements/Data/MGH_SpineMets/Spine-RT_Processed/Spine_008_T12/Therapy-scan/Structures_DL'
path_atlas = '../../Input/Atlas/MNI152_T1_1mm_brain_norm.nii.gz'

# load data

CT, static_grid2world, voxel_size = load_nifti(path_CT, return_voxsize=True)
origin = static_grid2world[0:3, 3]
template, moving_grid2world = load_nifti(path_atlas)

# load structures

RTs = Struct()
RTs.loadContours_folder(path_RTstructs_manual, ['GTV_T12', 'CTV_T12', 'SpinalCord'], contour_names=['GTV', 'CTV', 'SpinalCord'], contour_types=['GTV', None, None])
RTs.loadContours_folder(path_RTstructs_DL, ['T12', 'L1', 'T11', 'External'], contour_names=['Vertebra', 'Vertebra_below', 'Vertebra_above', 'External'], contour_types=[None, None, None, None])

# define barrier structures
#Vertebrae = np.logical_or(np.logical_or(RTs.getMaskByName('Vertebra').imageArray, RTs.getMaskByName('Vertebra_below').imageArray), RTs.getMaskByName('Vertebra_above').imageArray)
#RTs.setMask('BS', ~Vertebrae, spacing=voxel_size, origin=origin, roi_type='Barrier')
RTs.setMask('BS', ~RTs.getMaskByName('Vertebra').imageArray, spacing=voxel_size, origin=origin, roi_type='Barrier')

# reduce calculation grid of images and structures

external = RTs.getMaskByName('External').imageArray
CT = Image3D(imageArray=CT, spacing=voxel_size, origin=origin)

BB = RTs.getBoundingBox('Vertebra', margin=10)

CT.reduceGrid_mask(external)
RTs.reduceGrid_mask(external)

# reload contour masks

gtv = RTs.getMaskByName('GTV').imageArray
external = RTs.getMaskByName('External').imageArray

#SpinalCord_cropped = np.logical_and(BB.imageArray, RTs.getMaskByName('SpinalCord').imageArray)

X_world, Y_world, Z_world = CT.getMeshGridPositions()

################
################
################

principal_axis = RTs.principle_comps('SpinalCord')

# make z vector positive if necessary
if principal_axis[2] < 0:
    principal_axis = -principal_axis

target_axis = np.array([0, 0, 1])

# Normalize vectors
v = principal_axis / np.linalg.norm(principal_axis)
t = target_axis

# Compute rotation axis (cross-product)
axis = np.cross(v, t)
axis_norm = np.linalg.norm(axis)

if axis_norm < 1e-8:
    # Already aligned
    R = np.eye(3)
else:
    axis = axis / axis_norm

    # Angle between them
    angle = np.arccos(np.clip(np.dot(v, t), -1.0, 1.0))

    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    rotation = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)

rotation_point = np.array(com(RTs.getMaskByName('SpinalCord').imageArray))

# 1. Translate rotation_point to origin
T1 = np.eye(4)
T1[:3, 3] = -rotation_point

# 2. Perform rotation
R = np.eye(4)
R[:3, :3] = rotation

# 3. Translate back
T2 = np.eye(4)
T2[:3, 3] = rotation_point

# Combine matrices: T2 * R * T1
# The order is crucial: translate to origin, rotate, translate back
T = T2 @ R @ T1

# initialize
CT_transformed = CT.copy()
RTs_transformed = copy.deepcopy(RTs)

# transform image and structures and grid
CT_transformed.imageArray = scipy.ndimage.affine_transform(CT.imageArray.astype(float), np.linalg.inv(T), output_shape=CT.gridSize, cval=-1000., mode='grid-constant', order=3)
RTs_transformed.transformMasksAffineSCIPY(T)

# Flatten the meshgrid arrays into (N,) vectors
coords = np.vstack([
    X_world.ravel(),
    Y_world.ravel(),
    Z_world.ravel(),
    np.ones(X_world.size)         # homogeneous coordinates
])  # shape (4, N)

# Apply the 4×4 transformation
transformed = T @ coords   # shape (4, N)

# Extract x,y,z
Xt_world = transformed[0, :].reshape(X_world.shape)
Yt_world = transformed[1, :].reshape(Y_world.shape)
Zt_world = transformed[2, :].reshape(Z_world.shape)

####################
####################
####################

# create pie slices by estimating the angles (start - stop) in the plane around the spinal coord as origin at every slice
sc = RTs_transformed.getMaskByName('SpinalCord').imageArray

radius = 100

offset = 13.8
angle_sector = [153.7, 18.0, 59.5, 49.9, 61.1, 17.8]

# Build phi list
phi_1 = -offset - angle_sector[0]
phi_2 = -offset
phi_3 = -offset + angle_sector[1]
phi_4 = -offset + angle_sector[1] + angle_sector[2]
phi_5 = -offset + angle_sector[1] + angle_sector[2] + angle_sector[3]
phi_6 = -offset + angle_sector[1] + angle_sector[2] + angle_sector[3] + angle_sector[4]

phis = np.array([phi_1, phi_2, phi_3, phi_4, phi_5, phi_6])

alphas = [0.25, 0.4, 0.55, 0.7, 0.85, 1.]

_, _, idZ_all = np.where(sc)
idZ_unique = np.unique(idZ_all)

sectors = [np.zeros(CT.gridSize).astype(bool) for _ in range(len(phis))]

for idZ in idZ_unique:

    com_slice = np.array(com(sc[:, :, idZ])).astype(int)

    xt_slice = Xt_world[com_slice[0], com_slice[1], idZ]
    yt_slice = Yt_world[com_slice[0], com_slice[1], idZ]

    X_world_slice = Xt_world[:, :, idZ] - xt_slice
    Y_world_slice = Yt_world[:, :, idZ] - yt_slice

    radial_coordinates = np.sqrt(X_world_slice ** 2 + Y_world_slice ** 2)
    angular_coordinates = np.arctan2(Y_world_slice, X_world_slice) * 180 / np.pi

    for s, sector in enumerate(sectors):

        phi_start = phis[s]
        if s < len(phis) - 1:
            phi_stop = phis[s + 1]
            sector[:, :, idZ] = np.logical_and(radial_coordinates <= radius, np.logical_and(angular_coordinates > phi_start, angular_coordinates <= phi_stop))
        else:
            phi_stop = phis[0]
            sector[:, :, idZ] = np.logical_and(radial_coordinates <= radius, np.logical_or(angular_coordinates > phi_start, angular_coordinates <= phi_stop))

# add sectors to RTstruct
for s, sector in enumerate(sectors):
    RTs_transformed.setMask(f'Sector{s}', sector, spacing=voxel_size, origin=origin, roi_type=None)

def get_Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])

rotation_point = np.array(com(RTs_transformed.getMaskByName('SpinalCord').imageArray))

# 1. Translate rotation_point to origin
T1 = np.eye(4)
T1[:3, 3] = -rotation_point

# 2. Perform rotation
theta = 0*np.pi/180
Rz = get_Rz(theta)

# 3. Translate back
T2 = np.eye(4)
T2[:3, 3] = rotation_point

# Combine matrices: T2 * R * T1
# The order is crucial: translate to origin, rotate, translate back
T = T2 @ Rz @ T1

# transform the sectors
for s in range(len(sectors)):
    RTs_transformed.transformMasksAffineSCIPY(T, name=f'Sector{s}')

####################
####################
####################

# GTV-to-CTV margin
margin = 0

# Run fast marching method for classic CTV

CTV = CTVGeometric(rts=RTs_transformed, model={'model': None, 'obstacle': True})
CTV.setCTV_isodistance(margin)
#CTV.smoothMask()
    
# Create 2D plots

gtv_transformed = RTs_transformed.getMaskByName('GTV').imageArray
sc_transformed = RTs_transformed.getMaskByName('SpinalCord').imageArray

# voxel display
COM = np.array(com(gtv_transformed))
X_coord, Y_coord, Z_coord = int(COM[0]), int(COM[1]), int(COM[2])
plotX_world = Xt_world
plotY_world = Yt_world
plotZ_world = Zt_world

# prepare figures
plotCT = CT_transformed.imageArray.copy()
plotGTV = gtv_transformed.astype(float).copy()
plotGTV[~gtv_transformed] = np.NaN
plotSC = sc_transformed.astype(float).copy()
plotSC[~sc_transformed] = np.NaN
plotBarrier = RTs_transformed.getMaskByName('BS').imageArray
plotCTVmanual = RTs_transformed.getMaskByName('CTV').imageArray

plotSectors = [RTs_transformed.getMaskByName(f'Sector{s}').imageArray.astype(float).copy() for s in range(len(sectors))]
plotSectorsf = [RTs_transformed.getMaskByName(f'Sector{s}').imageArray.astype(float).copy() for s in range(len(sectors))]
for plotSectorf, sector in zip(plotSectorsf, sectors):
    plotSectorf[~sector] = np.NaN

for plotSectorf in plotSectorsf:
    plotSectorf[~(plotSectorf.astype(bool))] = np.NaN

# Build interactive plot

def plot_isodistance_sagittal(ax, X, plotCTV):

    y = plotY_world[X, :, 0]
    z = plotZ_world[X, 0, :]

    fig.add_axes(ax)
    ax.imshow(np.flip(plotCT[X, :, :].T, axis=0), extent=[y[0], y[-1], z[0], z[-1]], cmap='gray', vmin=-200, vmax=200)
    #ax.contourf(plotGTV[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='red', alpha=0.5)
    #ax.contour(plotCTVmanual[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='yellow', linewidths=1.5)

    ax.contour(plotCTV[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='green', linewidths=1.5)

    ax.contourf(plotSC[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='blue', alpha=0.2)
    ax.contour(plotBarrier[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='blue', linewidths=1.5)

    for alpha, plotSector in zip(alphas, plotSectorsf):
        ax.contourf(plotSector[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='orange', alpha=alpha)

    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

def plot_isodistance_coronal(ax, Y, plotCTV):

    x = plotX_world[:, Y, 0]
    z = plotZ_world[0, Y, :]

    fig.add_axes(ax)
    ax.imshow(np.flip(plotCT[:, Y, :].T, axis=0), extent=[x[0], x[-1], z[0], z[-1]], cmap='gray', vmin=-200, vmax=200)
    ax.contourf(plotGTV[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='red', alpha=0.5)
    #ax.contour(plotCTVmanual[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='yellow', linewidths=1.5)

    ax.contour(plotCTV[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='green', linewidths=1.5)

    ax.contourf(plotSC[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='blue', alpha=0.2)
    ax.contour(plotBarrier[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='blue', linewidths=1.5)

    for alpha, plotSector in zip(alphas, plotSectorsf):
        ax.contourf(plotSector[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='orange', alpha=0.5)

    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

def plot_isodistance_axial(ax, Z, plotCTV):

    x = plotX_world[:, 0, Z]
    y = plotY_world[0, :, Z]

    fig.add_axes(ax)
    ax.imshow(plotCT[:, :, Z].T, extent=[x[0], x[-1], y[0], y[-1]], cmap='gray', vmin=-200, vmax=200)
    ax.contourf(np.flip(plotGTV[:, :, Z].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='red', alpha=0.5)
    ax.contour(np.flip(plotCTVmanual[:, :, Z].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='yellow', linewidths=1.5)

    ax.contour(np.flip(plotCTV[:, :, Z].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='green', linewidths=1.5)

    ax.contourf(np.flip(plotSC[:, :, Z].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='blue', alpha=0.2)
    ax.contour(np.flip(plotBarrier[:, :, Z].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='blue', linewidths=1.5)

    for alpha, plotSector in zip(alphas, plotSectors):
        #ax.contourf(np.flip(plotSector[:, :, Z].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='orange', alpha=alpha)
        ax.contour(np.flip(plotSector[:, :, Z].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='orange', linewidths=1.5)

    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

# Plot
fig = plt.figure()
plt.axis('off')
# plt.title('Interactive slider')

ax1 = fig.add_subplot(131)
plot_isodistance_sagittal(ax1, X_coord, CTV.imageArray)
ax2 = fig.add_subplot(132)
plot_isodistance_coronal(ax2, Y_coord, CTV.imageArray)
ax3 = fig.add_subplot(133)
plot_isodistance_axial(ax3, Z_coord, CTV.imageArray)

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

# Make horizontal oriented slider to control the margin
alpha_axis_4 = plt.axes([0.3, 0.02, 0.4, 0.03])
alpha_slider_4 = Slider(
    ax=alpha_axis_4,
    label='Margin',
    valmin=0,
    valmax=20,
    valinit=0,
    valstep=1
)

def update(val):

    alpha1 = int(alpha_slider_1.val)
    alpha2 = int(alpha_slider_2.val)
    alpha3 = int(alpha_slider_3.val)
    alpha4 = int(alpha_slider_4.val)

    CTV.setCTV_isodistance(alpha4)

    ax1.cla()
    plot_isodistance_sagittal(ax1, alpha1, CTV.imageArray)
    ax2.cla()
    plot_isodistance_coronal(ax2, alpha2, CTV.imageArray)
    ax3.cla()
    plot_isodistance_axial(ax3, alpha3, CTV.imageArray)

    plt.draw()

alpha_slider_1.on_changed(update)
alpha_slider_2.on_changed(update)
alpha_slider_3.on_changed(update)
alpha_slider_4.on_changed(update)

plt.show()












