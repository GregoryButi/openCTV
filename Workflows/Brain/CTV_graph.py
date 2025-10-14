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
from Process.CTVs import CTVGeometric
from Process import Struct

# input

path_CT = '/media/gregory/Elements/Data/MGH_Glioma/GLIS-RT_Processed/GLI_004_GBM/Therapy-scan/MRI_CT/CT.nii.gz'
path_RTstructs = '/media/gregory/Elements/Data/MGH_Glioma/GLIS-RT_Processed/GLI_004_GBM/Therapy-scan/Structures'

# load data

ct, static_grid2world, voxel_size = load_nifti(path_CT, return_voxsize=True)
CT = Image3D(imageArray=ct, spacing=voxel_size)

# load structures

RTs = Struct()
# load manual contours
RTs.loadContours_folder(path_RTstructs, ['Brain', 'Brainstem', 'Cerebellum', 'Midline', 'Ventricles_connected', 'GTV'], contour_types=[None, None, None, None, None, 'GTV'])

GTV = RTs.getMaskByName('GTV').imageArray
Brain = RTs.getMaskByName('Brain').imageArray
Brainstem = RTs.getMaskByName('Brainstem').imageArray
Cerebellum = RTs.getMaskByName('Cerebellum').imageArray
Midline = RTs.getMaskByName('Midline').imageArray
Ventricles = RTs.getMaskByName('Ventricles_connected').imageArray

# define barrier structures

BS = np.logical_or((Brain + Brainstem + Cerebellum) == 0, (Midline + Ventricles) > 0)
RTs.setMask('BS', BS, spacing=voxel_size, roi_type='Barrier')

# define structure of preferred spread
RTs.setMask('PS', ~BS, spacing=voxel_size, roi_type='Barrier_soft')

margin = 15

# define calculation domain
GTVBox = RTs.getBoundingBox('GTV', margin+20)

# Run fast marching method

ctv_graph = CTVGeometric(rts=RTs, model={'model': 'Nonuniform', 'obstacle': True, 'resistance': 1.0})
ctv_graph.setCTV_isodistance(margin, solver='Dijkstra', domain=GTVBox)

ctv = CTVGeometric(rts=RTs, model={'model': 'Nonuniform', 'obstacle': True, 'resistance': 1.0})
ctv.setCTV_isodistance(margin, solver='FMM', domain=GTVBox)

# Create 2D plots

# voxel display
COM = np.array(com(GTV))
X_coord = int(COM[0])
Y_coord = int(COM[1])
Z_coord = int(COM[2])

x, y, z = CT.getMeshGridAxes()

# prepare figures
plotCT = ct.copy()
plotGTV = GTV.astype(float).copy()
plotGTV[~GTV] = np.NaN
plotBrain = Brain.astype(float).copy()
plotBrainstem = Brainstem.astype(float).copy()
plotCerebellum = Cerebellum.astype(float).copy()
plotVentricles = Ventricles.astype(float).copy()

# Build interactive plot

def plot_isodistance_sagittal(ax, X, plotCTV1, plotCTV2):

    fig.add_axes(ax)
    plt.imshow(np.flip(plotCT[X, :, :].transpose(), axis=0), extent=[y[0], y[-1], z[0], z[-1]], cmap='gray') #, vmin=0, vmax=2.5)
    plt.contourf(plotGTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='yellow', alpha=0.5)
    plt.contour(plotBrain[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='green', linewidths=0.5)
    plt.contour(plotBrainstem[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='red', linewidths=0.5)
    plt.contour(plotCerebellum[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='blue', linewidths=0.5)
    plt.contour(plotVentricles[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='orange', linewidths=0.5)
    plt.contour(plotCTV1[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='white', linewidths=1.5)
    plt.contour(plotCTV2[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')

    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

def plot_isodistance_coronal(ax, Y, plotCTV1, plotCTV2):

    fig.add_axes(ax)
    plt.imshow(np.flip(plotCT[:, Y, :].transpose(), axis=0), extent=[x[0], x[-1], z[0], z[-1]], cmap='gray') #, vmin=0, vmax=2.5)
    plt.contourf(plotGTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='yellow', alpha=0.5)
    plt.contour(plotBrain[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='green', linewidths=0.5)
    plt.contour(plotBrainstem[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='red', linewidths=0.5)
    plt.contour(plotCerebellum[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='blue', linewidths=0.5)
    plt.contour(plotVentricles[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='orange', linewidths=0.5)
    plt.contour(plotCTV1[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='white', linewidths=1.5)
    plt.contour(plotCTV2[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')

    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

def plot_isodistance_axial(ax, Z, plotCTV1, plotCTV2):

    fig.add_axes(ax)
    plt.imshow(np.flip(plotCT[:, :, Z].transpose(), axis=0), extent=[x[0], x[-1], y[0], y[-1]], cmap='gray') #, vmin=0, vmax=2.5)
    plt.contourf(plotGTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='yellow', alpha=0.5)
    plt.contour(plotBrain[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='green', linewidths=0.5)
    plt.contour(plotBrainstem[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', linewidths=0.5)
    plt.contour(plotCerebellum[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='blue', linewidths=0.5)
    plt.contour(plotVentricles[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='orange', linewidths=0.5)
    plt.contour(plotCTV1[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='white', linewidths=1.5)
    plt.contour(plotCTV2[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', linewidths=1.5, linestyles='dashed')

    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

# Plot
fig = plt.figure()
plt.axis('off')
# plt.title('Interactive slider')

ax1 = fig.add_subplot(131)
plot_isodistance_sagittal(ax1, X_coord, ctv.imageArray, ctv_graph.imageArray)
ax2 = fig.add_subplot(132)
plot_isodistance_coronal(ax2, Y_coord, ctv.imageArray, ctv_graph.imageArray)
ax3 = fig.add_subplot(133)
plot_isodistance_axial(ax3, Z_coord, ctv.imageArray, ctv_graph.imageArray)

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

def update(val):
    global previous_volume

    alpha1 = int(alpha_slider_1.val)
    alpha2 = int(alpha_slider_2.val)
    alpha3 = int(alpha_slider_3.val)
    alpha4 = int(alpha_slider_4.val)

    ctv.setCTV_isodistance(alpha4)
    ctv_graph.setCTV_isodistance(alpha4)

    ax1.cla()
    plot_isodistance_sagittal(ax1, alpha1, ctv.imageArray, ctv_graph.imageArray)
    ax2.cla()
    plot_isodistance_coronal(ax2, alpha2, ctv.imageArray, ctv_graph.imageArray)
    ax3.cla()
    plot_isodistance_axial(ax3, alpha3, ctv.imageArray, ctv_graph.imageArray)

    plt.draw()

alpha_slider_1.on_changed(update)
alpha_slider_2.on_changed(update)
alpha_slider_3.on_changed(update)
alpha_slider_4.on_changed(update)

plt.show()
