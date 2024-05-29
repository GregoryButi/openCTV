#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:04:17 2022

@author: gregory
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass as com
from dipy.io.image import load_nifti
import copy
from matplotlib.widgets import Slider

from opentps.core.data.images._image3D import Image3D
from Process.Tensors import TensorMetric
from Process.CTVs import CTVGeometric
from Process import Struct
from Analysis.contourComparison import dice_score, percentile_hausdorff_distance, jaccard_index, hausdorff_distance, mean_hausdorff_distance

# output
path_folder_output = '/home/gregory/Documents/Projects/CTV_RO1/Results/Glioma'

# Patient IDs
#PIDs = ['GLI_009_GBM', 'GLI_001_GBM', 'GLI_008_GBM', 'GLI_004_GBM', 'GLI_005_GBM', 'GLI_003_AAC', 'GLI_017_AAC', 'GLI_044_AC', 'GLI_046_AC', 'GLI_006_ODG']
#Margins = [20, 20, 20, 20, 20, 20, 20, 10, 10, 10]
PIDs = ['GLI_003_AAC'] # GLI_003_AAC
Margins = [20]

for PID, margin in  zip(PIDs, Margins):

    # input
    
    path_MRI = f'/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Glioma/GLIS-RT/{PID}/MNI152/T1_norm.nii.gz'
    path_RTstructs = f'/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Glioma/GLIS-RT/{PID}/MNI152/'
    path_tensor = f'/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Glioma/GLIS-RT/{PID}/MNI152/MT_deformable_PDR_DIPY.nii.gz'
    
    # load data
    
    MRI, grid2world, voxel_size = load_nifti(path_MRI, return_voxsize=True)
    
    tensor = TensorMetric()
    tensor.loadTensor(path_tensor, 'metric')
    
    # load structures
    
    RTs = Struct()
    RTs.loadContours_folder(path_RTstructs, ['GTV', 'CTV', 'BS', 'CC', 'Brain_mask', 'WM', 'GM', 'External'])
    
    RTs.smoothMasks(['GTV', 'CTV', 'CC', 'Brain_mask', 'External'])
    
    print('GTV volume: '+str(RTs.getMaskByName('GTV').imageArray.sum()*np.prod(RTs.getMaskByName('GTV').spacing)/1000)+' cc') 
    
    BS = np.logical_or(~RTs.getMaskByName('Brain_mask').imageArray, RTs.getMaskByName('BS').imageArray)
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
    Brain = RTs.getMaskByName('Brain_mask').imageArray
    CC = RTs.getMaskByName('CC').imageArray
    BS = RTs.getMaskByName('BS').imageArray
    WM = RTs.getMaskByName('WM').imageArray
    GM = RTs.getMaskByName('GM').imageArray
        
    # define model
    
    modelDTI = {
        'obstacle': True,
        'model': 'Anisotropic', # None, 'Nonuniform', 'Anisotropic'
        'model-DTI': 'Clatz', # 'Clatz', 'Rekik'
        'resistance': 0.1,
        'anisotropy': 1.0
        }
    
    # Run fast marching method for classic CTV
    
    ctv_classic = CTVGeometric()
    ctv_classic.setCTV_isodistance(margin, RTs, model = {'model':None, 'obstacle':True})
    volume = ctv_classic.getVolume()
    
    # smooth masks and remove holes
    ctv_classic.smoothMask(BS)
    
    # parameter space

    resistance = [0.5, 0.1, 0.05]
    anisotropy = [1.]
    
    # Initialize arrays for CTVs and evaluation metrics
    CTVs_dti = np.zeros(np.append(MRI.gridSize, [len(resistance),len(anisotropy)])).astype(bool)
    CTVs_wmgm = np.zeros(np.append(MRI.gridSize, [len(resistance)])).astype(bool)
    
    eval_metrics = [[{'DSC': np.nan, 'JI': np.nan, 'HD': np.nan, 'HD95': np.nan,'HD98': np.nan, 'HDmean': np.nan} for j in range(len(resistance))] for k in range(len(anisotropy))]
    eval_metrics_wmgm = [[{'DSC': np.nan, 'JI': np.nan, 'HD': np.nan, 'HD95': np.nan,'HD98': np.nan, 'HDmean': np.nan} for j in range(len(resistance))] for k in range(len(anisotropy))]
    
    # Iterate over resistance values
    for j in range(len(resistance)): 
                
        ctv_wmgm = CTVGeometric()
        ctv_wmgm.setCTV_volume(volume, RTs, tensor=copy.deepcopy(tensor), model = {'aniso': True, 'obstacle': True, 'model': 'Nonuniform', 'resistance': resistance[j]}, x0 = margin)
        ctv_wmgm.smoothMask(BS)
        
        # store mask in array
        CTVs_wmgm[:,:,:, j] = ctv_wmgm.imageArray
        
        # Iterate over parameter values
        for k in range(len(anisotropy)):
            
            # update model
            
            modelDTI['resistance'] = resistance[j]
            modelDTI['anisotropy'] = anisotropy[k]
            
            # Run fast marching method
            
            ctv_dti = CTVGeometric()
            ctv_dti.setCTV_volume(volume, RTs, tensor=copy.deepcopy(tensor), model = modelDTI, x0 = margin)
            ctv_dti.smoothMask(BS)
            
            # store mask in array
            CTVs_dti[:,:,:, j, k] = ctv_dti.imageArray
        
            # evaluate CTV against reference
            
            # exclude GTV from volume overlap analysis
            eval_metrics[k][j]['DSC'] = dice_score(np.logical_and(ctv_classic.imageArray,~GTV), np.logical_and(ctv_dti.imageArray,~GTV))
            eval_metrics_wmgm[k][j]['DSC'] = dice_score(np.logical_and(ctv_wmgm.imageArray,~GTV), np.logical_and(ctv_dti.imageArray,~GTV))
            
            eval_metrics[k][j]['JI'] = jaccard_index(ctv_classic.imageArray, ctv_dti.imageArray)
            eval_metrics_wmgm[k][j]['JI'] = jaccard_index(ctv_wmgm.imageArray, ctv_dti.imageArray)
            
            eval_metrics[k][j]['HD'] = hausdorff_distance(ctv_classic.getMeshpoints(), ctv_dti.getMeshpoints())
            eval_metrics_wmgm[k][j]['HD'] = hausdorff_distance(ctv_wmgm.getMeshpoints(), ctv_dti.getMeshpoints())
            
            eval_metrics[k][j]['HD95'] = percentile_hausdorff_distance(ctv_classic.getMeshpoints(), ctv_dti.getMeshpoints(), percentile=95)
            eval_metrics_wmgm[k][j]['HD95'] = percentile_hausdorff_distance(ctv_wmgm.getMeshpoints(), ctv_dti.getMeshpoints(), percentile=95)
            
            eval_metrics[k][j]['HD98'] = percentile_hausdorff_distance(ctv_classic.getMeshpoints(), ctv_dti.getMeshpoints(), percentile=98)
            eval_metrics_wmgm[k][j]['HD98'] = percentile_hausdorff_distance(ctv_wmgm.getMeshpoints(), ctv_dti.getMeshpoints(), percentile=98)
            
            eval_metrics[k][j]['HDmean'] = mean_hausdorff_distance(ctv_classic.getMeshpoints(), ctv_dti.getMeshpoints())
            eval_metrics_wmgm[k][j]['HDmean'] = mean_hausdorff_distance(ctv_wmgm.getMeshpoints(), ctv_dti.getMeshpoints())
        
    # save results
    
    # with open(os.path.join(os.path.join(path_folder_output,modelDTI['model-DTI']),'eval_metrics_'+PID+'.txt'), 'wb') as f:
    #     pickle.dump(eval_metrics,f) 
        
    # with open(os.path.join(os.path.join(path_folder_output,modelDTI['model-DTI']),'eval_metrics_wmgm_'+PID+'.txt'), 'wb') as f:
    #     pickle.dump(eval_metrics_wmgm,f)  
    
    # with open(os.path.join(os.path.join(path_folder_output,modelDTI['model-DTI']),'resistance_'+PID+'.txt'), 'wb') as f:
    #     pickle.dump(resistance,f)     
    
    # with open(os.path.join(os.path.join(path_folder_output,modelDTI['model-DTI']),'params_'+PID+'.txt'), 'wb') as f:
    #     pickle.dump(anisotropy,f)     
    
# Create 2D plots      

#_, _, maskZ_tmp = np.nonzero(CC)
#Z_coord = int(np.mean(maskZ_tmp))

# GTV COM for display
COM = np.array(com(GTV))
X_coord = int(COM[0])
Y_coord = int(COM[1])
Z_coord = int(COM[2])

plotMR = MRI.imageArray.copy()
plotMR[~External] = 0
plotCC = CC.astype(float).copy()
plotCC[~CC] = np.NaN
plotGTV = GTV.astype(float).copy()
plotGTV[~GTV] = np.NaN
ctv_dti.distance3D[BS] = np.NaN

#idx = [1,2,3]
idx = [0,1,2]

fig, axes = plt.subplots(1, len(idx))

# Remove spacing between subplots
fig.subplots_adjust(wspace=0.01)

i = 0
for j in idx:
    axes[i].imshow(np.flip(plotMR[:, :, Z_coord].transpose(), axis=0), cmap='gray', vmin=0, vmax=2.5)
    axes[i].contourf(np.flip(plotGTV[:, :, Z_coord].transpose(), axis=0), colors='yellow', alpha=0.4)
    axes[i].contourf(np.flip(plotCC[:, :, Z_coord].transpose(), axis=0), colors='blue', alpha=0.4)
    #axes[i].contour(np.flip(BS[:, :, Z_coord].transpose(), axis=0), colors='yellow', linewidths=1)
    #axes[i].imshow(np.flip(WM[:, :, Z_coord].transpose(), axis=0), alpha=0.5)

    axes[i].contour(np.flip(CTVs_wmgm[:, :, Z_coord, j].transpose(), axis=0), colors='red', linewidths=1.5)
    axes[i].contour(np.flip(ctv_classic.imageArray[:, :, Z_coord].transpose(), axis=0), colors='green', linewidths=1.5)
    
    axes[i].contour(np.flip(CTVs_dti[:, :, Z_coord, j, 0].transpose(), axis=0), colors='white', linewidths=1, linestyles='dashed', alpha=.75)
    axes[i].contour(np.flip(CTVs_dti[:, :, Z_coord, j, -1].transpose(), axis=0), colors='white', linewidths=1, linestyles='solid', alpha=.75)
    
    axes[i].text(0.01, 0.995, ''r'$\rho$ = '+str(round(resistance[j], 2)), transform=axes[i].transAxes, fontsize=8, color='white', ha='left', va='top')

    axes[i].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    
    i+=1
    
#plt.savefig(os.path.join(os.getcwd(),'CTV_'+modelDTI['model-DTI']+'.pdf'), format='pdf',bbox_inches='tight')

plt.show()
    
# Build interactive plot

# %matplotlib qt
# %matplotlib inline

def plot_isodistance(ax, Z, i, j, k):

    fig.add_axes(ax)
    plt.imshow(np.flip(plotMR[:, :, Z].transpose(), axis=0), cmap='gray')
    plt.clim(0,2.5)
    
    plt.contourf(np.flip(plotGTV[:, :, Z].transpose(), axis=0), colors='yellow', alpha=0.5)
    #plt.contour(np.flip(BS[:, :, Z].transpose(), axis=0), colors='yellow', linewidths=0.5)
    plt.contourf(np.flip(plotCC[:, :, Z].transpose(), axis=0), colors='blue', alpha=0.5)  
    
    #plt.contour(np.flip(ctv_wmgm.imageArray[:, :, Z].transpose(), axis=0), colors='red')
    #plt.contour(np.flip(ctv_classic.imageArray[:, :, Z].transpose(), axis=0), colors='blue')
    plt.contour(np.flip(CTVs_dti[:, :, Z, j, k].transpose(), axis=0), colors='white')
    
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

# Plot
fig = plt.figure()
plt.axis('off')
# plt.title('Interactive slider')

ax = fig.add_subplot(111)
plot_isodistance(ax, Z_coord, 0, 0, 0)

# Define sliders

# Make a vertically oriented slider to control the slice
# position x, position y, x-length, y-length
alpha_axis_1 = plt.axes([0.2, 0.2, 0.0125, 0.62])
alpha_slider_1 = Slider(
    ax=alpha_axis_1,
    label="Slice",
    valmin=0,
    valmax=MRI.gridSize[2],
    valinit=Z_coord,
    valstep=1,
    orientation="vertical"
)

# Make horizontal oriented slider to control the distance
alpha_axis_2 = plt.axes([0.3, 0.06, 0.4, 0.03])
alpha_slider_2 = Slider(
    ax=alpha_axis_2,
    label='Margin (mm)',
    valmin=0,
    valmax=0,
    valinit=0,
    valstep=1
)

# Make horizontal oriented slider to control the ''isotropicness''
alpha_axis_3 = plt.axes([0.3, 0.02, 0.4, 0.03])
alpha_slider_3 = Slider(
    ax=alpha_axis_3,
    label='Resistance',
    valmin=0,
    valmax=len(resistance)-1,
    valinit=0,
    valstep=1
)

# # Make horizontal oriented slider to control the ''isotropicness''
# alpha_axis_4 = plt.axes([0.3, 0.0, 0.4, 0.03])
# alpha_slider_4 = Slider(
#     ax=alpha_axis_4,
#     label='Anisotropy',
#     valmin=0,
#     valmax=len(anisotropy)-1,
#     valinit=0,
#     valstep=1
# )


def update(val):
    alpha1 = int(alpha_slider_1.val)
    alpha2 = int(alpha_slider_2.val)
    alpha3 = int(alpha_slider_3.val)
    #alpha4 = int(alpha_slider_4.val)

    ax.cla()
    #plot_isodistance(ax, alpha1, alpha2, alpha3, alpha4)
    plot_isodistance(ax, alpha1, alpha2, alpha3, 0)

    plt.draw()


alpha_slider_1.on_changed(update)
alpha_slider_2.on_changed(update)
alpha_slider_3.on_changed(update)
#alpha_slider_4.on_changed(update)

plt.show()
