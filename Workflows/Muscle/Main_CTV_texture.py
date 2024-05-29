import os
import numpy as np
import nibabel as nib
from dipy.io.image import load_nifti
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import statistics

from opentps.core.data.images._image3D import Image3D
from Process.Tensors import TensorMetric
from Process.CTVs import CTVGeometric
from Process import Struct
from Analysis.contourComparison import dice_score, percentile_hausdorff_distance

#PIDs = ['SAC_001','SAC_002','SAC_003','SAC_004', 'SAC_005', 'SAC_007', 'SAC_009', 'SAC_010', 'SAC_011', 'SAC_011','SAC_012', 'SAC_013', 'SAC_014', 'SAC_016', 'SAC_017', 'SAC_018', 'SAC_019', 'SAC_020', 'SAC_025']
PIDs = ['SAC_025']

# model parameters

margin_tissue = np.array([10, 10, 10])
margin_muscle = np.array([15, 20, 30])
resistance = (margin_tissue/margin_muscle)**2

eval_metrics_Ref2VHM = [[{'DSC': np.nan, 'HD95': np.nan, 'deltaV': np.nan} for j in range(len(resistance))] for k in range(len(PIDs))]
eval_metrics_Ref2VHF = [[{'DSC': np.nan, 'HD95': np.nan, 'deltaV': np.nan} for j in range(len(resistance))] for k in range(len(PIDs))]
eval_metrics_VHM2VHF = [[{'DSC': np.nan, 'HD95': np.nan} for j in range(len(resistance))] for k in range(len(PIDs))]
k = 0
for PID in PIDs:

    # input
    path_CT = f'/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Sarcoma/SAC-RT/{PID}/Therapy-scan/CT_norm.nii.gz'
    path_tensor_VHM = f'/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Sarcoma/SAC-RT/{PID}/tensor_warped_VHM.nii.gz'
    path_tensor_VHF = f'/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Sarcoma/SAC-RT/{PID}/tensor_warped_VHF.nii.gz'
    path_RTstructs = f'/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Sarcoma/SAC-RT/{PID}/Structures/'
    path_barriers = f'/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Sarcoma/SAC-RT/{PID}/Structures/TotalSegmentator/'
         
    CT, grid2world, voxel_size = load_nifti(path_CT, return_voxsize=True)
    tensor_warped_VHF = nib.load(path_tensor_VHF).get_fdata()
    tensor_warped_VHM = nib.load(path_tensor_VHM).get_fdata()
    label_map = nib.load(os.path.join(path_RTstructs, 'Label_map.nii.gz')).get_fdata()
        
    # set structures
    RTs = Struct()
    RTs.loadContours_folder(path_RTstructs, ['CTV', 'External'])
    RTs.loadContours_folder(path_barriers, ['hip_left', 'hip_right'])
    
    RTs.setMask('GTV', label_map == 1, voxel_size)
    RTs.setMask('BS', ~RTs.getMaskByName('External').imageArray, voxel_size)
    RTs.setMask('Muscles', np.logical_and(label_map>1, label_map<12), voxel_size)
    
    # define structure of preferred spread
    RTs.setMask('PS', np.logical_and(RTs.getMaskByName('Muscles').imageArray, ~RTs.getMaskByName('BS').imageArray), voxel_size)
    # define barriers
    RTs.setMask('BS', np.logical_or(RTs.getMaskByName('hip_left').imageArray, RTs.getMaskByName('hip_right').imageArray), voxel_size)
    
    # define computation domain
    domain = np.logical_or(RTs.getMaskByName('Muscles').imageArray,RTs.getMaskByName('CTV').imageArray)
    
    # reduce images
    RTs.reduceGrid_mask(domain)
    ct = Image3D(imageArray=CT, spacing=voxel_size)
    ct.reduceGrid_mask(domain)
    
    tensor_VHF = TensorMetric(imageArray=tensor_warped_VHF, spacing=voxel_size)
    tensor_VHF.reduceGrid_mask(domain)
    
    tensor_VHM = TensorMetric(imageArray=tensor_warped_VHM, spacing=voxel_size)
    tensor_VHM.reduceGrid_mask(domain)
        
    # remove variables to clear space
    del CT, tensor_warped_VHF, tensor_warped_VHM, label_map, domain
    
    model = {'model': 'Anisotropic',
             'model-DTI': 'Rekik',
             'obstacle': True,
             'resistance': None,
             'anisotropy': None
             }
    
    CTVs_classic = np.zeros(tuple(ct.gridSize) + (len(resistance),)).astype(bool)
    CTVs_VHF = np.zeros(tuple(ct.gridSize) + (len(resistance),)).astype(bool)
    CTVs_VHM = np.zeros(tuple(ct.gridSize) + (len(resistance),)).astype(bool)
    Distance_classic = np.zeros(tuple(ct.gridSize) + (len(resistance),))
    for i in range(len(resistance)): 
            
        ctv_classic = CTVGeometric(spacing=voxel_size)
        ctv_classic.setCTV_isodistance(0., RTs, model = {'model':None, 'obstacle':True})
        
        ctv_classic_1 = CTVGeometric(spacing=voxel_size)
        ctv_classic_1.distance3D = ctv_classic.distance3D
        ctv_classic_1.setCTV_isodistance(margin_tissue[i])
        
        ctv_classic_2 = CTVGeometric(spacing=voxel_size)
        ctv_classic_2.distance3D = ctv_classic.distance3D
        ctv_classic_2.setCTV_isodistance(margin_muscle[i])
        
        # set margins outside muscle to 10 mm, and inside muscles to 15 mm
        ctv_classic.imageArray = ctv_classic_1.imageArray
        ctv_classic.imageArray[RTs.getMaskByName('PS').imageArray] = ctv_classic_2.imageArray[RTs.getMaskByName('PS').imageArray]
        ctv_classic.smoothMask(RTs.getMaskByName('BS').imageArray)
        volume = ctv_classic.getVolume()
        
        # compute the CTV volume outside of the muscles
        volume_inside_PS = np.logical_and(ctv_classic.imageArray, RTs.getMaskByName('PS').imageArray).sum()*np.array(voxel_size).prod()
        
        # store mask in array
        CTVs_classic[:,:,:,i] = ctv_classic.imageArray
        
        # store distance map in array
        Distance_classic[:,:,:,i] = ctv_classic.distance3D
        
        model['resistance'] = resistance[i]
        
        #######
        # VHF #
        #######
        
        ctv_VHF = CTVGeometric(spacing=voxel_size)
        ctv_VHF.setCTV_volume(volume, RTs, tensor=tensor_VHF, model=model, x0 = margin_tissue[i])
        ctv_VHF.smoothMask(RTs.getMaskByName('BS').imageArray)
        
        volume_fraction_reduction = (np.logical_and(ctv_VHF.imageArray, RTs.getMaskByName('PS').imageArray).sum()*ctv_VHF.spacing.prod() - volume_inside_PS)/volume*100   
        
        eval_metrics_Ref2VHF[k][i]['deltaV'] = volume_fraction_reduction
        
        print("Volume fraction reduction inside muscles: " + str(volume_fraction_reduction.round()) + "%")
    
        # store mask in array
        CTVs_VHF[:,:,:,i] = ctv_VHF.imageArray
        
        #######
        # VHM #
        #######
        
        ctv_VHM = CTVGeometric(spacing=voxel_size)
        ctv_VHM.setCTV_volume(volume, RTs, tensor=tensor_VHM, model=model, x0 = margin_tissue[i])
        ctv_VHM.smoothMask(RTs.getMaskByName('BS').imageArray)
        
        volume_fraction_reduction = (np.logical_and(ctv_VHM.imageArray, RTs.getMaskByName('PS').imageArray).sum()*ctv_VHM.spacing.prod() - volume_inside_PS)/volume*100   
        
        eval_metrics_Ref2VHM[k][i]['deltaV'] = volume_fraction_reduction
        
        print("Volume fraction reduction inside muscles: " + str(volume_fraction_reduction.round()) + "%")
    
        # store mask in array
        CTVs_VHM[:,:,:,i] = ctv_VHM.imageArray
        
        #################
        ## EVALUATION ###
        #################
        
        # exclude GTV from volume overlap analysis
        eval_metrics_Ref2VHM[k][i]['DSC'] = dice_score(np.logical_and(ctv_classic.imageArray,~RTs.getMaskByName('GTV').imageArray), np.logical_and(ctv_VHM.imageArray,~RTs.getMaskByName('GTV').imageArray))
        eval_metrics_Ref2VHF[k][i]['DSC'] = dice_score(np.logical_and(ctv_classic.imageArray,~RTs.getMaskByName('GTV').imageArray), np.logical_and(ctv_VHF.imageArray,~RTs.getMaskByName('GTV').imageArray))
        eval_metrics_VHM2VHF[k][i]['DSC'] = dice_score(np.logical_and(ctv_VHM.imageArray,~RTs.getMaskByName('GTV').imageArray), np.logical_and(ctv_VHF.imageArray,~RTs.getMaskByName('GTV').imageArray))

        eval_metrics_Ref2VHM[k][i]['HD95'] = percentile_hausdorff_distance(ctv_classic.getMeshpoints(), ctv_VHM.getMeshpoints(), percentile=95)
        eval_metrics_Ref2VHF[k][i]['HD95'] = percentile_hausdorff_distance(ctv_classic.getMeshpoints(), ctv_VHF.getMeshpoints(), percentile=95)
        eval_metrics_VHM2VHF[k][i]['HD95'] = percentile_hausdorff_distance(ctv_VHM.getMeshpoints(), ctv_VHF.getMeshpoints(), percentile=95)

    k += 1
        
    # remove variables to clear space
    del tensor_VHF, tensor_VHM
    
    # Plot results
    
    #X,Y,Z = com(RTs.getMaskByName('GTV').imageArray)
    # X = int(X)
    # Y = int(Y)
    # Z = int(Z)
    
    x, y, z = ct.getMeshGridAxes()
    
    plotGTV = RTs.getMaskByName('GTV').imageArray.astype(float).copy()
    plotGTV[~RTs.getMaskByName('GTV').imageArray] = np.NaN
    plotMuscles = RTs.getMaskByName('Muscles').imageArray.astype(float).copy()
    plotMuscles[~RTs.getMaskByName('Muscles').imageArray] = np.NaN
    
        
    fig1, axes1 = plt.subplots(1, len(resistance))
    fig1.subplots_adjust(wspace=0.01) # Remove spacing between subplots
    
    for i in range(len(resistance)):
        
        # Find the maximum value within the mask
        max_value = np.max(Distance_classic[:,:,:,i][np.logical_and(CTVs_VHM[:,:,:,i],CTVs_VHF[:,:,:,i])])
        indices = np.where(Distance_classic[:,:,:,i] == max_value)
        
        # Extract X, Y, and Z indices
        X, Y, Z = indices
        X = X[0]
        Y = Y[0]
        Z = Z[0]
        
        axes1[i].imshow(np.transpose(ct.imageArray[:,:,Z]), cmap = 'gray', vmin=0.2, vmax=.8, extent=[x.min(), x.max(), y.min(), y.max()])
        axes1[i].contourf(x, y, np.flip(np.transpose(plotGTV[:,:,Z]), axis=0), colors='green', alpha=0.2)
        axes1[i].contourf(x, y, np.flip(np.transpose(plotMuscles[:,:,Z]), axis=0), colors='blue', alpha=0.2)
        axes1[i].contour(x, y, np.flip(np.transpose(CTVs_classic[:,:,Z,i]), axis=0), colors='yellow', linewidths=1)
        axes1[i].contour(x, y, np.flip(np.transpose(CTVs_VHM[:,:,Z,i]), axis=0), colors='red', linewidths=1)
        axes1[i].contour(x, y, np.flip(np.transpose(CTVs_VHF[:,:,Z,i]), axis=0), colors='white', linewidths=1, linestyles='dashed', alpha=.75)
        
        # Add lines with labels
        line_elements = [
            mlines.Line2D([0], [0], color='yellow', linewidth=2, label='Stand'),
            mlines.Line2D([0], [0], color='red', linewidth=2, label='VHM'),
            mlines.Line2D([0], [0], color='white', linewidth=2, linestyle='dashed', label='VHF')
            ]

        axes1[i].legend(handles=line_elements, loc='upper right', fontsize=6,  labelcolor='white', facecolor='none')
        
        axes1[i].text(0.01, 0.995, '('+str(margin_tissue[i])+' / '+str(margin_muscle[i])+') mm', transform=axes1[i].transAxes, fontsize=8, color='white', ha='left', va='top')
        axes1[i].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
   
    #plt.savefig(os.path.join(os.getcwd(),f'CTVs_axial_{PID}.pdf'), format='pdf',bbox_inches='tight')
    plt.show()
    
    fig2, axes2 = plt.subplots(1, len(resistance))
    fig2.subplots_adjust(wspace=0.01)
    
    for i in range(len(resistance)):
       
        # Find the maximum value within the mask
        max_value = np.max(Distance_classic[:,:,:,i][np.logical_and(CTVs_VHM[:,:,:,i],CTVs_VHF[:,:,:,i])])
        indices = np.where(Distance_classic[:,:,:,i] == max_value)
       
        # Extract X, Y, and Z indices
        X, Y, Z = indices
        X = X[0]
        Y = Y[0]
        Z = Z[0]     
   
        axes2[i].imshow(np.flip(np.transpose(ct.imageArray[:,Y,:]), axis=0), cmap = 'gray', vmin=0.2, vmax=.8, extent=[x.min(), x.max(), z.min(), z.max()])
        axes2[i].contourf(x, z, np.transpose(plotGTV[:,Y,:]), colors='green', alpha=0.2)
        axes2[i].contourf(x, z, np.transpose(plotMuscles[:,Y,:]), colors='blue', alpha=0.2)
        axes2[i].contour(x, z, np.transpose(CTVs_classic[:,Y,:,i]), colors='yellow', linewidths=1)
        axes2[i].contour(x, z, np.transpose(CTVs_VHM[:,Y,:,i]), colors='red', linewidths=1)
        axes2[i].contour(x, z, np.transpose(CTVs_VHF[:,Y,:,i]), colors='white', linewidths=1, linestyles='dashed', alpha=.75)
        
        # Add lines with labels
        line_elements = [
            mlines.Line2D([0], [0], color='yellow', linewidth=2, label='Stand'),
            mlines.Line2D([0], [0], color='red', linewidth=2, label='VHM'),
            mlines.Line2D([0], [0], color='white', linewidth=2, linestyle='dashed', label='VHF')
            ]

        axes2[i].legend(handles=line_elements, loc='lower right', fontsize=6,  labelcolor='white', facecolor='none')

        
        axes2[i].text(0.01, 0.995, '('+str(margin_tissue[i])+' / '+str(margin_muscle[i])+') mm', transform=axes2[i].transAxes, fontsize=8, color='white', ha='left', va='top')
        axes2[i].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    
    #plt.savefig(os.path.join(os.getcwd(),f'CTVs_coronal_{PID}.pdf'), format='pdf',bbox_inches='tight')
    plt.show()
    
    fig3, axes3 = plt.subplots(1, len(resistance))
    fig3.subplots_adjust(wspace=0.01)
    
    for i in range(len(resistance)):
        
        # Find the maximum value within the mask
        max_value = np.max(Distance_classic[:,:,:,i][np.logical_and(CTVs_VHM[:,:,:,i],CTVs_VHF[:,:,:,i])])
        indices = np.where(Distance_classic[:,:,:,i] == max_value)
        
        # Extract X, Y, and Z indices
        X, Y, Z = indices
        X = X[0]
        Y = Y[0]
        Z = Z[0]   
        
        axes3[i].imshow(np.flip(np.transpose(ct.imageArray[X,:,:]), axis=0), cmap = 'gray', vmin=0.2, vmax=.8, extent=[y.min(), y.max(), z.min(), z.max()])
        axes3[i].contourf(y, z, np.transpose(plotGTV[X,:,:]), colors='green', alpha=0.2)
        axes3[i].contourf(y, z, np.transpose(plotMuscles[X,:,:]), colors='blue', alpha=0.2)
        axes3[i].contour(y, z, np.transpose(CTVs_classic[X,:,:,i]), colors='yellow', linewidths=1)
        axes3[i].contour(y, z, np.transpose(CTVs_VHM[X,:,:,i]), colors='red', linewidths=1)
        axes3[i].contour(y, z, np.transpose(CTVs_VHF[X,:,:,i]), colors='white', linewidths=1, linestyles='dashed', alpha=.75)
        
        # Add lines with labels
        line_elements = [
            mlines.Line2D([0], [0], color='yellow', linewidth=2, label='Stand'),
            mlines.Line2D([0], [0], color='red', linewidth=2, label='VHM'),
            mlines.Line2D([0], [0], color='white', linewidth=2, linestyle='dashed', label='VHF')
            ]

        axes3[i].legend(handles=line_elements, loc='upper right', fontsize=6,  labelcolor='white', facecolor='none')

        
        axes3[i].text(0.01, 0.995, '('+str(margin_tissue[i])+' / '+str(margin_muscle[i])+') mm', transform=axes3[i].transAxes, fontsize=8, color='white', ha='left', va='top')
        axes3[i].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
     
    #plt.savefig(os.path.join(os.getcwd(),f'CTVs_sagittal_{PID}.pdf'), format='pdf',bbox_inches='tight')
    plt.show()
        
    del ctv_VHF, ctv_VHM, ctv_classic, CTVs_classic, CTVs_VHM, CTVs_VHF
    
# Volume and surface similarity 

for i in range(len(resistance)):
    
    print(str(round(statistics.mean([eval_metrics_Ref2VHM[k][i]['DSC'] for k in range(len(PIDs))]), 2)))
    print(str(round(statistics.stdev([eval_metrics_Ref2VHM[k][i]['DSC'] for k in range(len(PIDs))]), 2)))
    
    print(str(round(statistics.mean([eval_metrics_Ref2VHF[k][i]['DSC'] for k in range(len(PIDs))]), 2)))
    print(str(round(statistics.stdev([eval_metrics_Ref2VHF[k][i]['DSC'] for k in range(len(PIDs))]), 2)))
    
    print(str(round(statistics.mean([eval_metrics_VHM2VHF[k][i]['DSC'] for k in range(len(PIDs))]), 2)))
    print(str(round(statistics.stdev([eval_metrics_VHM2VHF[k][i]['DSC'] for k in range(len(PIDs))]), 2)))
    
    print(str(round(statistics.mean([eval_metrics_Ref2VHM[k][i]['HD95'] for k in range(len(PIDs))]), 2)))
    print(str(round(statistics.stdev([eval_metrics_Ref2VHM[k][i]['HD95'] for k in range(len(PIDs))]), 2)))
    
    print(str(round(statistics.mean([eval_metrics_Ref2VHF[k][i]['HD95'] for k in range(len(PIDs))]), 2)))
    print(str(round(statistics.stdev([eval_metrics_Ref2VHF[k][i]['HD95'] for k in range(len(PIDs))]), 2)))
    
    print(str(round(statistics.mean([eval_metrics_VHM2VHF[k][i]['HD95'] for k in range(len(PIDs))]), 2)))
    print(str(round(statistics.stdev([eval_metrics_VHM2VHF[k][i]['HD95'] for k in range(len(PIDs))]), 2)))

# Volume reduction 

for i in range(len(resistance)):
    
    print(str(round(statistics.mean([eval_metrics_Ref2VHM[k][i]['deltaV'] for k in range(len(PIDs))]), 2)))
    print(str(round(statistics.stdev([eval_metrics_Ref2VHM[k][i]['deltaV'] for k in range(len(PIDs))]), 2)))
    
    print(str(round(statistics.mean([eval_metrics_Ref2VHF[k][i]['deltaV'] for k in range(len(PIDs))]), 2)))
    print(str(round(statistics.stdev([eval_metrics_Ref2VHF[k][i]['deltaV'] for k in range(len(PIDs))]), 2)))
    