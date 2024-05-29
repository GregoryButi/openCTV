import os
import numpy as np
import nibabel as nib
from dipy.io.image import load_nifti
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass as com

from opentps.core.data.images._image3D import Image3D
from Process.Tensors import TensorDiffusion
from Process.CTVs import CTVGeometric
from Process import Struct
from Analysis.contourComparison import dice_score, percentile_hausdorff_distance

# PIDs = ['SAC_001','SAC_002','SAC_003','SAC_004', 'SAC_005', 'SAC_007', 'SAC_009', 'SAC_010', 'SAC_011', 'SAC_011','SAC_012', 'SAC_013', 'SAC_014', 'SAC_016', 'SAC_017', 'SAC_018', 'SAC_019', 'SAC_020', 'SAC_025']

PID = 'SAC_004'

# input
path_CT = f'/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Sarcoma/SAC-RT/{PID}/Therapy-scan/CT_norm.nii.gz'
path_tensor = f'/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Sarcoma/SAC-RT/{PID}/tensor_warped_atlas.nii.gz'
path_RTstructs = f'/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Sarcoma/SAC-RT/{PID}/Structures/'
path_barriers = f'/home/gregory/Documents/Projects/CTV_RO1/Data/MGH_Sarcoma/SAC-RT/{PID}/Structures/TotalSegmentator/'
     
tensor = TensorDiffusion()
tensor.loadTensor(path_tensor, 'diffusion')

CT, grid2world, voxel_size = load_nifti(path_CT, return_voxsize=True)
label_map = nib.load(os.path.join(path_RTstructs, 'Label_map.nii.gz')).get_fdata()

# set structures
RTs = Struct()
RTs.loadContours_folder(path_RTstructs, ['CTV', 'External'])
RTs.loadContours_folder(path_barriers, ['hip_left', 'hip_right'])

RTs.setMask('GTV', label_map==1, voxel_size)
RTs.setMask('BS', ~RTs.getMaskByName('External').imageArray, voxel_size)

muscles = ((label_map == 2).astype(int) + (label_map == 3).astype(int) + (label_map == 4).astype(int) + (label_map == 7).astype(int) + (label_map == 8).astype(int) + (label_map == 9).astype(int)) > 0
RTs.setMask('Muscles', muscles, voxel_size)

# define structure of preferred spread
RTs.setMask('PS', np.logical_and(RTs.getMaskByName('Muscles').imageArray, ~RTs.getMaskByName('BS').imageArray), voxel_size)
# define barriers
RTs.setMask('BS', np.logical_or(RTs.getMaskByName('hip_left').imageArray, RTs.getMaskByName('hip_right').imageArray), voxel_size)

# define computation domain
domain = np.logical_or(RTs.getMaskByName('Muscles').imageArray,RTs.getMaskByName('CTV').imageArray)

# define computation domain
domain = RTs.getMaskByName('Muscles').imageArray

# reduce images
RTs.reduceGrid_mask(domain)
ct = Image3D(imageArray=CT, spacing=voxel_size)
ct.reduceGrid_mask(domain)
tensor.reduceGrid_mask(domain)

model = {'model': 'Anisotropic',
         'model-DTI': 'Rekik',
         'obstacle': True,
         'resistance': None,
         'anisotropy': None} 

# model parameters
margin_tissue = np.array([10, 10])
margin_muscle = np.array([20, 30])
resistance = (margin_tissue/margin_muscle)**2

CTVs_classic = np.zeros(tuple(ct.gridSize) + (len(resistance),)).astype(bool)
CTVs_poisson = np.zeros(tuple(ct.gridSize) + (len(resistance),)).astype(bool)
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
        
    # store mask in array
    CTVs_classic[:,:,:,i] = ctv_classic.imageArray
    
    # store distance map in array
    Distance_classic[:,:,:,i] = ctv_classic.distance3D
    
    model['resistance'] = resistance[i]
    
    ctv_poisson = CTVGeometric(spacing=voxel_size)
    ctv_poisson.setCTV_volume(volume, RTs, tensor=tensor, model=model, x0 = margin_tissue[i])
    ctv_poisson.smoothMask(RTs.getMaskByName('BS').imageArray)
            
    # store mask in array
    CTVs_poisson[:,:,:,i] = ctv_poisson.imageArray
    
    print('HD95: ' + str(percentile_hausdorff_distance(ctv_classic.getMeshpoints(), ctv_poisson.getMeshpoints(), percentile=95)))
    
# Plot results

X,Y,Z = com(RTs.getMaskByName('GTV').imageArray)
X = int(X)
Y = int(Y)+5
Z = int(Z)

x, y, z = ct.getMeshGridAxes()

plotGTV = RTs.getMaskByName('GTV').imageArray.astype(float).copy()
plotGTV[~RTs.getMaskByName('GTV').imageArray] = np.NaN
plotMuscles = RTs.getMaskByName('Muscles').imageArray.astype(float).copy()
plotMuscles[~RTs.getMaskByName('Muscles').imageArray] = np.NaN
    
fig1, axes1 = plt.subplots(1, len(resistance))
fig1.subplots_adjust(wspace=0.01) # Remove spacing between subplots

for i in range(len(resistance)):
    
    # Find the maximum value within the mask
    max_value = np.max(Distance_classic[:,:,:,i][CTVs_poisson[:,:,:,i]])
    indices = np.where(Distance_classic[:,:,:,i] == max_value)
    
    # Extract X, Y, and Z indices
    X, Y, Z = indices
    X = X[0]
    Y = Y[0]
    Z = Z[0]
    
    axes1[i].imshow(np.transpose(ct.imageArray[:,:,Z]), cmap = 'gray', vmin=0, vmax=1, extent=[x.min(), x.max(), y.min(), y.max()])
    axes1[i].contourf(x, y, np.flip(np.transpose(plotGTV[:,:,Z]), axis=0), colors='yellow', alpha=0.5)
    axes1[i].contourf(x, y, np.flip(np.transpose(plotMuscles[:,:,Z]), axis=0), colors='blue', alpha=0.2)
    axes1[i].contour(x, y, np.flip(np.transpose(CTVs_classic[:,:,Z,i]), axis=0), colors='green', linewidths=1)
    axes1[i].contour(x, y, np.flip(np.transpose(CTVs_poisson[:,:,Z,i]), axis=0), colors='red', linewidths=1)
    axes1[i].text(0.01, 0.995, '('+str(margin_tissue[i])+' / '+str(margin_muscle[i])+') mm', transform=axes1[i].transAxes, fontsize=8, color='white', ha='left', va='top')
    axes1[i].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
   
#plt.savefig(os.path.join(os.getcwd(),'CTVs_axial.pdf'), format='pdf',bbox_inches='tight')
plt.show()

fig2, axes2 = plt.subplots(1, len(resistance))
fig2.subplots_adjust(wspace=0.01)

for i in range(len(resistance)):
   
    # Find the maximum value within the mask
    max_value = np.max(Distance_classic[:,:,:,i][CTVs_poisson[:,:,:,i]])
    indices = np.where(Distance_classic[:,:,:,i] == max_value)
   
    # Extract X, Y, and Z indices
    X, Y, Z = indices
    X = X[0]
    Y = Y[0]
    Z = Z[0]     
   
    axes2[i].imshow(np.flip(np.transpose(ct.imageArray[:,Y,:]), axis=0), cmap = 'gray', vmin=0, vmax=1, extent=[x.min(), x.max(), z.min(), z.max()])
    axes2[i].contourf(x, z, np.transpose(plotGTV[:,Y,:]), colors='yellow', alpha=0.5)
    axes2[i].contourf(x, z, np.transpose(plotMuscles[:,Y,:]), colors='blue', alpha=0.2)
    axes2[i].contour(x, z, np.transpose(CTVs_classic[:,Y,:,i]), colors='green', linewidths=1)
    axes2[i].contour(x, z, np.transpose(CTVs_poisson[:,Y,:,i]), colors='red', linewidths=1)
    axes2[i].text(0.01, 0.995, '('+str(margin_tissue[i])+' / '+str(margin_muscle[i])+') mm', transform=axes2[i].transAxes, fontsize=8, color='white', ha='left', va='top')
    axes2[i].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

#plt.savefig(os.path.join(os.getcwd(),'CTVs_coronal.pdf'), format='pdf',bbox_inches='tight')
plt.show()

fig3, axes3 = plt.subplots(1, len(resistance))
fig3.subplots_adjust(wspace=0.01)

for i in range(len(resistance)):
    
    # Find the maximum value within the mask
    max_value = np.max(Distance_classic[:,:,:,i][CTVs_poisson[:,:,:,i]])
    indices = np.where(Distance_classic[:,:,:,i] == max_value)
    
    # Extract X, Y, and Z indices
    X, Y, Z = indices
    X = X[0]
    Y = Y[0]
    Z = Z[0]   
    
    axes3[i].imshow(np.flip(np.transpose(ct.imageArray[X,:,:]), axis=0), cmap = 'gray', vmin=0, vmax=1, extent=[y.min(), y.max(), z.min(), z.max()])
    axes3[i].contourf(y, z, np.transpose(plotGTV[X,:,:]), colors='yellow', alpha=0.5)
    axes3[i].contourf(y, z, np.transpose(plotMuscles[X,:,:]), colors='blue', alpha=0.2)
    axes3[i].contour(y, z, np.transpose(CTVs_classic[X,:,:,i]), colors='green', linewidths=1)
    axes3[i].contour(y, z, np.transpose(CTVs_poisson[X,:,:,i]), colors='red', linewidths=1)
    axes3[i].text(0.01, 0.995, '('+str(margin_tissue[i])+' / '+str(margin_muscle[i])+') mm', transform=axes3[i].transAxes, fontsize=8, color='white', ha='left', va='top')
    axes3[i].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
 
#plt.savefig(os.path.join(os.getcwd(),'CTVs_sagittal.pdf'), format='pdf',bbox_inches='tight')
plt.show()    