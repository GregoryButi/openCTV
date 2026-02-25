#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:12:05 2024

@author: gregory
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass as com
from matplotlib.widgets import Slider, Button
from dipy.io.image import load_nifti, save_nifti
from dipy.viz import regtools

from opentps.core.data.images._image3D import Image3D

from Process import TensorDiffusion
from Process import CTVGeometric
from Process import Struct
from Process import TransformTensorDeformable
from Process import ImageRegistrationDeformable

visualization = False
interactive = False

# atlas
path_moving = '../../../Input/Atlas/MNI152_T1_1mm_brain_norm.nii.gz'
path_tensor = '../../../Input/Atlas/FSL_HCP1065_tensor_1mm_Ants.nii.gz'
path_segmentation = '../../../Input/Atlas/seg.mgz'

# patient IDs
patient_dir = '/media/gregory/Elements/Data/KRDI_Glioma/data_GliODIL_essential'
processed_dir = '/media/gregory/Elements/Data/KRDI_Glioma/Processed'
PIDs_all = sorted(os.listdir(patient_dir))
PIDs_all.remove('data_433')
PIDs_all.remove('data_454')
PIDs_all.remove('data_530')
PIDs_all.remove('data_703')
PIDs_all.remove('data_704')
PIDs_all.remove('data_735')
#PIDs_all = sorted(os.listdir(processed_dir))

# output
path_output = '/home/gregory/Downloads/test'

# CTV model
model_NU = {
    'obstacle': True,
    'model': 'Nonuniform',
    'resistance': 0.1
}

model_Aniso = {
    'obstacle': True,
    'model': 'Anisotropic',
    'model-DTI': 'Clatz',
    'resistance': 0.05,
    'anisotropy': 1.0
}

margin = 15
#resistances = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
resistances = [0.02, 0.03, 0.04, 0.05, 0.1, 0.2]
anisotropies = [2.0, 1.5, 1.0, 0.5, 0.0]

def calculate_recurrence_coverage_PIDs(PIDs, model, fromDistance):

    recurrence_coverage_PIDs = np.zeros((len(PIDs), ))
    recurrence_coverage_ref_PIDs = np.zeros((len(PIDs), ))
    for PID, i in zip(PIDs, range(len(PIDs))):

        # input

        path_gtv = os.path.join(patient_dir, f'{PID}/segm.nii.gz')
        path_rec = os.path.join(patient_dir, f'{PID}/segm_rec.nii.gz')
        path_wm = os.path.join(patient_dir, f'{PID}/t1_wm.nii.gz')
        path_gm = os.path.join(patient_dir, f'{PID}/t1_gm.nii.gz')
        path_csf = os.path.join(patient_dir, f'{PID}/t1_csf.nii.gz')
        path_tensor_warped = os.path.join(processed_dir, f'{PID}/Tensor_warped.nii.gz')

        # load data

        gtv_comp, static_grid2world, voxel_size = load_nifti(path_gtv, return_voxsize=True)
        rec, _, _ = load_nifti(path_rec, return_voxsize=True)
        wm_prob, _, _ = load_nifti(path_wm, return_voxsize=True)
        gm_prob, _, _ = load_nifti(path_gm, return_voxsize=True)
        csf_prob, _, _ = load_nifti(path_csf, return_voxsize=True)

        # load structures

        gtv = (gtv_comp == 1) + (gtv_comp == 2) + (gtv_comp == 4) # exclude edema
        rec = rec == 1 # only enhancing core

        tissue_idx = np.argmax(np.concatenate((wm_prob[..., np.newaxis], gm_prob[..., np.newaxis], csf_prob[..., np.newaxis]), axis=-1), axis=-1)
        wm = np.logical_and(tissue_idx == 0, wm_prob > 0.1)
        gm = np.logical_and(tissue_idx == 1, gm_prob > 0.1)
        csf = np.logical_and(tissue_idx == 2, csf_prob > 0.1)

        #brain_mask = (wm_prob + gm_prob + csf_prob) > 0.3
        #bs = np.logical_or(csf, ~brain_mask)

        brain_mask = (wm_prob + gm_prob) > 0.5
        bs = ~brain_mask

        RTs = Struct()
        RTs.setMask('BS', bs, voxel_size)
        RTs.setMask('GTV', gtv, voxel_size)

        # create pseudo mri
        t1w = gm + 2 * wm
        T1w = Image3D(imageArray=t1w, spacing=voxel_size)

        # voxel display
        COM = np.array(com(gtv))
        X_coord = int(COM[0])
        Y_coord = int(COM[1])
        Z_coord = int(COM[2])

        #################
        #### DTI IR #####
        #################

        RTs.setMask('PS', wm, voxel_size)

        if os.path.exists(path_tensor_warped):

            tensor_transformed = TensorDiffusion()
            tensor_transformed.loadTensor(path_tensor_warped)

        else:

            # load data

            static = t1w.copy()

            moving, moving_grid2world, voxel_size = load_nifti(path_moving, return_voxsize=True)

            tensor = TensorDiffusion()
            tensor.loadTensor(path_tensor, format='ANTs')

            IR = ImageRegistrationDeformable(static, static_grid2world, moving, moving_grid2world, static_mask=(~gtv).astype(int), metric='CC')

            mapping = IR.get_mapping()

            if visualization:

                # plot DIR results
                warped = mapping.transform(moving)
                regtools.overlay_slices(static, warped, None, 2, "Fixed", "Deformed", None)
                plt.rcParams.update({'font.size': 8})
                plt.savefig(os.path.join(path_output,f'DIR_{PID}.pdf'), format='pdf',bbox_inches='tight')
                #plt.show()

            transform = TransformTensorDeformable(mapping)

            # apply transformations
            tensor_transformed = transform.getTensorDiffusionTransformed(tensor, method='FS')

            if visualization:
                _, _, RGB = tensor_transformed.get_FA_MD_RGB()

                plt.figure()
                plt.imshow(np.flip(np.transpose(RGB[:, :, Z_coord], axes=(1, 0, 2)), axis=0))

                plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
                plt.savefig(os.path.join(path_output,f'tensor_{PID}.pdf'), format='pdf',bbox_inches='tight')
                #plt.show()

            # save
            os.makedirs(os.path.dirname(path_tensor_warped)) # create folder first
            save_nifti(path_tensor_warped, tensor_transformed.imageArray, static_grid2world)

        # Run fast marching method

        GTVBox = RTs.getBoundingBox('GTV', int(margin + 15))

        CTV_ref = CTVGeometric(rts=RTs, spacing=voxel_size, model={'obstacle': True, 'model': None})
        CTV_ref.setCTV_isodistance(margin)

        if fromDistance:
            CTV_model = CTV_ref
        else:
            CTV_model = CTVGeometric(rts=RTs, spacing=voxel_size, tensor=tensor_transformed, model=model)
            CTV_model.setCTV_metric(CTV_ref, metric='volume' , method='Nelder-Mead', x0=margin, domain=GTVBox)

        # smoothing to remove holes etc.
        #CTV_DL.smoothMask()

        # calculate recurrence coverage

        if rec.sum() == 0:
            recurrence_coverage_PIDs[i] = 1.
            recurrence_coverage_ref_PIDs[i] = 1.
        else:
            recurrence_coverage_PIDs[i] = np.logical_and(CTV_model.imageArray, rec).sum()/rec.sum()
            recurrence_coverage_ref_PIDs[i] = np.logical_and(CTV_ref.imageArray, rec).sum()/rec.sum()

        print(f"patient {i+1}/{len(PIDs)} -- {PID}")

        print(f"Recurrence coverage CTV: {recurrence_coverage_PIDs[i] * 100:.2f}%")

        print(f"     -------     ")
        print(f"     -------     ")

        # Create 2D plots

        if visualization or interactive:

            x, y, z = T1w.getMeshGridAxes()

            # prepare figures
            plotT1w = t1w.copy()
            plotGTV = gtv.astype(float).copy()
            plotGTV[~gtv] = np.NaN
            plotRec = rec.astype(float).copy()
            plotRec[~rec] = np.NaN
            plotBS = bs.astype(float).copy()

            plotDistance = CTV_ref.distance3D.copy()
            plotDistance[bs] = np.NaN

        if visualization:

            plt.figure()
            plt.imshow(np.flip(plotT1w[:, :, Z_coord].transpose(), axis=0), extent=[x[0], x[-1], y[0], y[-1]], cmap='gray')
            plt.contour(plotBS[:, :, Z_coord].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='blue', linewidths=1)
            plt.contourf(plotGTV[:, :, Z_coord].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='yellow', alpha=0.5)
            plt.contourf(plotRec[:, :, Z_coord].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', alpha=0.5)
            plt.contour(CTV_ref.imageArray[:, :, Z_coord].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='green', linewidths=2)
            plt.contour(CTV_model.imageArray[:, :, Z_coord].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', linewidths=2, linestyles='dashed')

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

            plt.savefig(os.path.join(path_output, f'CTV_comparison_{PID}.pdf'), format='pdf', bbox_inches='tight')
            #plt.show()

            # Create a figure with three subplots side by side
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Adjust the figsize as needed

            # Plot for Standard Distance
            axs[0].imshow(plotDistance[:, :, Z_coord].transpose(), extent=[x[0], x[-1], y[0], y[-1]], cmap='jet', vmin=0, vmax=80)
            axs[0].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            axs[0].set_title('Standard Distance')

            # Plot for Riemann Distance
            axs[1].imshow(CTV_model.distance3D[:, :, Z_coord].transpose(), extent=[x[0], x[-1], y[0], y[-1]], cmap='jet', vmin=0, vmax=80*np.sqrt(0.1))
            axs[1].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            axs[1].set_title('Nonuniform Distance')

            plt.tight_layout()
            plt.savefig(os.path.join(path_output, f'Distance_comparison_{PID}.pdf'), format='pdf', bbox_inches='tight')
            # plt.show()

        if interactive:

            # Build interactive plot

            def plot_isodistance_sagittal(ax, X, plotCTV, plotCTV_model):

                fig.add_axes(ax)
                plt.imshow(np.flip(plotT1w[X, :, :].transpose(), axis=0), extent=[y[0], y[-1], z[0], z[-1]], cmap='gray')
                plt.contour(plotBS[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='blue', linewidths=1)
                plt.contourf(plotGTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='yellow', alpha=0.5)
                plt.contourf(plotRec[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='red', alpha=0.5)
                plt.contour(plotCTV[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='green', linewidths=1.5)
                plt.contour(plotCTV_model[X, :, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')

                plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

            def plot_isodistance_coronal(ax, Y, plotCTV, plotCTV_model):

                fig.add_axes(ax)
                plt.imshow(np.flip(plotT1w[:, Y, :].transpose(), axis=0), extent=[x[0], x[-1], z[0], z[-1]], cmap='gray')
                plt.contour(plotBS[:, Y, :].transpose(), extent=[y[0], y[-1], z[0], z[-1]], colors='blue', linewidths=1)
                plt.contourf(plotGTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='yellow', alpha=0.5)
                plt.contourf(plotRec[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='red', alpha=0.5)
                plt.contour(plotCTV[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='green', linewidths=1.5)
                plt.contour(plotCTV_model[:, Y, :].transpose(), extent=[x[0], x[-1], z[0], z[-1]], colors='red', linewidths=1.5, linestyles='dashed')

                plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

            def plot_isodistance_axial(ax, Z, plotCTV, plotCTV_model):

                fig.add_axes(ax)
                plt.imshow(np.flip(plotT1w[:, :, Z].transpose(), axis=0), extent=[x[0], x[-1], y[0], y[-1]], cmap='gray')
                plt.contour(plotBS[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='blue', linewidths=1)
                plt.contourf(plotGTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='yellow', alpha=0.5)
                plt.contourf(plotRec[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', alpha=0.5)
                plt.contour(plotCTV[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='green', linewidths=1.5)
                plt.contour(plotCTV_model[:, :, Z].transpose(), extent=[x[0], x[-1], y[0], y[-1]], colors='red', linewidths=1.5, linestyles='dashed')

                plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

            # Plot
            fig = plt.figure()
            plt.axis('off')
            # plt.title('Interactive slider')

            ax1 = fig.add_subplot(131)
            plot_isodistance_sagittal(ax1, X_coord, CTV_ref.imageArray, CTV_model.imageArray)
            ax2 = fig.add_subplot(132)
            plot_isodistance_coronal(ax2, Y_coord, CTV_ref.imageArray, CTV_model.imageArray)
            ax3 = fig.add_subplot(133)
            plot_isodistance_axial(ax3, Z_coord, CTV_ref.imageArray, CTV_model.imageArray)

            # Define sliders

            # Make a vertically oriented slider to control the slice
            # position x, position y, x-length, y-length
            alpha_axis_1 = plt.axes([0.1, 0.3, 0.0125, 0.42])
            alpha_slider_1 = Slider(
                ax=alpha_axis_1,
                label="Slice",
                valmin=0,
                valmax=T1w.gridSize[0],
                valinit=X_coord,
                valstep=1,
                orientation="vertical"
            )

            alpha_axis_2 = plt.axes([0.38, 0.3, 0.0125, 0.42])
            alpha_slider_2 = Slider(
                ax=alpha_axis_2,
                label="Slice",
                valmin=0,
                valmax=T1w.gridSize[1],
                valinit=Y_coord,
                valstep=1,
                orientation="vertical"
            )

            alpha_axis_3 = plt.axes([0.65, 0.3, 0.0125, 0.42])
            alpha_slider_3 = Slider(
                ax=alpha_axis_3,
                label="Slice",
                valmin=0,
                valmax=T1w.gridSize[2],
                valinit=Z_coord,
                valstep=1,
                orientation="vertical"
            )

            # Make horizontal oriented slider to control the distance
            alpha_axis_4 = plt.axes([0.3, 0.06, 0.4, 0.03])
            alpha_slider_4 = Slider(
                ax=alpha_axis_4,
                label='Margin (mm)',
                valmin=0.0,
                valmax=30.0,
                valinit=CTV_ref.isodistance,
                valstep=1.0
            )

            alpha4_prev = None # initialize
            def update(val):
                global alpha4_prev

                alpha1 = int(alpha_slider_1.val)
                alpha2 = int(alpha_slider_2.val)
                alpha3 = int(alpha_slider_3.val)
                alpha4 = alpha_slider_4.val

                # update CTV
                CTV_model.setCTV_isodistance(alpha4)

                ax1.cla()
                plot_isodistance_sagittal(ax1, alpha1, CTV_ref.imageArray, CTV_model.imageArray)
                ax2.cla()
                plot_isodistance_coronal(ax2, alpha2, CTV_ref.imageArray, CTV_model.imageArray)
                ax3.cla()
                plot_isodistance_axial(ax3, alpha3, CTV_ref.imageArray, CTV_model.imageArray)

                plt.draw()

            alpha_slider_1.on_changed(update)
            alpha_slider_2.on_changed(update)
            alpha_slider_3.on_changed(update)
            alpha_slider_4.on_changed(update)

            # Create axes for reset button and create button
            resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
            button = Button(resetax, 'Reset', color='gold',
                            hovercolor='skyblue')


            # Create a function resetSlider to set slider to
            # initial values when Reset button is clicked

            def resetSlider(event):
                alpha_slider_1.reset()
                alpha_slider_2.reset()
                alpha_slider_3.reset()

            # Call resetSlider function when clicked on reset button
            button.on_clicked(resetSlider)

            plt.show()

    return recurrence_coverage_PIDs, recurrence_coverage_ref_PIDs

#
recurrence_coverage_model, recurrence_coverage_Stand = calculate_recurrence_coverage_PIDs(PIDs_all, model_Aniso, False)

print('Differences')
difference = np.array(recurrence_coverage_model) - np.array(recurrence_coverage_Stand)
print(difference)

print(np.sum(difference>=0.01))

print(np.sum(difference<=-0.01))

print(np.mean(difference[difference>=0.01]))
print(np.mean(difference[difference<=-0.01]))


breakpoint()

# perform n-fold cross validation

nfolds = 5

resistance_best_train_NU = np.zeros((nfolds, ))
resistance_best_train_Aniso = np.zeros((nfolds, ))
anisotropy_best_train_Aniso = np.zeros((nfolds, ))
recurrence_coverage_CTV_NU_PIDs = np.zeros((len(PIDs_all), ))
recurrence_coverage_CTV_Aniso_PIDs = np.zeros((len(PIDs_all), ))

from sklearn.model_selection import KFold
kf = KFold(n_splits=nfolds, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(PIDs_all)):

    print(f'Fold {fold+1}/{nfolds}')

    # Split data into training and testing sets
    train_PIDs = [PIDs_all[i] for i in train_idx]
    test_PIDs = [PIDs_all[i] for i in test_idx]

    recurrence_coverage_CTV_NU_PIDs_mean = np.zeros((len(resistances),))
    recurrence_coverage_CTV_Aniso_PIDs_mean = np.zeros((len(resistances), len(anisotropies)))
    recurrence_coverage_CTV_Stand_PIDs_mean = np.zeros((len(resistances),))
    for resistance, i in zip(resistances, range(len(resistances))):

        print(f'Evaluating resistance parameter: {resistance}')
        model_NU['resistance'] = resistance

        recurrence_coverage_NU, recurrence_coverage_Stand = calculate_recurrence_coverage_PIDs(train_PIDs, model_NU, False)

        # calculate average
        recurrence_coverage_CTV_NU_PIDs_mean[i] = np.mean(recurrence_coverage_NU)
        recurrence_coverage_CTV_Stand_PIDs_mean[i] = np.mean(recurrence_coverage_Stand)

        for anisotropy, j in zip(anisotropies, range(len(anisotropies))):

            print(f'Evaluating anisotropy parameter: {anisotropy}')

            model_Aniso['resistance'] = resistance
            model_Aniso['anisotropy'] = anisotropy

            recurrence_coverage_Aniso, _ = calculate_recurrence_coverage_PIDs(train_PIDs, model_Aniso, False)

            # calculate average
            recurrence_coverage_CTV_Aniso_PIDs_mean[i, j]= np.mean(recurrence_coverage_Aniso)

    plt.figure()
    plt.plot(resistances, recurrence_coverage_CTV_NU_PIDs_mean, '-r', label='Non-uniform plan')
    for anisotropy, j in zip(anisotropies, range(len(anisotropies))):
        plt.plot(resistances, recurrence_coverage_CTV_Aniso_PIDs_mean[:, j], label=f'Anisotropic plan ({anisotropy})')
    plt.plot(resistances, recurrence_coverage_CTV_Stand_PIDs_mean, '--k', label='Standard plan')
    plt.legend()
    plt.xlabel('Resistance coefficient')
    plt.ylabel('Recurrence coverage')

    plt.savefig(os.path.join(path_output, f'Recurrence_coverage_fold_{fold}.pdf'), format='pdf', bbox_inches='tight')

    idx_NU = np.argmax(recurrence_coverage_CTV_NU_PIDs_mean)
    idx_Aniso = np.unravel_index(recurrence_coverage_CTV_Aniso_PIDs_mean.argmax(), recurrence_coverage_CTV_Aniso_PIDs_mean.shape)

    resistance_best_train_NU[fold] = np.array(resistances)[idx_NU]
    resistance_best_train_Aniso[fold] = np.array(resistances)[idx_Aniso[0]]
    anisotropy_best_train_Aniso[fold] = np.array(anisotropies)[idx_Aniso[1]]

    # update model and run on test set
    model_NU['resistance'] = resistance_best_train_NU[fold]
    model_Aniso['resistance'] = resistance_best_train_Aniso[fold]
    model_Aniso['anisotropy'] = anisotropy_best_train_Aniso[fold]

    recurrence_coverage_CTV_NU_PIDs[test_idx], _ = calculate_recurrence_coverage_PIDs(test_PIDs, model_NU, False)
    recurrence_coverage_CTV_Aniso_PIDs[test_idx], _ = calculate_recurrence_coverage_PIDs(test_PIDs, model_Aniso, False)

#
recurrence_coverage_CTV_Standard_PIDs = calculate_recurrence_coverage_PIDs(PIDs_all, {'obstacle': True, 'model': None}, True)

# calculate statistics
recurrence_coverage_CTV_Standard_mean, recurrence_coverage_CTV_Standard_std = np.mean(recurrence_coverage_CTV_Standard_PIDs), np.std(recurrence_coverage_CTV_Standard_PIDs)
recurrence_coverage_CTV_NU_PIDs_mean, recurrence_coverage_CTV_NU_PIDs_std = np.mean(recurrence_coverage_CTV_NU_PIDs), np.std(recurrence_coverage_CTV_NU_PIDs)
recurrence_coverage_CTV_Aniso_PIDs_mean, recurrence_coverage_CTV_Aniso_PIDs_std = np.mean(recurrence_coverage_CTV_Aniso_PIDs), np.std(recurrence_coverage_CTV_Aniso_PIDs)

print(f"##################")
print(f"##################")

print(f"Resistance coefficients NU CTV: {resistance_best_train_NU}")
print(f"Resistance coefficients Aniso CTV: {resistance_best_train_Aniso}")

print(f"Mean recurrence coverage test Standard CTV: {recurrence_coverage_CTV_Standard_mean*100:.2f}% +/- {recurrence_coverage_CTV_Standard_std*100:.2f}%")
print(f"Mean recurrence coverage test NU CTV: {recurrence_coverage_CTV_NU_PIDs_mean*100:.2f}% +/- {recurrence_coverage_CTV_NU_PIDs_std*100:.2f}%")
print(f"Mean recurrence coverage test Aniso CTV: {recurrence_coverage_CTV_Aniso_PIDs_mean*100:.2f}% +/- {recurrence_coverage_CTV_Aniso_PIDs_std*100:.2f}%")

print(f"##################")
print(f"##################")

plt.figure()
plt.plot(range(nfolds), resistance_best_train_NU, '-r', label='Non-uniform plan')
plt.plot(range(nfolds), resistance_best_train_Aniso, '-b', label='Anisotropic plan')
plt.xlabel('Fold')
plt.ylabel('Resistance coefficient')
plt.savefig(os.path.join(path_output, f'Resistance_coefficient_best.pdf'), format='pdf', bbox_inches='tight')