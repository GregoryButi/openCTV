#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:12:05 2024

@author: gregory
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass as com
from matplotlib.widgets import Slider
from dipy.io.image import load_nifti
import time
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import copy

from opentps.core.data.images._image3D import Image3D

from Process.CTVs import CTVGeometric
from Process import Struct
from Process.Analysis.contourComparison import dice_score, percentile_hausdorff_distance, surface_dice_score

PIDs = ['Spine_002_T7', 'Spine_005_L3', 'Spine_008_T12', 'Spine_007_T10', 'Spine_009_T11', 'Spine_010_L5', 'Spine_011_T3', 'Spine_012_T12', 'Spine_013_L3', 'Spine_021_T8', 'Spine_023_T12']
names_GTV = ['GTV_T7', 'GTV_L3', 'GTV_T12', 'GTV_T10', 'GTV_T11', 'GTV_L5', 'GTV_T3', 'GTV_T12', 'GTV_L3', 'GTV_T8', 'GTV_T12']
names_CTV = ['CTV_T7', 'CTV_L3', 'CTV_T12', 'CTV_T10', 'CTV_T11', 'CTV_L5', 'CTV_T3', 'CTV_T12', 'CTV_L3', 'CTV_T8', 'CTV_T12']
names_SC = ['SpinalCord', 'CaudaEquina', 'SpinalCord', 'SpinalCord', 'SpinalCord', 'ThecalSac', 'SpinalCord', 'SpinalCord', 'ThecalSac', 'SpinalCord', 'SpinalCord']
names_Verteb = ['T7', 'L3', 'T12', 'T10', 'T11', 'L5', 'T3', 'T12', 'L3', 'T8', 'T12']

names_levels = [f'C{i}' for i in range(1,8)] + [f'T{i}' for i in range(1,13)] + [f'L{i}' for i in range(1,6)]

visualization = False

# ------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------
radius = 50
target_axis = np.array([0, 0, 1])
n_levels_upper = [4, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2]
n_levels_lower = [4, 2, 3, 2, 2, 0, 2, 2, 2, 2, 2]
poly_deg = 3
theta_bound = 20.0
refinement = False

if refinement:
    thetas = np.linspace(-theta_bound, theta_bound, 10) # degrees
else:
    thetas = [0]

# Define all angle sector sets
ANGLE_SETS = {
    "C": {   # Cervical
        "offset": 20.5,
        "angles": [90.1, 25.9, 87.5, 45.4, 83.7, 26.0]
    },
    "T": {   # Thoracic
        "offset": 13.8,
        "angles": [153.7, 18.0, 59.5, 49.9, 61.1, 17.8]
    },
    "L": {   # Lumbar
        "offset": 14.3,
        "angles": [152.9, 18.9, 69.0, 29.6, 72.8, 16.8]
    }
}

alphas = [0.25, 0.4, 0.55, 0.7, 0.85, 1.]

def get_barrier_segments(sectors, gtv):
    """
    Given:
        sectors : list/array of boolean masks (each mask = one sector)
        gtv     : boolean mask for the GTV

    Returns:
        bs_sector : sorted numpy array of barrier sector indices (1–6)
    """

    # --- Determine which sectors contain GTV ---
    gtv_sector = []
    for s, sector in enumerate(sectors):
        if np.any(sector[gtv]):
            gtv_sector.append(s + 1)   # sectors nscipyumbered 1..6

    # --- Sector mapping rules ---
    rules = {
        1: [3, 4, 5],
        2: [4, 5, 6],
        3: [1, 5, 6],
        4: [1, 2, 6],
        5: [1, 2, 3],
        6: [2, 3, 4],
    }

    # --- Collect all barrier sectors ---
    bs_sector = []
    for g in gtv_sector:
        bs_sector.extend(rules[g])

    # Unique + filter out sectors that contain GTV
    bs_sector = np.unique(bs_sector)
    bs_sector = np.array([s for s in bs_sector if s not in gtv_sector])

    return bs_sector

def fit_polynomial_curve_3d(coms, deg=2, n_points=200):
    """
    Fits a parametric polynomial curve x(t), y(t), z(t)
    through given 3D points.
    """
    coms = np.asarray(coms)  # shape (N,3)

    # Parameter t values distributed evenly from 0 to 1
    t = np.linspace(0, 1, len(coms))

    # Fit polynomial for x(t), y(t), z(t)
    px = np.polyfit(t, coms[:, 0], deg)
    py = np.polyfit(t, coms[:, 1], deg)
    pz = np.polyfit(t, coms[:, 2], deg)

    # Evaluate curve
    t_fine = np.linspace(0, 1, n_points)
    x_fine = np.polyval(px, t_fine)
    y_fine = np.polyval(py, t_fine)
    z_fine = np.polyval(pz, t_fine)

    return (px, py, pz) , np.vstack([x_fine, y_fine, z_fine]).T

def derivative(poly):
    """Compute derivative polynomial coefficients."""
    n = len(poly) - 1
    return np.array([poly[i] * (n - i) for i in range(n)])

def tangent_at_t(t, px_d, py_d, pz_d):
    """Compute tangent vector at parameter t."""
    dx = np.polyval(px_d, t)
    dy = np.polyval(py_d, t)
    dz = np.polyval(pz_d, t)
    v = np.array([dx, dy, dz])
    return v

def t_from_z_world(slice_z, pz):
    """Directly compute t for a given world Z coordinate."""
    deg = len(pz) - 1
    if deg == 1:  # linear
        a, b = pz
        return (slice_z - b) / a
    elif deg == 2:  # quadratic
        a, b, c = pz
        disc = b ** 2 - 4 * a * (c - slice_z)
        t1 = (-b + np.sqrt(disc)) / (2 * a)
        t2 = (-b - np.sqrt(disc)) / (2 * a)
        t_candidates = [t for t in (t1, t2) if 0 <= t <= 1]
        return t_candidates[0]
    else:  # cubic
        roots = np.roots(np.append(pz[:-1], pz[-1] - slice_z))
        roots = roots[np.isreal(roots)].real
        t_candidates = roots[(roots >= 0) & (roots <= 1)]
        return t_candidates[0]

def apply_transformation(v_norm, target_axis, rotation_point, Xf, Yf, Zf, theta=0):

    # Compute rotation axis (cross-product)
    axis = np.cross(v_norm, target_axis)
    axis_norm = np.linalg.norm(axis)

    axis = axis / axis_norm

    # Angle between them
    angle = np.arccos(np.clip(np.dot(v_norm, target_axis), -1.0, 1.0))

    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    Rz = np.eye(4)
    Rz[:3, :3] = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    rotation = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)

    # 1. Translate rotation_point to origin
    T1 = np.eye(4)
    T1[:3, 3] = -rotation_point

    # 2. Perform rotation
    R = np.eye(4)
    R[:3, :3] = rotation

    R_total = Rz @ R

    # 3. Translate back
    T2 = np.eye(4)
    T2[:3, 3] = rotation_point

    # Combine matrices: T2 * R * T1
    # The order is crucial: translate to origin, rotate, translate back
    T = T2 @ R_total @ T1

    coords = np.vstack([Xf, Yf, Zf, np.ones(CT.numberOfVoxels)])  # homogeneous coordinates with shape (4, N)

    # Apply the 4×4 transformation
    transformed = T @ coords   # shape (4, N)

    # Extract x,y,z
    Xt_world = transformed[0, :].reshape(CT.gridSize)
    Yt_world = transformed[1, :].reshape(CT.gridSize)
    Zt_world = transformed[2, :].reshape(CT.gridSize)

    return T, Xt_world, Yt_world, Zt_world

class SliceViewer:
    def __init__(self, img, mask, x, y, z, Z0=0,
    ):
        """
        vertebra: 3D array (Nx, Ny, Nz) — segmentation mask
        x, y: physical coordinate vectors for extent
        """
        self.img = img
        self.mask = mask
        self.x = x
        self.y = y
        self.z = z
        self.Z = Z0

        self.clicked_xyz = []
        self.clicked_ijk = []

        self.fig, self.ax = plt.subplots()
        self.draw()

        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        plt.show()

    def draw(self):
        self.ax.clear()

        # Show vertebra segmentation
        self.ax.imshow(
            np.flip(self.img[:, :, self.Z].T, axis=0),
            extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],
            cmap="gray",
            origin="lower",
        )

        # Optional contour for clarity
        if np.any(self.mask[:, :, self.Z]):
            self.ax.contour(
                np.flip(self.mask[:, :, self.Z].T, axis=0),
                extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],
                colors="lime",
                linewidths=1.5,
            )

        self.ax.set_title(f"Vertebra — Z = {self.Z}")
        self.ax.tick_params(
            left=False, right=False, labelleft=False,
            bottom=False, labelbottom=False
        )

        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.button == "up":
            self.Z = min(self.Z + 1, self.mask.shape[2] - 1)
        elif event.button == "down":
            self.Z = max(self.Z - 1, 0)

        self.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        x = event.xdata
        y = event.ydata

        # ---- voxel indices ----
        i = int(round((x - self.x[0]) / (self.x[-1] - self.x[0]) * (len(self.x) - 1)))
        j = int(round((y - self.y[0]) / (self.y[-1] - self.y[0]) * (len(self.y) - 1)))
        k = self.Z

        # clamp to valid range
        i = np.clip(i, 0, self.mask.shape[0] - 1)
        j = np.clip(j, 0, self.mask.shape[1] - 1)

        # adjust
        j = -(j - self.mask.shape[1])

        self.clicked_xyz.append((self.x[i], self.y[j], self.z[k]))
        self.clicked_ijk.append((i, j, k))
        print(f"Clicked point: i={i:.2f}, j={j:.2f}, k={j}")

        self.ax.plot(x, y, "ro", markersize=5)
        self.fig.canvas.draw_idle()


dice_ctv = np.zeros((len(PIDs),))
HD95_ctv = np.zeros((len(PIDs),))
SDS_ctv = np.zeros((len(PIDs),))
dice_vert = np.zeros((len(PIDs),))
HD95_vert = np.zeros((len(PIDs),))
SDS_vert = np.zeros((len(PIDs),))
time_elapsed = np.zeros((len(PIDs),))

cosine_similarities = [[] for _ in enumerate(PIDs)]
for idp, PID in reversed(list(enumerate(PIDs))):
#for idp, PID in enumerate(PIDs):

    print(f"Patient: {PID}")

    # input

    path_CT = f'/media/gregory/Elements/Data/MGH_SpineMets/Spine-RT_Processed/{PID}/Therapy-scan/MRI_CT/CT.nii.gz'
    path_RTstructs_manual = f'/media/gregory/Elements/Data/MGH_SpineMets/Spine-RT_Processed/{PID}/Therapy-scan/Structures_manual'
    path_RTstructs_DL = f'/media/gregory/Elements/Data/MGH_SpineMets/Spine-RT_Processed/{PID}/Therapy-scan/Structures_DL'

    verteb_level = names_Verteb[idp]
    group_letter = verteb_level[0]

    # Select vertebrae set
    idl = names_levels.index(verteb_level)
    names_Verteb_interest = names_levels[idl-n_levels_upper[idp] : idl+n_levels_lower[idp]+1]

    # Select correct angle set
    config = ANGLE_SETS[group_letter]

    offset = config["offset"]
    angle_sector_1, angle_sector_2, angle_sector_3, \
        angle_sector_4, angle_sector_5, angle_sector_6 = config["angles"]

    # Build phi list
    phi_1 = -offset - angle_sector_1
    phi_2 = -offset
    phi_3 = -offset + angle_sector_2
    phi_4 = -offset + angle_sector_2 + angle_sector_3
    phi_5 = -offset + angle_sector_2 + angle_sector_3 + angle_sector_4
    phi_6 = -offset + angle_sector_2 + angle_sector_3 + angle_sector_4 + angle_sector_5

    phis = np.array([phi_1, phi_2, phi_3, phi_4, phi_5, phi_6])

    # load data

    CT, static_grid2world, voxel_size = load_nifti(path_CT, return_voxsize=True)
    origin = static_grid2world[0:3, 3]

    # load structures

    RTs = Struct()
    RTs.loadContours_folder(path_RTstructs_manual, [names_GTV[idp], names_CTV[idp], names_SC[idp], f'{verteb_level}_ISRC'], contour_names=['GTV', 'CTV', 'SpinalCord', 'Vertebra_GT'], contour_types=['GTV', None, None, None])
    RTs.loadContours_folder(path_RTstructs_DL, [verteb_level, 'External'], contour_names=['Vertebra', 'External'])
    RTs.loadContours_folder(path_RTstructs_DL, names_Verteb_interest, contour_names=names_Verteb_interest)

    start = time.perf_counter()

    # reduce calculation grid of images and structures

    external = RTs.getMaskByName('External').imageArray
    CT = Image3D(imageArray=CT, spacing=voxel_size, origin=origin)

    BB = RTs.getBoundingBox('External', margin=20)

    # reduce grid
    CT.reduceGrid_mask(BB)
    RTs.reduceGrid_mask(BB)

    # resample
    #factor = 2
    #spacing_new = (voxel_size[0]/factor, voxel_size[1]/factor, voxel_size[2]/factor)
    #shape_new = (CT.gridSize[0]*factor, CT.gridSize[1]*factor, CT.gridSize[2]*factor)

    #CT.resample(spacing_new, shape_new, CT.origin, fillValue=-1000, tryGPU=True)
    #RTs.resampleMasks(spacing_new, shape_new, CT.origin)

    # reload contour masks

    gtv = RTs.getMaskByName('GTV').imageArray
    sc = RTs.getMaskByName('SpinalCord').imageArray
    external = RTs.getMaskByName('External').imageArray
    verteb_lesion = RTs.getMaskByName('Vertebra').imageArray

    # voxel display
    COM = np.array(com(gtv))
    X_coord, Y_coord, Z_coord = int(COM[0]), int(COM[1]), int(COM[2])

    x, y, z = CT.getMeshGridAxes()
    X_world, Y_world, Z_world = CT.getMeshGridPositions()

    Xf = X_world.ravel()
    Yf = Y_world.ravel()
    Zf = Z_world.ravel()

    landmarks_ijk = [np.empty((0,0)) for _ in range(len(phis))]
    landmarks_xyz = [np.empty((0,0)) for _ in range(len(phis))]
    if refinement:
        for s in range(len(phis)):

            print(f'Click on coordinates of sector {s+1}')

            # verteb_lesion is your 3D numpy array
            viewer = SliceViewer(CT.imageArray, verteb_lesion, x, y, z, Z_coord)

            # After closing the window:
            print("All clicked coordinates:")
            print(viewer.clicked_xyz)

            landmarks_ijk[s] = np.array(viewer.clicked_ijk)
            landmarks_xyz[s] = np.array(viewer.clicked_xyz)

    #######################################################
    #######################################################
    #######################################################

    principal_axis = RTs.principle_comps('SpinalCord', coord_space='World')

    # make z vector positive if necessary
    if principal_axis[2] < 0:
        principal_axis = -principal_axis

    # Normalize vectors
    v_check = principal_axis / np.linalg.norm(principal_axis)
    #rotation_point = RTs.getMaskByName('SpinalCord').centerOfMass

    #T, Xt_world, Yt_world, Zt_world = apply_transformation(v, target_axis, rotation_point, Xf, Yf, Zf)

    #T_inv = np.linalg.inv(T)

    ####################
    ####################
    ####################

    # calculate vertebral centroids
    coms = RTs.getMaskByName(names_Verteb_interest[0]).centerOfMass
    for name in names_Verteb_interest[1:]:
        coms = np.vstack([coms, RTs.getMaskByName(name).centerOfMass])

    (px, py, pz), fitted_points = fit_polynomial_curve_3d(coms, deg=poly_deg)

    _, _, idZ_all = np.where(sc)
    idZ_unique = np.unique(idZ_all)

    _, _, idZ_verteb = np.where(verteb_lesion)
    idZ_verteb_unique = np.unique(idZ_verteb)

    # Precompute derivatives of polynomial in world coordinates
    px_d = derivative(px)
    py_d = derivative(py)
    pz_d = derivative(pz)

    ssd_theta = np.zeros(len(thetas))
    pts = [np.empty((3, 0)) for _ in range(len(thetas))]
    coord_sectors_theta = [[np.empty((3, 0)) for _ in range(len(phis))] for _ in range(len(thetas))]
    # ------------------------------------------------------------------
    # THETA LOOP
    # ------------------------------------------------------------------
    for it, theta in enumerate(thetas):

        ssd = np.zeros(len(phis))
        cos_t = np.cos(np.deg2rad(theta))
        sin_t = np.sin(np.deg2rad(theta))

        for idZ in idZ_unique:

            # ----------------------------------------------------------
            # Slice world coordinate
            # ----------------------------------------------------------
            slice_z = idZ * CT.spacing[2] + CT.origin[2]

            try:
                t = t_from_z_world(slice_z, pz)
                tangent_vec = tangent_at_t(t, px_d, py_d, pz_d)
                tangent_vec_norm = tangent_vec / np.linalg.norm(tangent_vec)

                if tangent_vec_norm[2] < 0:
                    tangent_vec_norm = -tangent_vec_norm

                if idZ in idZ_verteb_unique:
                    cosine_similarities[it].append(
                        np.abs(np.dot(v_check, tangent_vec_norm))
                    )

                rotation_point_new = np.array([
                    np.polyval(px, t),
                    np.polyval(py, t),
                    np.polyval(pz, t)
                ])

            except Exception:
                continue

            # ----------------------------------------------------------
            # Transformation (theta = 0 here)
            # ----------------------------------------------------------
            T, Xt_world, Yt_world, Zt_world = apply_transformation(
                tangent_vec_norm,
                target_axis,
                rotation_point_new,
                Xf, Yf, Zf,
                theta=0
            )

            T_inv = np.linalg.inv(T)

            # ----------------------------------------------------------
            # Slice centering
            # ----------------------------------------------------------
            com_slice = np.array(com(sc[:, :, idZ])).astype(int)

            xt_slice = Xt_world[com_slice[0], com_slice[1], idZ]
            yt_slice = Yt_world[com_slice[0], com_slice[1], idZ]
            zt_slice = Zt_world[com_slice[0], com_slice[1], idZ]

            Xt0 = Xt_world[:, :, idZ] - xt_slice
            Yt0 = Yt_world[:, :, idZ] - yt_slice

            # ----------------------------------------------------------
            # In-plane rotation by theta
            # ----------------------------------------------------------
            Xt_rot = cos_t * Xt0 - sin_t * Yt0
            Yt_rot = sin_t * Xt0 + cos_t * Yt0

            radial = np.sqrt(Xt_rot ** 2 + Yt_rot ** 2)
            angular = np.degrees(np.arctan2(Yt_rot, Xt_rot))

            # ----------------------------------------------------------
            # Sector loop
            # ----------------------------------------------------------
            for s in range(len(phis)):

                phi_start = phis[s]
                phi_stop = phis[(s + 1) % len(phis)]

                if s < len(phis) - 1:
                    sector = (
                            (radial <= radius) &
                            (angular > phi_start) &
                            (angular <= phi_stop)
                    )
                else:
                    sector = (
                            (radial <= radius) &
                            ((angular > phi_start) | (angular <= phi_stop))
                    )

                if not np.any(sector):
                    continue

                # ------------------------------------------------------
                # Coordinates in transformed space
                # ------------------------------------------------------
                coords = np.vstack([
                    Xt_world[:, :, idZ][sector],
                    Yt_world[:, :, idZ][sector],
                    zt_slice * np.ones(sector.sum()),
                    np.ones(sector.sum())
                ])

                # Apply the inverse 4×4 transformation
                coords_inv = (T_inv @ coords)[:3, :]

                coord_sectors_theta[it][s] = np.hstack([coord_sectors_theta[it][s], coords_inv])

                # ------------------------------------------------------
                # Landmark SSD
                # ------------------------------------------------------
                if landmarks_xyz[s].size == 0:
                    continue

                mask = landmarks_ijk[s][:, 2] == idZ
                if not np.any(mask):
                    continue

                pts[s] = np.hstack([pts[s], landmarks_xyz[s][mask].T])  # (3, N)

                diff = coords_inv[:, :, None] - pts[s][:, None, :]
                dists = np.linalg.norm(diff, axis=0)

                ssd[s] += np.mean(dists, axis=0).sum() # loss function on the slice level

        # TODO:
        #diff = coord_sectors_theta[it][s][:, :, None] - pts[s][:, None, :]
        #dists = np.linalg.norm(diff, axis=0)
        #ssd[s] += np.mean(dists, axis=0).sum() # loss function on the sector level

        # --------------------------------------------------------------
        # Store global SSD for this theta
        # --------------------------------------------------------------
        ssd_theta[it] = ssd.sum()

    # ------------------------------------------------------------------
    # OPTIMAL THETA
    # ------------------------------------------------------------------

    best_idx = np.argmin(ssd_theta)

    best_theta = thetas[best_idx]
    best_ssd = ssd_theta[best_idx]
    coord_sectors = coord_sectors_theta[best_idx]

    print(f"Best theta = {best_theta:.2f} degrees")

    # Build KD-tree once
    tree = cKDTree(np.column_stack([Xf, Yf, Zf]))

    # Prepare output masks
    sectors = [np.zeros(CT.gridSize).astype(bool) for _ in range(len(phis))]

    # Process each sector
    for s, coord in enumerate(coord_sectors):
        # coord is shape (3, N)
        query_pts = coord.T  # shape (N, 3)

        # Fast nearest neighbor lookup
        dd, idxs = tree.query(query_pts, k=1)  # idxs shape (N,)

        # Convert flat indices to 3D
        i, j, k = np.unravel_index(idxs, CT.gridSize)

        # Mark all in bulk (no loop!)
        sectors[s][i, j, k] = True

        names_sector = f'Sector_{s}'
        RTs.setMask(names_sector, sectors[s], spacing=voxel_size, origin=origin)
        # smooth sector to remove small holes
        RTs.smoothMasks([names_sector])
        # reload
        sectors[s] = RTs.getMaskByName(names_sector).imageArray

    # check if coordinates are in sector

    for landmarks, sector, s in zip(landmarks_ijk, sectors, range(len(sectors))):

        if len(landmarks) > 0:
            i, j, k = landmarks.T
            if ~(sector[i, j, k].any()):
                print(f'Rotation needed for sector {s+1}')

    bs_sector = get_barrier_segments(sectors, gtv)
    print(f'Barrier structure segments are {bs_sector}')

    bs = np.zeros(CT.gridSize).astype(bool)
    for s in bs_sector:
        bs = np.logical_or(bs, sectors[s - 1]) # subtract 1 again

    # define barrier structures
    RTs.setMask('BS', np.logical_or(bs, ~RTs.getMaskByName('Vertebra').imageArray), spacing=voxel_size, origin=origin, roi_type='Barrier')
    #RTs.setMask('BS', ~RTs.getMaskByName('Vertebra').imageArray, spacing=voxel_size, origin=origin, roi_type='Barrier')

    # Run fast marching method for CTV

    CTV = CTVGeometric(rts=RTs, model={'model': None, 'obstacle': True})
    CTV.setCTV_metric(RTs.getMaskByName('CTV'), metric='dice', method='Nelder-Mead')
    #CTV.smoothMask()

    end = time.perf_counter()

    time_elapsed[idp] = (end - start)/60
    print(f"Elapsed time: {time_elapsed[idp]:.6f} minutes")

    # compute and store metrics for CTVs
    dice_ctv[idp] = dice_score(CTV.imageArray, RTs.getMaskByName('CTV').imageArray)
    dice_vert[idp] = dice_score(RTs.getMaskByName('Vertebra').imageArray, RTs.getMaskByName('Vertebra_GT').imageArray)

    SDS_ctv[idp] = surface_dice_score(CTV.imageArray, RTs.getMaskByName('CTV').imageArray, 2, voxel_spacing=CTV.spacing)
    SDS_vert[idp] = surface_dice_score(RTs.getMaskByName('Vertebra').imageArray, RTs.getMaskByName('Vertebra_GT').imageArray, 2, voxel_spacing=CTV.spacing)

    HD95_ctv[idp] = percentile_hausdorff_distance(CTV.getMeshpoints(), RTs.getMaskByName('CTV').getMeshpoints(), percentile=95, voxel_spacing=voxel_size)
    HD95_vert[idp] = percentile_hausdorff_distance(RTs.getMaskByName('Vertebra').getMeshpoints(), RTs.getMaskByName('Vertebra_GT').getMeshpoints(), percentile=95, voxel_spacing=voxel_size)

    print(f"DSC = {dice_ctv[idp]}")
    print(f"SDS (2 mm) = {SDS_ctv[idp]}")

    if visualization:
        # create 3D plots

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for name in names_Verteb_interest:
            verts, faces = RTs.getMarchingCubesMesh(name)
            mesh = Poly3DCollection(verts[faces], alpha=0.3, linewidth=0.1)
            mesh.set_facecolor((0.8, 0.3, 0.3))  # light red
            ax.add_collection3d(mesh)

        verts, faces = RTs.getMarchingCubesMesh('SpinalCord')
        mesh = Poly3DCollection(verts[faces], alpha=0.7, linewidth=0.1)
        mesh.set_facecolor('lightblue')
        ax.add_collection3d(mesh)

        verts, faces = RTs.getMarchingCubesMesh('GTV')
        mesh = Poly3DCollection(verts[faces], linewidth=0.3)
        mesh.set_facecolor('green')  # light green
        ax.add_collection3d(mesh)

        # 2) plot curve
        ax.plot(
            fitted_points[:, 0],
            fitted_points[:, 1],
            fitted_points[:, 2],
            color='blue',
            linewidth=3,
            label='Polynomial Fit'
        )

        # 3) axis labels
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title("3D Vertebra Mask (Mesh) + Polynomial Curve")

        # 4) equal aspect ratio
        all_points = np.vstack([verts, fitted_points])
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        centers = (mins + maxs) / 2
        max_range = (maxs - mins).max() / 2

        ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
        ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
        ax.set_zlim(centers[2] - max_range, centers[2] + max_range)

        ax.legend()
        plt.tight_layout()
        plt.show()

        # Create 2D plots

        gtv = RTs.getMaskByName('GTV').imageArray
        sc = RTs.getMaskByName('SpinalCord').imageArray

        # prepare figures
        plotCT = CT.imageArray.copy()
        plotGTV = gtv.astype(float).copy()
        plotGTV[~gtv] = np.NaN
        plotSC = sc.astype(float).copy()
        plotSC[~sc] = np.NaN
        plotBarrier = RTs.getMaskByName('BS').imageArray
        plotCTVmanual = RTs.getMaskByName('CTV').imageArray
        plotDistance = CTV.distance3D
        plotDistance[plotDistance>30] = np.NaN

        plotSectors = [sector.astype(float).copy() for sector in sectors]
        plotSectorsf = [sector.astype(float).copy() for sector in sectors]
        for plotSectorf, sector in zip(plotSectorsf, sectors):
            plotSectorf[~sector] = np.NaN

        fig = plt.figure()
        plt.axis('off')
        plt.imshow(plotCT[:, :, Z_coord].T, extent=[x[0], x[-1], y[0], y[-1]], cmap='gray', vmin=-200, vmax=200)
        plt.imshow(plotDistance[:, :, Z_coord].T, extent=[x[0], x[-1], y[0], y[-1]], cmap='jet', alpha=0.5)
        plt.contourf(np.flip(plotGTV[:, :, Z_coord].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='red', alpha=0.5)
        plt.show()

        # Build interactive plot

        def plot_isodistance_sagittal(ax, X, plotCTV):

            fig.add_axes(ax)
            ax.imshow(np.flip(plotCT[X, :, :].T, axis=0), extent=[y[0], y[-1], z[0], z[-1]], cmap='gray', vmin=-200, vmax=200)
            ax.contourf(plotGTV[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='red', alpha=0.5)
            ax.contour(plotCTVmanual[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='yellow', linewidths=1.5)

            ax.contour(plotCTV[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='green', linewidths=1.5)

            ax.contourf(plotSC[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='blue', alpha=0.5)
            #ax.contour(plotBarrier[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='white', linewidths=1.5)

            for alpha, plotSector in zip(alphas, plotSectorsf):
                ax.contourf(plotSector[X, :, :].T, extent=[y[0], y[-1], z[0], z[-1]], colors='orange', alpha=alpha)

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_isodistance_coronal(ax, Y, plotCTV):

            fig.add_axes(ax)
            ax.imshow(np.flip(plotCT[:, Y, :].T, axis=0), extent=[x[0], x[-1], z[0], z[-1]], cmap='gray', vmin=-200, vmax=200)
            ax.contourf(plotGTV[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='red', alpha=0.5)
            ax.contour(plotCTVmanual[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='yellow', linewidths=1.5)

            ax.contour(plotCTV[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='green', linewidths=1.5)

            ax.contourf(plotSC[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='blue', alpha=0.5)
            #ax.contour(plotBarrier[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='white', linewidths=1.5)

            for alpha, plotSector in zip(alphas, plotSectorsf):
                ax.contourf(plotSector[:, Y, :].T, extent=[x[0], x[-1], z[0], z[-1]], colors='orange', alpha=0.5)

            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        def plot_isodistance_axial(ax, Z, plotCTV):

            fig.add_axes(ax)
            ax.imshow(plotCT[:, :, Z].T, extent=[x[0], x[-1], y[0], y[-1]], cmap='gray', vmin=-200, vmax=200)
            ax.contourf(np.flip(plotGTV[:, :, Z].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='red', alpha=0.5)

            for alpha, plotSector, lm_ijk, lm_xyz in zip(alphas, plotSectors, landmarks_ijk, landmarks_xyz):
                ax.contour(np.flip(plotSector[:, :, Z].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='orange', alpha=alpha, linewidths=1., linestyles='dashed')
                #ax.contourf(np.flip(plotSector[:, :, Z].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='orange', alpha=alpha)

                if lm_xyz.size > 0:
                    mask = lm_ijk[:, 2] == Z
                    if np.any(mask):
                        xy = lm_xyz[mask][:, :2]
                        xs = xy[:, 0]
                        ys = xy[:, 1]

                        ax_vox = ax.inset_axes([0, 0, 1, 1], transform=ax.transAxes)

                        ax_vox.patch.set_visible(False)
                        ax_vox.set_facecolor('none')
                        ax_vox.set_frame_on(False)

                        ax_vox.set_xticks([])
                        ax_vox.set_yticks([])

                        ax_vox.scatter(xs, ys, s=20, c='cyan', marker='o', edgecolors='cyan', linewidths=0.5, zorder=10)
                        ax_vox.set_xlim(x[0], x[-1])
                        ax_vox.set_ylim(y[0], y[-1])
                        ax_vox.invert_yaxis()

            ax.contour(np.flip(plotCTVmanual[:, :, Z].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='magenta', linewidths=1.5)
            ax.contour(np.flip(plotCTV[:, :, Z].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='green', linewidths=1.5)
            ax.contourf(np.flip(plotSC[:, :, Z].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='blue', alpha=0.5)
            #ax.contour(np.flip(plotBarrier[:, :, Z].T, axis=0), extent=[x[0], x[-1], y[0], y[-1]], colors='white', linewidths=1.5)

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
            valmax=30,
            valinit=CTV.isodistance,
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

plt.figure()
for cosines in cosine_similarities:
    plt.hist(
        cosines,
        bins=20,
        range=(0.95, 1),
        alpha=0.5
    )

plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.title("Overlaid Histograms of Cosine Similarities")
plt.show()

# calculate statistics
dice_ctv_mean, dice_ctv_std = np.mean(dice_ctv), np.std(dice_ctv)
SDS_ctv_mean, SDS_ctv_std= np.mean(SDS_ctv), np.std(SDS_ctv)
HD95_ctv_mean, HD95_ctv_std = np.mean(HD95_ctv), np.std(HD95_ctv)

dice_vert_mean, dice_vert_std = np.mean(dice_vert), np.std(dice_vert)
SDS_vert_mean, SDS_vert_std= np.mean(SDS_vert), np.std(SDS_vert)
HD95_vert_mean, HD95_vert_std = np.mean(HD95_vert), np.std(HD95_vert)

print('-----------------')

print(f"CTV mean dice: {dice_ctv_mean*100:.2f}% +/- {dice_ctv_std*100:.2f}%")
print(f"CTV mean surface dice: {SDS_ctv_mean*100:.2f}% +/- {SDS_ctv_std*100:.2f}%")
print(f"CTV mean HD95 : {HD95_ctv_mean:.2f} mm +/- {HD95_ctv_std:.2f} mm")

print('-----------------')

print(f"Vertebra mean dice: {dice_vert_mean*100:.2f}% +/- {dice_vert_std*100:.2f}%")
print(f"Vertebra mean surface dice: {SDS_vert_mean*100:.2f}% +/- {SDS_vert_std*100:.2f}%")
print(f"Vertebra mean HD95 : {HD95_vert_mean:.2f} mm +/- {HD95_vert_std:.2f} mm")

print('-----------------')

print(f"Mean elapsed time: {np.mean(time_elapsed):.6f} minutes")