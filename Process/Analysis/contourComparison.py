#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:49:53 2023

@author: gregory
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

def dice_score(A, B):
    """
    Compute the Dice score between two binary masks A and B.
    """
    intersection = np.logical_and(A, B)
    score = 2.0 * intersection.sum() / (A.sum() + B.sum())
    return score

def surface_dice_score(A, B, tolerance_mm, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Compute the Surface Dice Score (SDS) between two binary masks A and B.

    Parameters:
    A, B : ndarray
        Binary masks to compare.
    tolerance_mm : float
        Maximum surface distance in mm to consider surfaces overlapping.
    voxel_spacing : tuple of floats
        Physical spacing of voxels in each dimension.

    Returns:
    float
        Surface Dice Score.
    """
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape.")

    # Compute surface masks
    A_surface = np.logical_xor(A, np.pad(A, 1, mode='constant')[:-2, 1:-1, 1:-1])
    B_surface = np.logical_xor(B, np.pad(B, 1, mode='constant')[:-2, 1:-1, 1:-1])

    # Compute distance maps
    A_dist = distance_transform_edt(~A_surface, sampling=voxel_spacing)
    B_dist = distance_transform_edt(~B_surface, sampling=voxel_spacing)

    # Count surface points within tolerance
    A_close_to_B = (A_surface & (B_dist <= tolerance_mm)).sum()
    B_close_to_A = (B_surface & (A_dist <= tolerance_mm)).sum()

    # Calculate surface dice score
    sds = (A_close_to_B + B_close_to_A) / (A_surface.sum() + B_surface.sum())

    return sds

def jaccard_index(A, B):
    """
    Compute the Jaccard index between two binary masks A and B.
    """
    intersection = np.logical_and(A, B)
    union = np.logical_or(A, B)
    score = intersection.sum() / union.sum()
    return score

def compute_bidirectional_distance_volume(surface_A, surface_B, volume_shape, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Compute the bidirectional local distance for each surface point in A to B
    and store them in a 3D array corresponding to volume A.

    Parameters:
        surface_A (numpy array): Nx3 array of surface points from volume A
        surface_B (numpy array): Mx3 array of surface points from volume B
        volume_shape (tuple): Shape of volume A (Dx, Dy, Dz)

    Returns:
        distance_volume (numpy array): 3D array where each voxel stores the bidirectional distance
    """
    # Create KD-Trees
    tree_B = cKDTree(surface_B)
    tree_A = cKDTree(surface_A)

    # Compute nearest neighbor distances
    distances_A_to_B, _ = tree_B.query(surface_A)
    #distances_B_to_A, _ = tree_A.query(surface_B)

    # Compute bidirectional distance (average)
    #bidirectional_distances = (distances_A_to_B + distances_B_to_A) / 2
    bidirectional_distances = distances_A_to_B

    # Initialize 3D array to store distances
    distance_volume = np.zeros(volume_shape, dtype=np.float32)

    # Convert surface points to voxel indices (assuming they are already in voxel space) with neirest neighbour approach
    voxel_coords = np.round(surface_A/voxel_spacing).astype(int)  # Convert to nearest voxel index

    # Ensure indices are within bounds
    valid_mask = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < volume_shape[0]) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < volume_shape[1]) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < volume_shape[2])
    )

    # Assign distances to corresponding voxel locations
    distance_volume[voxel_coords[valid_mask, 0], voxel_coords[valid_mask, 1], voxel_coords[valid_mask, 2]] = \
    bidirectional_distances[valid_mask]

    return distance_volume

def percentile_hausdorff_distance(A, B, percentile, voxel_spacing=None):
    """
    Compute the percentile Hausdorff distance between two sets of points A and B,
    accounting for voxel spacing if provided.

    Parameters
    ----------
    A : array-like, shape (n_points, ndim)
        Coordinates of the first set of points (in voxel indices).
    B : array-like, shape (m_points, ndim)
        Coordinates of the second set of points (in voxel indices).
    percentile : float
        Percentile for the Hausdorff distance (e.g., 95 for HD95).
    voxel_spacing : tuple or list, optional
        Voxel spacing along each axis (e.g. (sx, sy, sz)).
        If None, isotropic unit spacing is assumed.
    """
    A = np.asarray(A)
    B = np.asarray(B)

    # Apply voxel spacing if provided
    if voxel_spacing is not None:
        spacing = np.asarray(voxel_spacing)
        A = A * spacing
        B = B * spacing

    # Build KD-trees for A and B
    tree_A = cKDTree(A)
    tree_B = cKDTree(B)

    # Compute one-directional distances
    distances_A = tree_A.query(B)[0]
    distances_B = tree_B.query(A)[0]

    hausdorff_1 = np.percentile(distances_A, percentile)
    hausdorff_2 = np.percentile(distances_B, percentile)

    # Return maximum of one-directional Hausdorff distances
    return max(hausdorff_1, hausdorff_2)