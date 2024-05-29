#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:49:53 2023

@author: gregory
"""

import numpy as np

def dice_score(A, B):
    """
    Compute the Dice score between two binary masks A and B.
    """
    intersection = np.logical_and(A, B)
    score = 2.0 * intersection.sum() / (A.sum() + B.sum())
    return score


def jaccard_index(A, B):
    """
    Compute the Jaccard index between two binary masks A and B.
    """
    intersection = np.logical_and(A, B)
    union = np.logical_or(A, B)
    score = intersection.sum() / union.sum()
    return score


from scipy.spatial import cKDTree


def hausdorff_distance(A, B):
    """
    Compute the Hausdorff distance between two sets of points A and B.
    """
    # Build KD-trees for A and B
    tree_A = cKDTree(A)
    tree_B = cKDTree(B)

    # Compute one-directional Hausdorff distance
    distances_A = tree_A.query(B)[0]
    distances_B = tree_B.query(A)[0]
    hausdorff_1 = np.max(distances_A)
    hausdorff_2 = np.max(distances_B)

    # Return maximum of one-directional Hausdorff distances
    return max(hausdorff_1, hausdorff_2)


def percentile_hausdorff_distance(A, B, percentile):
    """
    Compute the percentile Hausdorff distance between two sets of points A and B.
    """
    # Build KD-trees for A and B
    tree_A = cKDTree(A)
    tree_B = cKDTree(B)

    # Compute one-directional Hausdorff distance
    distances_A = tree_A.query(B)[0]
    distances_B = tree_B.query(A)[0]
    hausdorff_1 = np.percentile(distances_A, percentile)
    hausdorff_2 = np.percentile(distances_B, percentile)

    # Return maximum of one-directional Hausdorff distances
    return max(hausdorff_1, hausdorff_2)


def mean_hausdorff_distance(A, B):
    """
    Compute the mean Hausdorff distance between two sets of points A and B.
    """
    # Build KD-trees for A and B
    tree_A = cKDTree(A)
    tree_B = cKDTree(B)

    # Compute one-directional Hausdorff distance
    distances_A = tree_A.query(B)[0]
    distances_B = tree_B.query(A)[0]
    hausdorff_1 = np.mean(distances_A)
    hausdorff_2 = np.mean(distances_B)

    # Return maximum of one-directional Hausdorff distances
    return max(hausdorff_1, hausdorff_2)