#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:54:34 2023

@author: gregory
"""

import numpy as np
import networkx as nx
from tqdm import tqdm

from opentps.core.data.images._image3D import Image3D

from Process.Solvers import Solver
from Process.Tensors import TensorMetric

class SolverDijkstra(Solver):
    def __init__(self, source=None, boundary=None, tensor=None, domain=None):
      
      if tensor is None:
          MT = np.zeros(tuple(source.gridSize) + (3, 3))
          MT[..., 0:3, 0:3] = np.eye(3)          
          tensor = TensorMetric(imageArray=MT, spacing=source.spacing, origin=source.origin)

      if boundary is None:
          boundary = Image3D(imageArray=np.zeros(source.gridSize).astype(bool), spacing=source.spacing, origin=source.origin)

      super().__init__(source=source, boundary=boundary, tensor=tensor, domain=domain)

    def getDistance(self):

        # Define target / walls
    
        target = self.source.imageArray
        wall = np.logical_and(self.boundary.imageArray, ~target)
        metric = self.tensor.imageArray

        # Convert list to a set of tuples for efficient lookup
        targetContourSet = set(map(tuple, self.source.getMeshpoints()))

        # get physical domain positions
        X_world, Y_world, Z_world = self.source.getMeshGridPositions()

        # 26 nearest neighbours in a cubic grid
        nearest_neighbours = [(-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0), (-1, 0, 1), (-1, 1, -1),
                              (-1, 1, 0), (-1, 1, 1), (0, -1, -1), (0, -1, 0), (0,-1, 1), (0, 0,-1), (0, 0, 1), (0, 1, -1),
                              (0, 1, 0), (0, 1, 1), (1, -1, -1), (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 0), (1, 0, 1),
                              (1, 1, -1), (1, 1, 0), (1, 1, 1)]

        # construct undirected graph
        graph, target_nodes = getGraph(target, targetContourSet, wall, nearest_neighbours, metric, X_world, Y_world, Z_world)

        # Perform multi-source shortest path calculation

        shortest_path_lengths = nx.multi_source_dijkstra_path_length(graph, sources=target_nodes, weight="weight")

        # Initialize with infinity outside and zero within target
        distance3D = np.where(target, 0, np.inf)

        # Iterate over all nodes in the graph
        for node, path_length in shortest_path_lengths.items():
            # Get the 3D coordinates of the node
            i, j, k = node  # These should be the indices in the X_world, Y_world, Z_world grids

            # Assign the shortest path length to the corresponding position in the 3D array
            distance3D[i, j, k] = path_length

        if self.domain is not None:
            distance3D = self.getArray_fullGrid(distance3D)

        return distance3D

# Helper function to check proximity
def is_within_proximity(node, set, threshold=(0.1, 0.1, 0.1)):
    for n in set:
        if all(abs(node[i] - n[i]) <= threshold[i] for i in range(3)):
            return True
    return False

def getGraph(targetMask, targetContourSet, wall, nearest_neighbours, metric, X, Y, Z):

    # construct graph
    graph = nx.Graph()

    # Get the shape of the meshgrid
    shape = targetMask.shape

    # Calculate the total number of iterations for the progress bar
    total_iterations = shape[0] * shape[1] * shape[2]

    # Create progress bar
    with tqdm(total=total_iterations, desc="Processing nodes", unit="node") as pbar:

        target_nodes = []
        # Iterate through each node in the meshgrid
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    # Update the progress bar
                    pbar.update(1)

                    # Current node
                    current_node = (i, j, k)
                    #node_coords = (X[i, j, k], Y[i, j, k], Z[i, j, k])

                    # Skip adding the node if it is marked as wall or target,
                    # unless its coordinates are in targetContourSet
                    if wall[i, j, k]:
                        continue

                    #if targetMask[i, j, k] and not is_within_proximity(node_coords, targetContourSet):
                        #continue

                    #if is_within_proximity(node_coords, targetContourSet):
                        #target_nodes.append(current_node)

                    if targetMask[i, j, k]:
                        target_nodes.append(current_node)


                    # Add the node to the graph
                    graph.add_node(current_node)

                    # Add edges to nearest neighbors with weights as Euclidean distances
                    for di, dj, dk in nearest_neighbours:
                        ni, nj, nk = i + di, j + dj, k + dk
                        # Ensure the neighbor is within bounds
                        if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
                            #neighbor_coords = (X[ni, nj, nk], Y[ni, nj, nk], Z[ni, nj, nk])

                            # Skip adding edge if the neighbor is marked as wall or target,
                            # unless its coordinates are in targetContourSet
                            #if (wall[ni, nj, nk] or targetMask[ni, nj, nk]) and neighbor_coords not in targetContourSet:
                                #continue

                            # Skip adding edge if the neighbor is marked as wall
                            if wall[ni, nj, nk]:
                                continue

                            # Calculate the weight

                            # unit vector
                            n = np.array([X[i, j, k] - X[ni, nj, nk],
                                          Y[i, j, k] - Y[ni, nj, nk],
                                          Z[i, j, k] - Z[ni, nj, nk]])

                            # average (better alternative?)
                            mt = (metric[i, j, k] + metric[ni, nj, nk]) / 2

                            # projection of metric tensor on unit vector
                            distance = np.sqrt(np.dot(n, np.dot(mt, n))) # Eq.(2) in Bortfeld and Buti (2022)

                            # Add the edge with the weight
                            graph.add_edge(current_node, (ni, nj, nk), weight=distance)

        # check
        if len(target_nodes) > len(targetContourSet):
            print("Warning: not all surface target points are in the graph")

        return graph, target_nodes