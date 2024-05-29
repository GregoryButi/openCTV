#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:56:55 2023

@author: gregory
"""
import numpy as np

from opentps.core.data.images._image3D import Image3D
from Process.Solvers import Solver
from Process.Tensors import TensorDiffusion

class SolverPDE(Solver):
    def __init__(self, source=None, barriers=None, tensor=None, domain=None):
        
      if tensor is None:
          DT = np.zeros(tuple(source.gridSize) + (3,3))
          DT[..., 0:3, 0:3] = np.eye(3)          
          tensor = TensorDiffusion(imageArray=DT, spacing=source.spacing, origin=source.origin)
        
      super().__init__(source=source, barriers=barriers, tensor=tensor, domain=domain)
    
    def getDensity(self, timepoint, dict, deltat = 0.01):
            
        # get diffusion coefficients      
        Dxx = self.tensor.imageArray[:, :, :, 0, 0]
        Dxy = self.tensor.imageArray[:, :, :, 0, 1]
        Dxz = self.tensor.imageArray[:, :, :, 0, 2]
        Dyy = self.tensor.imageArray[:, :, :, 1, 1]
        Dyz = self.tensor.imageArray[:, :, :, 1, 2]
        Dzz = self.tensor.imageArray[:, :, :, 2, 2]
            
        # Get grid positions in world space
        x, y, z = self.tensor.getMeshGridAxes()
        
        #initialize
        cells_list = []
        cells = self.source.imageArray*dict['cell_capacity']
        iplot = 0
        for t in np.linspace(0, timepoint[-1], num=int(timepoint[-1]/deltat), endpoint=True) :
            
            # store metrics
            if t >= timepoint[iplot]:
                
                cells_list.append(cells)
                iplot += 1
            
            if np.count_nonzero(np.isnan(cells))>0:
                print("Loop paused due to np.nan")
                breakpoint()
                break 
                     
            # compute cell gradients
            dx_cells, dy_cells, dz_cells = np.gradient(cells, x, y, z)   
            
            diff_term = np.gradient(Dxx * dx_cells + Dxy * dy_cells + Dxz * dz_cells, x, axis=0) + \
                        np.gradient(Dxy * dx_cells + Dyy * dy_cells + Dyz * dz_cells, y, axis=1) + \
                        np.gradient(Dxz * dx_cells + Dyz * dy_cells + Dzz * dz_cells, z, axis=2)    
                        
            if dict['system'] == 'reaction_diffusion': 
                react_term = dict['proliferation_rate'] * np.multiply(cells, (1 - cells / dict['cell_capacity']))   
            elif dict['system'] == 'diffusion': 
                react_term = 0
                
            cells_next = cells + deltat * (diff_term + react_term)
            cells = cells_next
                
            print("time: " + str(t))
            
        return cells_list
    
    def getDensity_xyz(self, timepoint, transform, domain_uvw, dict, deltat = 0.01):
        
        # Find the minimum and maximum indices along each dimension
        min_indices = np.min(np.where(self.domain), axis=1)
        max_indices = np.max(np.where(self.domain), axis=1)
        
        # Get diffusion coefficients
        Dxx = self.tensor.imageArray[:, :, :, 0, 0]
        Dxy = self.tensor.imageArray[:, :, :, 0, 1]
        Dxz = self.tensor.imageArray[:, :, :, 0, 2]
        Dyy = self.tensor.imageArray[:, :, :, 1, 1]
        Dyz = self.tensor.imageArray[:, :, :, 1, 2]
        Dzz = self.tensor.imageArray[:, :, :, 2, 2]
        
        # Get grid positions in world space
        x, y, z = self.tensor.getMeshGridAxes()
                        
        # initialize
        cells_list = []
        cells = self.source.imageArray*dict['cell_capacity']
        iplot = 0
        for t in np.linspace(0, timepoint[-1], num=int(timepoint[-1]/deltat), endpoint=True) :
            
            # store metrics
            if t >= timepoint[iplot]:
                
                cells_codomain = np.zeros(transform.mapping.domain_shape) # initialize
                cells_codomain[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2]] = cells
                # transform cell distribution back to local space
                cells_domain = Image3D(imageArray=transform.mapping.transform(cells_codomain))
                cells_domain.reduceGrid_mask(domain_uvw)     
                
                cells_list.append(cells_domain.imageArray)
                iplot += 1
            
            if np.count_nonzero(np.isnan(cells))>0:
                print("Loop paused due to np.nan")
                breakpoint()
                break 
                     
            # compute cell gradients
            dx_cells, dy_cells, dz_cells = np.gradient(cells, x, y, z)   
            
            diff_term = np.gradient(Dxx * dx_cells + Dxy * dy_cells + Dxz * dz_cells, x, axis=0) + \
                        np.gradient(Dxy * dx_cells + Dyy * dy_cells + Dyz * dz_cells, y, axis=1) + \
                        np.gradient(Dxz * dx_cells + Dyz * dy_cells + Dzz * dz_cells, z, axis=2)    
                   
            if dict['system'] == 'reaction_diffusion': 
                react_term = dict['proliferation_rate'] * np.multiply(cells, (1 - cells / dict['cell_capacity']))   
            elif dict['system'] == 'diffusion': 
                react_term = 0
                
            cells_next = cells + deltat * (diff_term + react_term)
            cells = cells_next
                
            print("time: " + str(t))
            
        return cells_list    

    def getDensity_uvw(self, timepoint, transform, dict, deltat = 0.01):
            
        # get Jacobian determinant and reduce image
        tmp = Image3D(imageArray=transform.getJacobianDeterminantDomain())
        tmp.reduceGrid_mask(self.domain)
        jac_det_domain = tmp.imageArray
                
        # get diffusion coefficients    
        Duu = self.tensor.imageArray[:, :, :, 0, 0]
        Duv = self.tensor.imageArray[:, :, :, 0, 1]
        Duw = self.tensor.imageArray[:, :, :, 0, 2]
        Dvv = self.tensor.imageArray[:, :, :, 1, 1]
        Dvw = self.tensor.imageArray[:, :, :, 1, 2]
        Dww = self.tensor.imageArray[:, :, :, 2, 2]
        
        # get grid positions in world space (CHECK THIS!!!)
        u, v, w = self.tensor.getMeshGridAxes()
                    
        #initialize
        cells_list = []
        cells = self.source.imageArray*dict['cell_capacity']
        iplot = 0
        for t in np.linspace(0, timepoint[-1], num=int(timepoint[-1]/deltat), endpoint=True) :
            
            # store metrics
            if t >= timepoint[iplot]: 
                
                cells_list.append(cells)
                iplot += 1
            
            if np.count_nonzero(np.isnan(cells))>0:
                print("Loop paused due to np.nan")
                breakpoint()
                break 
                           
            # compute cell gradients
            du_cells, dv_cells, dw_cells = np.gradient(cells, u, v, w, edge_order=2) 
            
            diff_term = np.gradient(np.multiply(jac_det_domain, Duu * du_cells + Duv * dv_cells + Duw * dw_cells), u, axis=0) + \
                        np.gradient(np.multiply(jac_det_domain, Duv * du_cells + Dvv * dv_cells + Dvw * dw_cells), v, axis=1) + \
                        np.gradient(np.multiply(jac_det_domain, Duw * du_cells + Dvw * dv_cells + Dww * dw_cells), w, axis=2)    
            
            if dict['system'] == 'reaction_diffusion': 
                react_term = dict['proliferation_rate'] * np.multiply(cells, (1 - cells / dict['cell_capacity']))     
            elif dict['system'] == 'diffusion': 
                react_term = 0
            
            cells_next = cells + deltat * (np.divide(diff_term, jac_det_domain) + react_term)
            
            cells = cells_next
                
            print("time: " + str(t))
            
        return cells_list