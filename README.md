INSTALLATION (tested for python 3.12)

INSTALL DEPENDENT PACKAGES IN A VIRTUAL ENVIRONMENT

  pip install -r requirements.txt

INSTALL FAST MARCHING METHOD FROM https://github.com/Mirebeau/HamiltonFastMarching. SETUP:

  1. Clone HamiltonFastMarching
  2. Build binaries in ~/HamiltonFastMarching/Interfaces/FileHFM
  3. Move all FileHFM* executables to ~/HamiltonFastMarching/Interfaces/
  4. Set export FILEHFM_BINARY_DIR="~/HamiltonFastMarching/Interfaces/"

MODIFICATION OF EXISTING PACKAGES



After installing the opentps package, inside the package, do the following:

1. add the following argument to the __init__ function in the ROIMask class in opentps/core/data/images/_roiMask.py: grid2world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
2. add the following function to the ROIMask class in opentps/core/data/images/_roiMask.py:

    def getMeshpoints(self):
        polygonMeshList = self.getROIContour().polygonMesh

        polygonMeshArray = np.empty((0, 3), int)
        for zSlice in polygonMeshList:
            for point in np.arange(0, len(zSlice), 3):
                meshpoint = np.zeros((1, 3))
                meshpoint[0, 0] = zSlice[point]
                meshpoint[0, 1] = zSlice[point + 1]
                meshpoint[0, 2] = zSlice[point + 2]

                polygonMeshArray = np.append(polygonMeshArray, meshpoint, axis=0)

        return polygonMeshArray

3. add the following argument to the __init__ function in the Image3D class in opentps/core/data/images/_image3D.py: grid2world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], and add the following attribute self._grid2world = np.array(grid2world).
4. add the following functions to the Image3D class in opentps/core/data/images/_image3D.py:

    @property
    def grid2world(self) -> np.ndarray:
        return self._grid2world
    
    @grid2world.setter
    def grid2world(self, grid2world):
        self._grid2world = np.array(grid2world)
        self.dataChangedSignal.emit()
    
    def median(self):
        """
        Get the median value of the image array.
    
        Returns
        -------
        float
            Median value of the image array.
        """
        return np.median(self._imageArray)
    
    def getVoxelGridPositions(self) -> np.ndarray:
        """
        Get the voxel grid positions of the image in integers.
    
        Returns
        -------
        np.ndarray
            Voxel grid positions of the image in integers.
        """
        x = np.arange(self.gridSize[0])
        y = np.arange(self.gridSize[1])
        z = np.arange(self.gridSize[2])
        return np.meshgrid(x, y, z, indexing='ij')
    
    def getMeshGridAxes(self) -> np.ndarray:
        """
        Get the mesh grid axes of the image in mm.
    
        Returns
        -------
        np.ndarray
            Mesh grid axes of the image in mm.
        """
        x = self.origin[0] + np.arange(self.gridSize[0]) * self.spacing[0]
        y = self.origin[1] + np.arange(self.gridSize[1]) * self.spacing[1]
        z = self.origin[2] + np.arange(self.gridSize[2]) * self.spacing[2]
        return x, y, z
    
    def getVoxelGridAxes(self) -> np.ndarray:
        """
        Get the voxel grid axes of the image in integers.
    
        Returns
        -------
        np.ndarray
            Voxel grid axes of the image in integers.
        """
        x = np.arange(self.gridSize[0])
        y = np.arange(self.gridSize[1])
        z = np.arange(self.gridSize[2])
        return x, y, z
    
    def reduceGrid_mask(self, mask):
        """Crop image using the non-zero indices of an image given as input and update origin."""
        min_indices = np.min(np.where(mask), axis=1)
        max_indices = np.max(np.where(mask), axis=1)
        x, y, z = self.getMeshGridAxes()
    
        self.imageArray = self._imageArray[
            min_indices[0]:max_indices[0] + 1,
            min_indices[1]:max_indices[1] + 1,
            min_indices[2]:max_indices[2] + 1,
            ...
        ]
    
        self.origin = (
            x[min_indices[0]],
            y[min_indices[1]],
            z[min_indices[2]]
        )
    
    def intensityScaling(self, mode='min_max'):
        """Scale image values between zero and 1."""
        if mode == 'min_max':
            self.imageArray = (
                (self._imageArray - self.min()) /
                (self.max() - self.min())
            )
    
        elif mode == 'robust':
            q1 = np.percentile(self._imageArray, 5)
            q3 = np.percentile(self._imageArray, 95)
            iqr = q3 - q1
            self.imageArray = (
                (self._imageArray - self.median()) / iqr
            )
          
After installing the DIPY package, replace the following the line Dinv = self.domain_grid2world in get_simplified_transform function of DiffeomorphicMap class in imwarp file by Dinv = self.domain_grid2world

IMAGE SEGMENTATION

Follow the following steps if you want to generate CTV for the brain using a constrained distance tranform where the expansion is constrained by autosegmented brain barrier strucures (as in the paper "Buti et al (2025) - Clinical target volumes for glioma–Automated delineation to improve neuroanatomic consistency" (https://doi.org/10.1016/j.phro.2025.100865)):

1. Please download the model training results from folder https://huggingface.co/GregoryButi/CT_Brain_Segmentation_Radiotherapy
2. Create a folder in your Python project starting with "Dataset_" (in order to use the nnUNet inference scripts available in this GitHub), for example "~/openCTV/Models/nnUNet_results/Dataset_CT_Brain_Segmentation_Radiotherapy".
3. Create a folder with the patient data (e.g. ~/openCTV/Input) and upload nifti files of a CT image and GTV mask.
3. Run the script Workflows/Brain/CTV_segmentationBarriers_slider.py or Workflows/Brain/CTV_barriers_slider_validation.py for comparison to TotalSegmentator. Please consult https://github.com/wasserth/TotalSegmentator for obtaining the license of the brain_structures model.
