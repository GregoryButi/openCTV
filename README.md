INSTALLATION

Install dependent packages in a virtual environment

    pip3 install -r requirements.txt

MODIFICATION OF EXISTING PACKAGES

After installing the openTPS package, add the following functions to the Image3D class in opentps.core.data.images._image3D:

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
        """crop image using the non zero indices of an image given as input and update origin.

        Parameters
        ----------
        otherImage : numpy array
        outputType : numpy data type
            type of the output.
        """

        # Find the minimum and maximum indices along each dimension
        min_indices = np.min(np.where(mask), axis=1)
        max_indices = np.max(np.where(mask), axis=1)
        x, y, z = self.getMeshGridAxes()

        self.imageArray = self._imageArray[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1],
                          min_indices[2]:max_indices[2], ...]
        self.origin = (x[min_indices[0]], y[min_indices[1]], z[min_indices[2]])

    def intensityScaling(self, mode='min_max'):
        """scale image values between zero and 1.

        Parameters
        ----------
        mode : string
        outputType : numpy data type
            type of the output.
        """

        if mode == 'min_max':
            # Perform min-max scaling imageArray, scaling the values between 0 and 1.
            self.imageArray = (self._imageArray - self.min()) / (self.max() - self.min())

        elif mode == 'robust':
            # Perform robust scaling on the input data using the interquartile range (IQR).
            q1 = np.percentile(self._imageArray, 25)
            q3 = np.percentile(self._imageArray, 75)
            iqr = q3 - q1

            self.imageArray = (self._imageArray - self.median()) / iqr

After installing the DIPY package, replace the following lines in get_simplified_transform function of DiffeomorphicMap class in imwarp file:

line 939
    self.codomain_shape
line 944
    Dinv = self.domain_grid2world
line 951
    self.domain_shape
    

