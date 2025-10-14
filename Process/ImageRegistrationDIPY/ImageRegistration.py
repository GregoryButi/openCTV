

class ImageRegistration(object):

    def __init__(self, static=None, static_grid2world=None, moving=None, moving_grid2world=None, static_mask=None, moving_mask=None):
        self._static = static
        self._moving = moving
        self._static_grid2world = static_grid2world
        self._moving_grid2world = moving_grid2world
        self._static_mask = static_mask
        self._moving_mask = moving_mask
        self._mapping = None

    @property
    def static(self):
        return self._static

    @static.setter
    def static(self, array):
        self._static = array

    @property
    def moving(self):
        return self._moving

    @moving.setter
    def moving(self, array):
        self._moving = array

    @property
    def static_grid2world(self):
        return self._static_grid2world

    @static_grid2world.setter
    def static_grid2world(self, array):
        self._static_grid2world = array

    @property
    def moving_grid2world(self):
        return self._moving_grid2world

    @moving_grid2world.setter
    def moving_grid2world(self, array):
        self._moving_grid2world = array

    @property
    def static_mask(self):
        return self._static_mask

    @static_mask.setter
    def static_mask(self, array):
        self._static_mask = array

    @property
    def moving_mask(self):
        return self._moving_mask

    @moving_mask.setter
    def moving_mask(self, array):
        self._moving_mask = array

    @property
    def mapping(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @mapping.setter
    def mapping(self, value):
        self._mapping = value

    def getImageTransformed(self, image):
        warped = self.mapping.transform(image)
        return warped

    def getImageTransformedInverse(self, image):
        warped = self.mapping.transform_inverse(image)
        return warped