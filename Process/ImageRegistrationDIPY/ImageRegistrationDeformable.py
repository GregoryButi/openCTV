from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric, SSDMetric

from Process.ImageRegistrationDIPY import ImageRegistration, ImageRegistrationRigid

class ImageRegistrationDeformable(ImageRegistration):
    def __init__(self, static, static_grid2world, moving, moving_grid2world, static_mask=None, moving_mask=None, level_iters=[10, 10, 5], metric='CC'):
        super().__init__(static, static_grid2world, moving, moving_grid2world, static_mask=static_mask, moving_mask=moving_mask)

        self.rigid_registration = ImageRegistrationRigid(static=static, static_grid2world=static_grid2world, moving=moving, moving_grid2world=moving_grid2world, static_mask=static_mask, moving_mask=moving_mask)
        self.level_iters = level_iters
        self.metric = metric

    @property
    def mapping(self):
        if self._mapping is None:
            self._mapping = self.get_mapping()
        return self._mapping

    def get_mapping(self):
        # Perform rigid registration
        rigid = self.rigid_registration.get_mapping()

        # [STAGE 4]
        if self.metric == 'CC':
            metric = CCMetric(3)  # Cross Correlation
        if self.metric == 'SSD':
            metric = SSDMetric(3, smooth=8)
        sdr = SymmetricDiffeomorphicRegistration(metric, self.level_iters)
        mapping = sdr.optimize(self.static, self.moving, self.static_grid2world, self.moving_grid2world, prealign=rigid.affine)

        return mapping