from dipy.align.imaffine import (transform_centers_of_mass,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D)

from Process.ImageRegistrationDIPY import ImageRegistration

class ImageRegistrationRigid(ImageRegistration):
    def __init__(self, static=None, static_grid2world=None, moving=None, moving_grid2world=None, static_mask=None, moving_mask=None):
        super().__init__(static=static, static_grid2world=static_grid2world, moving=moving, moving_grid2world=moving_grid2world, static_mask=static_mask, moving_mask=moving_mask)

    @property
    def mapping(self):
        if self._mapping is None:
            self._mapping = self.get_mapping()
        return self._mapping

    def get_mapping(self):

        # Perform registration

        level_iters = [10000, 1000, 100]
        sigmas = [3.0, 1.0, 0.0]
        factors = [4, 2, 1]

        nbins = 32
        sampling_prop = None
        metric = MutualInformationMetric(nbins, sampling_prop)
        c_of_mass = transform_centers_of_mass(self.static, self.static_grid2world,
                                              self.moving, self.moving_grid2world)

        affreg = AffineRegistration(metric=metric,
                                    level_iters=level_iters,
                                    sigmas=sigmas,
                                    factors=factors)
        # [STAGE 1]

        transform = TranslationTransform3D()
        params0 = None
        starting_affine = c_of_mass.affine
        translation = affreg.optimize(self.static, self.moving, transform, params0,
                                      self.static_grid2world, self.moving_grid2world,
                                      starting_affine=starting_affine, static_mask=self.static_mask,
                                      moving_mask=self.moving_mask)

        # [STAGE 2]

        transform = RigidTransform3D()
        params0 = None
        starting_affine = translation.affine
        mapping = affreg.optimize(self.static, self.moving, transform, params0,
                                  self.static_grid2world, self.moving_grid2world,
                                  starting_affine=starting_affine, static_mask=self.static_mask,
                                  moving_mask=self.moving_mask)

        return mapping