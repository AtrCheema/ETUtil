from .et_methods import *
from .utils import Utils


class ETModel(Utils):
    """
    comparison of different ET models
    """
    def __init__(self, methods, data, units, constants, **kwargs):
        self.methods = methods
        super(ETModel, self).__init__(data.copy(),
                                      units.copy(),
                                      constants.copy(),
                                      **kwargs)
