from new_api.NewETUtil.et_methods import *
from new_api.NewETUtil.new_utils import Utils


class ETModel(Utils):
    """
    comparison of different ET models
    """
    def __init__(self, methods, input_df, units, constants, **kwargs):
        self.methods = methods
        super(ETModel, self).__init__(input_df.copy(),
                                      units.copy(),
                                      constants.copy(),
                                      **kwargs)
