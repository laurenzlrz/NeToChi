"""
Constants for the neuromorphic mapping objectives.
"""

# Objective Names (Keys for Reporting)
OBJ_NAME_LL = "LogLikelihoodObjective"
OBJ_NAME_INCONSISTENCY = "InconsistencyObjective"
OBJ_NAME_HW_SIZE = "MosaicHardwareSizeObjective"

# Log-Likelihood Constants
LL_INVALID_PENALTY_LOG = -23.02585092994046  # log(1e-10)
LL_LAPLACIAN_SMOOTHING_NUM = 1.0
LL_LAPLACIAN_SMOOTHING_DEN = 2.0

# Hardware Cost Constants
DEFAULT_HW_WEIGHT = 1.0

# Inconsistency Constants
DEFAULT_INCONSISTENCY_WEIGHT = 1.0
