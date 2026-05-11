# Error Message Constants for Mapping
CORE_ASSIGNMENT_NOT_1D = "neuron_core_idxs_assignment must be a 1D array."
CORE_ASSIGNMENT_SIZE_MISMATCH = "Core assignment size ({actual}) does not match graph vertices ({expected})."
SLICE_ASSIGNMENT_NOT_2D = "neuron_slice_assignments must be a 2D array."
SLICE_ASSIGNMENT_ROWS_MISMATCH = "Slice assignment row count must match number of neurons."
SLICE_ASSIGNMENT_COLS_MISMATCH = "Slice assignment columns ({actual}) must match router levels + 1 ({expected})."
CORE_INDEX_OUT_OF_RANGE = "One or more core indices are out of valid hardware range (0 to {max_cores})."
LOCAL_INDEX_NOT_1D = "neuron_local_idxs_assignment must be a 1D array."
LOCAL_INDEX_SIZE_MISMATCH = "Local index assignment size must match number of neurons."
LOCAL_INDEX_OUT_OF_RANGE = "One or more local indices are out of valid range."
INVALID_MAPPING_INPUT_TYPE = "Invalid mapping input type provided to mapper."
HARDWARE_CAPACITY_EXCEEDED = "Hardware capacity exceeded: Could not assign node {node} (Total neurons: {total}, Capacity: {capacity})."

# MCMC Mapper Constants
MCMC_TIME_LIMIT_S = 10.0
MCMC_DEFAULT_ITERATIONS = 200
MCMC_DEFAULT_INITIAL_TEMP = 5.0
MCMC_DEFAULT_SEED = 42
MCMC_BETA_MULTIPLIER = 1000.0
MCMC_NUM_MOVE_TYPES = 2

# Debug Messages
DEBUG_MCMC_RUN_START = "DEBUG: MCMC Mapper: Starting run..."
DEBUG_MCMC_ENTROPY_CALL = "DEBUG: MCMC State: entropy() called"
DEBUG_MCMC_SWEEP_CALL = "DEBUG: MCMC State: mcmc_sweep() called with beta={beta}"
DEBUG_MCMC_RESTORE_BEST = "DEBUG: MCMC State: Restoring best state..."

# Joint Inference Constants
JOINT_NUM_MOVE_TYPES = 4  # swap, slice, add_core, remove_core
JOINT_P_SWAP = 0.45
JOINT_P_SLICE = 0.45
JOINT_P_ADD_CORE = 0.05
JOINT_P_REMOVE_CORE = 0.05
JOINT_P_SPLIT = 0.5  # Bernoulli p for neuron redistribution on split
RISSANEN_C0 = 2.865064

# Joint Inference Debug Messages
DEBUG_JOINT_RUN_START = "DEBUG: Joint Inference Mapper: Starting run with K_range=[{k_min}, {k_max}]..."
DEBUG_JOINT_SWEEP_CALL = "DEBUG: Joint State: mcmc_sweep() called with beta={beta}, K={k}"
DEBUG_JOINT_CORE_ADDED = "DEBUG: Joint State: Core added (K={k_old} -> {k_new})"
DEBUG_JOINT_CORE_REMOVED = "DEBUG: Joint State: Core removed (K={k_old} -> {k_new})"
DEBUG_JOINT_RESTORE_BEST = "DEBUG: Joint State: Restoring best state (K={k}, energy={energy:.4f})"
