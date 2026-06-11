
INEQUAL_ASSIGNMENT_OBJECTS = "The hardware configuration in the assignment does not match the hardware configuration in the input."

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
JOINT_NUM_MOVE_TYPES = 7  # swap, slice, add_core, remove_core, change_nc, change_nr, change_l
JOINT_P_SWAP = 0.35
JOINT_P_SLICE = 0.35
JOINT_P_ADD_CORE = 0.07
JOINT_P_REMOVE_CORE = 0.07
JOINT_P_NC = 0.06
JOINT_P_NR = 0.05
JOINT_P_L = 0.05
JOINT_P_SPLIT = 0.5  # Bernoulli p for neuron redistribution on split
RISSANEN_C0 = 2.865064
JOINT_MAX_L = 10  # Preallocation size for slice assignments

# Joint Inference Debug Messages
DEBUG_JOINT_RUN_START = "DEBUG: Joint Inference Mapper: Starting run with K_range=[{k_min}, {k_max}]..."
DEBUG_JOINT_SWEEP_CALL = "DEBUG: Joint State: mcmc_sweep() called with beta={beta}, K={k}, Nc={nc}, Nr={nr}, L={l}"
DEBUG_JOINT_CORE_ADDED = "DEBUG: Joint State: Core added (K={k_old} -> {k_new})"
DEBUG_JOINT_CORE_REMOVED = "DEBUG: Joint State: Core removed (K={k_old} -> {k_new})"
DEBUG_JOINT_NC_CHANGED = "DEBUG: Joint State: Nc changed (Nc={nc_old} -> {nc_new})"
DEBUG_JOINT_NR_CHANGED = "DEBUG: Joint State: Nr changed (Nr={nr_old} -> {nr_new})"
DEBUG_JOINT_L_CHANGED = "DEBUG: Joint State: L changed (L={l_old} -> {l_new})"
DEBUG_JOINT_RESTORE_BEST = "DEBUG: Joint State: Restoring best state (K={k}, Nc={nc}, Nr={nr}, L={l}, energy={energy:.4f})"
