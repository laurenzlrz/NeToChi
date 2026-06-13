
INEQUAL_ASSIGNMENT_OBJECTS = "The hardware configuration in the assignment does not match the hardware configuration in the input."
TOO_MANY_NEURONS = "Graph has more neurons than the hardware can accommodate."

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
DEFAULT_TIME_LIMIT_S = 600.0
PIPELINE_LOG_FORMAT = "[{mapper}] {graph_type}: LL={ll:.2f}, Time={elapsed:.3f}s"
MSG_TASK_HEADER = "\n--- Task: {mapper} on {graph_type} (Nodes={nodes}){baseline_info} ---"
MSG_WITH_BASELINE = " (with baseline)"
MSG_MAPPER_FAILED = "Mapper failed, cannot evaluate"
KEY_GRAPH_TYPE = "graph_type"
KEY_NODES = "nodes"
KEY_UNKNOWN = "Unknown"
DEFAULT_METRIC_VALUE = -1.0
DEFAULT_REL_METRIC_VALUE = 0.0
REPORT_DIVIDER = "=" * 130
REPORT_SUBDIVIDER = "-" * 130
REPORT_HEADER_BASELINE = "SUMMARY REPORT: BASELINE COMPARISONS (Relative Improvement)"
REPORT_HEADER_PURE = "SUMMARY REPORT: PURE RESULTS (Absolute Values)"
TABLE_COL_MAPPER = "Mapper"
TABLE_COL_GRAPH = "Graph"
TABLE_COL_TIME = "Time (s)"
TABLE_COL_REL_LL = "Rel-Likelihood"
TABLE_COL_REL_INC = "Rel-Incons"
TABLE_COL_REL_HW = "Rel-HW-Size"
TABLE_COL_RAW_LL = "Raw-Likelihood"
TABLE_COL_RAW_INC = "Raw-Incons"
TABLE_COL_RAW_HW = "Raw-HW-Size"
TABLE_ROW_REL_FORMAT = "{mapper:<45} | {graph_type:<20} | {rel_ll:<15.2f} | {rel_inc:<10.0f} | {rel_hw:<12.2f} | {elapsed:<10.3f}"
TABLE_ROW_RAW_FORMAT = "{mapper:<45} | {graph_type:<20} | {raw_ll:<15.2f} | {raw_inc:<10.0f} | {raw_hw:<12.2f} | {elapsed:<10.3f}"
TABLE_HEADER_REL_FORMAT = f"{TABLE_COL_MAPPER:<45} | {TABLE_COL_GRAPH:<20} | {TABLE_COL_REL_LL:<15} | {TABLE_COL_REL_INC:<10} | {TABLE_COL_REL_HW:<12} | {TABLE_COL_TIME:<10}"
TABLE_HEADER_RAW_FORMAT = f"{TABLE_COL_MAPPER:<45} | {TABLE_COL_GRAPH:<20} | {TABLE_COL_RAW_LL:<15} | {TABLE_COL_RAW_INC:<10} | {TABLE_COL_RAW_HW:<12} | {TABLE_COL_TIME:<10}"
OBJ_NAME_LL = "LogLikelihoodObjective"
OBJ_NAME_INCONSISTENCY = "InconsistencyObjective"
OBJ_NAME_HW_SIZE = "MosaicHardwareSizeObjective"
LL_INVALID_PENALTY_LOG = -23.02585092994046  # log(1e-10)
LL_LAPLACIAN_SMOOTHING_NUM = 1.0
LL_LAPLACIAN_SMOOTHING_DEN = 2.0
DEFAULT_HW_WEIGHT = 1.0
DEFAULT_INCONSISTENCY_WEIGHT = 1.0
