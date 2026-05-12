"""
Constants for the neuromorphic mapping pipeline.
"""

# Timeouts and Iterations
DEFAULT_TIME_LIMIT_S = 600.0

# Logging Formats and Messages
PIPELINE_LOG_FORMAT = "[{mapper}] {graph_type}: LL={ll:.2f}, Time={elapsed:.3f}s"
MSG_TASK_HEADER = "\n--- Task: {mapper} on {graph_type} (Nodes={nodes}){baseline_info} ---"
MSG_WITH_BASELINE = " (with baseline)"
MSG_MAPPER_FAILED = "Mapper failed, cannot evaluate"

# Metadata Keys
KEY_GRAPH_TYPE = "graph_type"
KEY_NODES = "nodes"
KEY_UNKNOWN = "Unknown"

# Default Metric Values
DEFAULT_METRIC_VALUE = -1.0
DEFAULT_REL_METRIC_VALUE = 0.0

# Reporting Constants
REPORT_DIVIDER = "=" * 100
REPORT_SUBDIVIDER = "-" * 100
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

# Formatting Strings
TABLE_ROW_REL_FORMAT = "{mapper:<25} | {graph_type:<20} | {rel_ll:<15.2f} | {rel_inc:<10.0f} | {rel_hw:<12.2f} | {elapsed:<10.3f}"
TABLE_ROW_RAW_FORMAT = "{mapper:<25} | {graph_type:<20} | {raw_ll:<15.2f} | {raw_inc:<10.0f} | {raw_hw:<12.2f} | {elapsed:<10.3f}"
TABLE_HEADER_REL_FORMAT = f"{TABLE_COL_MAPPER:<25} | {TABLE_COL_GRAPH:<20} | {TABLE_COL_REL_LL:<15} | {TABLE_COL_REL_INC:<10} | {TABLE_COL_REL_HW:<12} | {TABLE_COL_TIME:<10}"
TABLE_HEADER_RAW_FORMAT = f"{TABLE_COL_MAPPER:<25} | {TABLE_COL_GRAPH:<20} | {TABLE_COL_RAW_LL:<15} | {TABLE_COL_RAW_INC:<10} | {TABLE_COL_RAW_HW:<12} | {TABLE_COL_TIME:<10}"
