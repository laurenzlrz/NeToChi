"""
Building blocks of the TheoreticalBoundsMapper (see netochi/mapping/theoretical_bounds_mapper.py).

The helpers in this subpackage are pure numpy / scipy / cvxpy code and do not use graph-tool:
  - sdp_partition:  balanced-partition SDP and its rounding to exactly balanced blocks
  - tree_recovery:  block statistics, recursive balanced bisection, class co-occurrence counts
  - cell_recovery:  the full recovery pipeline (cores, router tree, prefixes, address blocks)
"""
