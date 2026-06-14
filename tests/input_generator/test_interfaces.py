"""Tests for input_generator interface models and validators."""

from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

from netochi.definitions.exceptions import DimensionError, InvalidAssignmentError
from netochi.input_generator.interfaces import MosaicAssignment, MosaicMappingInput

from tests.input_generator.helpers import (
    assignment_with_core,
    assignment_with_fewer_slice_columns,
    assignment_with_local_idx,
    assignment_with_slice_value,
    assignment_with_truncated_neurons,
    make_hw_config,
    make_mapping_input,
    make_valid_assignment,
    max_slice_index,
)


class TestFrozenInterfaceModels(unittest.TestCase):
    """Interface models are frozen; mutations must go through model_copy or new instances."""

    def setUp(self) -> None:
        self.hw = make_hw_config()
        self.valid = make_valid_assignment(self.hw)

    def test_mosaic_assignment_rejects_field_reassignment(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            self.valid.neuron_core_pre_assignment = np.zeros(1, dtype=np.int64)
        self.assertIn("frozen_instance", str(ctx.exception))

    def test_mosaic_assignment_rejects_setattr(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            self.valid.__setattr__(
                "neuron_slice_assignment",
                self.valid.neuron_slice_assignment[:, :-1],
            )
        self.assertIn("frozen_instance", str(ctx.exception))

    def test_model_copy_update_produces_independent_instance(self) -> None:
        modified = assignment_with_core(self.valid, neuron_idx=0, core=99)
        self.assertEqual(modified.neuron_core_pre_assignment[0], 99)
        self.assertEqual(self.valid.neuron_core_pre_assignment[0], 0)

    def test_mosaic_mapping_input_rejects_assignment_reassignment(self) -> None:
        mapping_input = make_mapping_input(self.hw, assignment=self.valid)
        other = make_valid_assignment(self.hw)
        with self.assertRaises(ValidationError) as ctx:
            mapping_input.assignment = other
        self.assertIn("frozen_instance", str(ctx.exception))


class TestMosaicAssignmentValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.hw = make_hw_config()
        self.valid = make_valid_assignment(self.hw)

    def test_accepts_valid_assignment(self) -> None:
        self.assertEqual(self.valid.neuron_core_pre_assignment.size, self.hw.total_neurons)

    def test_rejects_non_1d_core_assignment(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            MosaicAssignment(
                neuron_core_pre_assignment=np.zeros((2, 2), dtype=np.int64),
                neuron_idx_pre_assignment=self.valid.neuron_idx_pre_assignment,
                neuron_slice_assignment=self.valid.neuron_slice_assignment,
            )
        self.assertIn("neuron_core_pre_assignment must be a 1D array", str(ctx.exception))

    def test_rejects_non_1d_idx_assignment(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            MosaicAssignment(
                neuron_core_pre_assignment=self.valid.neuron_core_pre_assignment,
                neuron_idx_pre_assignment=np.zeros((2, 2), dtype=np.int64),
                neuron_slice_assignment=self.valid.neuron_slice_assignment,
            )
        self.assertIn("neuron_idx_pre_assignment must be a 1D array", str(ctx.exception))

    def test_rejects_non_2d_slice_assignment(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            MosaicAssignment(
                neuron_core_pre_assignment=self.valid.neuron_core_pre_assignment,
                neuron_idx_pre_assignment=self.valid.neuron_idx_pre_assignment,
                neuron_slice_assignment=np.zeros(self.hw.total_neurons, dtype=np.int64),
            )
        self.assertIn("neuron_slice_assignment must be a 2D array", str(ctx.exception))

    def test_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            MosaicAssignment(
                neuron_core_pre_assignment=np.zeros(3, dtype=np.int64),
                neuron_idx_pre_assignment=np.zeros(2, dtype=np.int64),
                neuron_slice_assignment=np.zeros((3, self.hw.router_levels + 1), dtype=np.int64),
            )
        self.assertIn("must have the same length", str(ctx.exception))

    def test_rejects_slice_rows_not_matching_pre_assignments(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            MosaicAssignment(
                neuron_core_pre_assignment=np.zeros(4, dtype=np.int64),
                neuron_idx_pre_assignment=np.zeros(4, dtype=np.int64),
                neuron_slice_assignment=np.zeros((3, self.hw.router_levels + 1), dtype=np.int64),
            )
        self.assertIn("First dimension of neuron_slice_assignment", str(ctx.exception))


class TestMosaicMappingInputValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.hw = make_hw_config()
        self.valid_assignment = make_valid_assignment(self.hw)

    def test_accepts_none_assignment(self) -> None:
        mapping_input = make_mapping_input(self.hw, assignment=None)
        self.assertIsNone(mapping_input.assignment)

    def test_accepts_valid_assignment_on_construction(self) -> None:
        mapping_input = make_mapping_input(self.hw, assignment=self.valid_assignment)
        self.assertIsNotNone(mapping_input.assignment)

    def test_rejects_invalid_assignment_on_construction(self) -> None:
        bad = assignment_with_core(self.valid_assignment, neuron_idx=0, core=999)
        with self.assertRaises(ValidationError) as ctx:
            make_mapping_input(self.hw, assignment=bad)
        self.assertIn("exceeds hardware limits", str(ctx.exception))


class TestMosaicMappingInputVerifyAssignment(unittest.TestCase):
    """Direct tests for verify_assignment (used by later pipeline stages)."""

    def setUp(self) -> None:
        self.hw = make_hw_config()
        self.mapping_input = make_mapping_input(self.hw, assignment=None)
        self.valid_assignment = make_valid_assignment(self.hw)

    def test_verify_assignment_accepts_valid_data(self) -> None:
        self.mapping_input.verify_assignment(self.valid_assignment)

    def test_rejects_wrong_number_of_neurons(self) -> None:
        bad = assignment_with_truncated_neurons(self.valid_assignment)
        with self.assertRaises(DimensionError) as ctx:
            self.mapping_input.verify_assignment(bad)
        self.assertIn("must match total neurons", str(ctx.exception))

    def test_rejects_invalid_core_index(self) -> None:
        bad = assignment_with_core(
            self.valid_assignment,
            neuron_idx=0,
            core=self.hw.total_cores,
        )
        with self.assertRaises(InvalidAssignmentError) as ctx:
            self.mapping_input.verify_assignment(bad)
        self.assertIn("exceeds hardware limits", str(ctx.exception))

        bad = assignment_with_core(
            self.valid_assignment,
            neuron_idx=0,
            core=-1,
        )
        with self.assertRaises(InvalidAssignmentError) as ctx:
            self.mapping_input.verify_assignment(bad)
        self.assertIn("exceeds hardware limits", str(ctx.exception))

    def test_rejects_invalid_local_neuron_index(self) -> None:
        bad = assignment_with_local_idx(
            self.valid_assignment,
            neuron_idx=0,
            local_idx=self.hw.neurons_per_core,
        )
        with self.assertRaises(InvalidAssignmentError) as ctx:
            self.mapping_input.verify_assignment(bad)
        self.assertIn("exceeds hardware limits", str(ctx.exception))

    def test_rejects_wrong_slice_column_count(self) -> None:
        bad = assignment_with_fewer_slice_columns(self.valid_assignment)
        with self.assertRaises(DimensionError) as ctx:
            self.mapping_input.verify_assignment(bad)
        self.assertIn("router levels", str(ctx.exception))

    def test_rejects_negative_slice_index(self) -> None:
        bad = assignment_with_slice_value(self.valid_assignment, neuron_idx=0, level=0, value=-1)
        with self.assertRaises(InvalidAssignmentError) as ctx:
            self.mapping_input.verify_assignment(bad)
        self.assertIn("invalid assignment", str(ctx.exception))

    def test_rejects_slice_index_above_max_for_distance(self) -> None:
        """Per level, valid slice indices are 0 .. slice_factor^level - 1."""
        for level in range(self.hw.router_levels + 1):
            max_idx = max_slice_index(self.hw, level)
            invalid_value = max_idx + 1
            with self.subTest(level=level, max_idx=max_idx, invalid_value=invalid_value):
                bad = assignment_with_slice_value(
                    self.valid_assignment,
                    neuron_idx=0,
                    level=level,
                    value=invalid_value,
                )
                with self.assertRaises(InvalidAssignmentError, msg=f"No exception for level {level}") as ctx:
                    self.mapping_input.verify_assignment(bad)
                message = str(ctx.exception)
                self.assertIn("Maximum allowed for this slice", message, f"Error message should indicate the maximum allowed index for level {level}")
                self.assertIn(str(max_idx), message, f"Error message should include the maximum allowed index {max_idx} for level {level}")
                self.assertIn(f"{self.hw.slice_factor}^{level} - 1", message, f"Error message should indicate the formula for maximum allowed index for level {level}")


if __name__ == "__main__":
    unittest.main()
