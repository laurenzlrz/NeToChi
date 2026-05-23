"""Tests for MosaicHardwareConfig distance and slice logic."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig

from tests.input_generator.helpers import make_hw_config, slice_bounds_cover_core


class TestMosaicHardwareConfigValidation(unittest.TestCase):
    def test_rejects_slice_factor_greater_than_neurons_per_core(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            MosaicHardwareConfig(
                nodes_per_router=2,
                neurons_per_core=4,
                router_levels=1,
                slice_factor=8,
            )
        self.assertIn("slice_factor cannot be greater than neurons_per_core", str(ctx.exception))


class TestMosaicHardwareConfigComputedFields(unittest.TestCase):
    def setUp(self) -> None:
        self.hw = make_hw_config()

    def test_total_cores_and_neurons(self) -> None:
        self.assertEqual(self.hw.total_cores, 4)
        self.assertEqual(self.hw.total_neurons, 32)
        self.assertEqual(self.hw.max_distance, 2)


class TestCoreDistance(unittest.TestCase):
    """core_distance for distances 0, 1, and 2."""

    def setUp(self) -> None:
        self.hw = make_hw_config(router_levels=2)

    def test_same_core_has_distance_zero(self) -> None:
        for core in range(self.hw.total_cores):
            self.assertEqual(self.hw.core_distance(core, core), 0)

    def test_distance_one_between_sibling_cores(self) -> None:
        self.assertEqual(self.hw.core_distance(0, 1), 1)
        self.assertEqual(self.hw.core_distance(1, 0), 1)
        self.assertEqual(self.hw.core_distance(2, 3), 1)
        self.assertEqual(self.hw.core_distance(3, 2), 1)

    def test_distance_two_between_cousin_cores(self) -> None:
        self.assertEqual(self.hw.core_distance(0, 2), 2)
        self.assertEqual(self.hw.core_distance(0, 3), 2)
        self.assertEqual(self.hw.core_distance(1, 2), 2)
        self.assertEqual(self.hw.core_distance(1, 3), 2)


class TestNumSlicesAtDistance(unittest.TestCase):
    def setUp(self) -> None:
        self.hw = make_hw_config(neurons_per_core=8, slice_factor=2, router_levels=2)

    def test_distance_zero_single_slice(self) -> None:
        self.assertEqual(self.hw.num_slices_at_distance(0), 1)

    def test_distance_one_two_slices(self) -> None:
        self.assertEqual(self.hw.num_slices_at_distance(1), 2)

    def test_distance_two_four_slices(self) -> None:
        self.assertEqual(self.hw.num_slices_at_distance(2), 4)


class TestGetSliceBounds(unittest.TestCase):
    def setUp(self) -> None:
        self.hw = make_hw_config(neurons_per_core=8, slice_factor=2, router_levels=2)

    def test_distance_zero_covers_entire_core(self) -> None:
        self.assertEqual(self.hw.get_slice_bounds(0, 0), (0, 8))

    def test_distance_one_halves_core(self) -> None:
        self.assertEqual(self.hw.get_slice_bounds(1, 0), (0, 4))
        self.assertEqual(self.hw.get_slice_bounds(1, 1), (4, 8))

    def test_distance_two_quarters_core(self) -> None:
        expected = [(0, 2), (2, 4), (4, 6), (6, 8)]
        actual = [
            self.hw.get_slice_bounds(2, idx)
            for idx in range(self.hw.num_slices_at_distance(2))
        ]
        self.assertEqual(actual, expected)

    def test_slice_partitions_are_contiguous_for_distances_0_1_2(self) -> None:
        for distance in (0, 1, 2):
            with self.subTest(distance=distance):
                slice_bounds_cover_core(self.hw, distance)


class TestIsValidConnection(unittest.TestCase):
    def setUp(self) -> None:
        self.hw = make_hw_config(neurons_per_core=8, router_levels=2)

    def test_same_core_distance_zero_accepts_any_local_except_out_of_range(self) -> None:
        core = 0
        for local_addr in range(self.hw.neurons_per_core):
            self.assertTrue(
                self.hw.is_valid_connection(core, core, local_addr, target_slice_idx=0)
            )
        self.assertFalse(self.hw.is_valid_connection(core, core, 8, target_slice_idx=0))

    def test_distance_one_only_accepts_addresses_in_chosen_half(self) -> None:
        source_core, target_core = 0, 1
        self.assertTrue(self.hw.is_valid_connection(source_core, target_core, 0, 0))
        self.assertTrue(self.hw.is_valid_connection(source_core, target_core, 3, 0))
        self.assertFalse(self.hw.is_valid_connection(source_core, target_core, 4, 0))
        self.assertTrue(self.hw.is_valid_connection(source_core, target_core, 4, 1))
        self.assertFalse(self.hw.is_valid_connection(source_core, target_core, 3, 1))

    def test_distance_two_only_accepts_addresses_in_chosen_quarter(self) -> None:
        source_core, target_core = 0, 2
        self.assertTrue(self.hw.is_valid_connection(source_core, target_core, 1, 0))
        self.assertFalse(self.hw.is_valid_connection(source_core, target_core, 2, 0))
        self.assertTrue(self.hw.is_valid_connection(source_core, target_core, 5, 2))


if __name__ == "__main__":
    unittest.main()
